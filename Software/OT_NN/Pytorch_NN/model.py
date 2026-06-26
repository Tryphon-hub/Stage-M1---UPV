# model.py
import torch
import torch.nn as nn


# ─── Basic blocks ────────────────────────────────────────────────────────────

class MultipleConv(nn.Module):
    """
    A stack of N_conv (Conv2d 3x3 → BatchNorm → ReLU) layers at one U-Net level.
    The first conv maps in_channels → out_channels; the rest keep out_channels.
    """
    def __init__(self, in_channels, out_channels, N_conv=3):
        """
        Parameters
        ----------
        in_channels  : int — input channels.
        out_channels : int — output channels (kept after the first conv).
        N_conv       : int — number of conv-bn-relu layers in the block.
        """
        super().__init__()

        layers = []
        for k in range(N_conv):
            c_in = in_channels if k == 0 else out_channels   # first conv: in→out, others: out→out
            layers += [
                nn.Conv2d(c_in, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Apply the conv-bn-relu stack to `x`."""
        return self.block(x)


class Down(nn.Module):
    """Encoder step: 2x2 max-pooling (halves H, W) followed by a MultipleConv."""
    def __init__(self, in_channels, out_channels, N_conv=3):
        """
        Parameters
        ----------
        in_channels  : int — input channels.
        out_channels : int — output channels after the conv block.
        N_conv       : int — number of conv layers in the block.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            MultipleConv(in_channels, out_channels, N_conv=N_conv)
        )

    def forward(self, x):
        """Downsample then convolve `x`."""
        return self.block(x)


class Up(nn.Module):
    """
    Decoder step: transposed-conv upsampling (doubles H, W and halves channels),
    concatenation with the encoder skip connection, then a MultipleConv.
    """
    def __init__(self, in_channels, out_channels, N_conv=3):
        """
        Parameters
        ----------
        in_channels  : int — channels of the deeper feature map (before upsampling).
        out_channels : int — output channels after the conv block.
        N_conv       : int — number of conv layers in the block.
        """
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = MultipleConv(in_channels, out_channels, N_conv=N_conv)

    def forward(self, x, skip):
        """Upsample `x`, concatenate the `skip` feature map, then convolve."""
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─── CBAM module ─────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    CBAM stage 1: which channels matter?
    Global AvgPool + MaxPool → shared MLP → per-channel weight vector [C, 1, 1].
    `ratio` is the MLP bottleneck compression factor (default 8).
    """
    def __init__(self, channels, ratio=8):
        """
        Parameters
        ----------
        channels : int — number of feature channels.
        ratio    : int — channel-reduction factor inside the shared MLP.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Return per-channel gates in [0, 1], shape [B, C, 1, 1]."""
        avg = self.mlp(self.avg_pool(x))   # [B, C]
        mx  = self.mlp(self.max_pool(x))   # [B, C]  — shared MLP
        out = self.sigmoid(avg + mx)       # [B, C]
        return out.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]


class SpatialAttention(nn.Module):
    """
    CBAM stage 2: where to look in the 2D map?
    Channel-wise Avg + Max → 7x7 conv → single-channel spatial map [B, 1, H, W].
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Return a spatial gate in [0, 1], shape [B, 1, H, W]."""
        avg = x.mean(dim=1, keepdim=True)       # [B, 1, H, W]
        mx, _ = x.max(dim=1, keepdim=True)      # [B, 1, H, W]
        out = torch.cat([avg, mx], dim=1)        # [B, 2, H, W]
        return self.sigmoid(self.conv(out))      # [B, 1, H, W]


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module: sequentially applies channel attention
    then spatial attention, re-weighting the feature map along each axis.
    """
    def __init__(self, channels, ratio=8):
        """
        Parameters
        ----------
        channels : int — number of feature channels.
        ratio    : int — channel-attention reduction ratio.
        """
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        """Apply channel then spatial attention to `x`."""
        x = x * self.channel_att(x)   # weight the channels
        x = x * self.spatial_att(x)   # weight the spatial positions
        return x


# ─── Main U-Net ──────────────────────────────────────────────────────────────

class UNetTopo(nn.Module):
    """
    U-Net for 2D topology optimization (density + tractions → stress fields).

    Parameters
    ----------
    nif       : number of filters at level 0 (main capacity hyperparameter)
    n_in      : input channels  — ρ, tx, ty (, border_mask)
    n_out     : output channels — σx, σy, τxy = 3
    use_cbam  : enable the CBAM module at the bottleneck

    Architecture (with nif=32, input 32x32)
    ----------------------------------------
    Level 0   : 32x32,  nif    = 32  filters
    Level 1   : 16x16,  nifx2  = 64  filters
    Level 2   :  8x8,   nifx4  = 128 filters
    Level 3   :  4x4,   nifx8  = 256 filters
    Bottleneck:  2x2,   nifx16 = 512 filters  ← CBAM
    """
    def __init__(self, nif=32, n_in=4, n_out=3, use_cbam=True, N_conv=3):
        """
        Parameters
        ----------
        nif      : int  — base number of filters (level 0).
        n_in     : int  — number of input channels.
        n_out    : int  — number of output channels.
        use_cbam : bool — insert a CBAM block at the bottleneck.
        N_conv   : int  — conv layers per level.
        """
        super().__init__()
        self.use_cbam = use_cbam

        f = nif  # short alias

        # Encoder — 4 levels as in the reference paper
        self.inc   = MultipleConv(n_in, f, N_conv=N_conv)          # 32x32 → f
        self.down1 = Down(f,     f * 2, N_conv=N_conv)           # 16x16 → fx2
        self.down2 = Down(f * 2, f * 4, N_conv=N_conv)           #  8x8  → fx4
        self.down3 = Down(f * 4, f * 8, N_conv=N_conv)           #  4x4  → fx8

        # Bottleneck
        self.bottleneck = Down(f * 8, f * 16, N_conv=N_conv)     #  2x2  → fx16
        if use_cbam:
            self.cbam = CBAM(f * 16)

        # Decoder — symmetric to the encoder
        self.up1 = Up(f * 16, f * 8, N_conv=N_conv)             #  4x4  → fx8
        self.up2 = Up(f * 8,  f * 4, N_conv=N_conv)             #  8x8  → fx4
        self.up3 = Up(f * 4,  f * 2, N_conv=N_conv)             # 16x16 → fx2
        self.up4 = Up(f * 2,  f, N_conv=N_conv)                 # 32x32 → f

        # Output head — linear (no activation: stresses are unbounded)
        self.outc = nn.Conv2d(f, n_out, kernel_size=1)

    def forward(self, x):
        """
        Map an input image stack to the three stress fields.

        Parameters
        ----------
        x : torch.Tensor [B, n_in, 32, 32] — ρ + traction channels.

        Returns
        -------
        torch.Tensor [B, 3, 32, 32] — predicted (σx, σy, τxy).
        """
        # Encoder
        x1 = self.inc(x)       # [B, f,    32, 32]
        x2 = self.down1(x1)    # [B, fx2,  16, 16]
        x3 = self.down2(x2)    # [B, fx4,   8,  8]
        x4 = self.down3(x3)    # [B, fx8,   4,  4]

        # Bottleneck (+ optional attention)
        xb = self.bottleneck(x4)          # [B, fx16,  2,  2]
        if self.use_cbam:
            xb = self.cbam(xb)

        # Decoder (with skip connections)
        x = self.up1(xb, x4)  # [B, fx8,   4,  4]
        x = self.up2(x,  x3)  # [B, fx4,   8,  8]
        x = self.up3(x,  x2)  # [B, fx2,  16, 16]
        x = self.up4(x,  x1)  # [B, f,    32, 32]

        return self.outc(x)    # [B, 3,    32, 32] — σx, σy, τxy
    


# ─── Boundary-Embedding ──────────────────────────────────────────────────────

class BoundaryEmbedding(nn.Module):
    """
    Embed the boundary tractions directly into the U-Net latent space to better
    guide the reconstruction.

    Idea: instead of feeding the tractions (tx, ty) as input channels (which can
    be harder to learn), concatenate an embedding of them at the U-Net
    bottleneck — after the encoder convolutions, before the CBAM and the
    decoder. The network thus "sees" the boundary loads exactly when it has to
    reconstruct the stresses, and can use them as cues.

    Parameters
    ----------
    in_channels : int
        Input dimension (e.g. 16 = 2 components x 8 nodes).
    hidden_layers : list[int]
        Hidden-layer sizes of the MLP (e.g. [32, 64]).
    out_channels : int
        MLP output dimension, typically the number of channels added at the
        U-Net bottleneck.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_layers: list[int],
                 out_channels: int):
        super().__init__()

        layers = []
        prev_dim = in_channels

        # Hidden layers (Linear + ReLU)
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        # Output layer (no ReLU)
        layers.append(nn.Linear(prev_dim, out_channels))

        self.fclayers = nn.Sequential(*layers)

        # Upsample [B, C, 1, 1] -> [B, C, 2, 2] to match the bottleneck spatial size
        self.upsample = nn.ConvTranspose2d(
            out_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        """
        Embed the nodal tractions into a small spatial feature map.

        Parameters
        ----------
        x : torch.Tensor [B, in_channels] — flattened nodal tractions.

        Returns
        -------
        torch.Tensor [B, out_channels, 2, 2] — bottleneck-sized embedding.
        """
        e = self.fclayers(x)                # [B, out_channels]
        e = e.unsqueeze(-1).unsqueeze(-1)   # [B, out_channels, 1, 1]
        e = self.upsample(e)                # [B, out_channels, 2, 2]
        return e

# ─── Main BE-UNet ────────────────────────────────────────────────────────────

class BE_UNetTopo(nn.Module):
    """
    U-Net for 2D topology optimization with a Boundary Embedding.

    Same backbone as UNetTopo, but the tractions are injected as an embedding
    concatenated at the bottleneck (see BoundaryEmbedding) instead of as input
    channels, so the input is the density field alone.

    Parameters
    ----------
    nif               : number of filters at level 0 (main hyperparameter)
    n_in              : input channels  — ρ only = 1
    n_out             : output channels — σx, σy, τxy = 3
    use_cbam          : enable the CBAM module at the bottleneck
    hidden_layers_MLP : hidden-layer sizes of the embedding MLP
    embed_out         : embedding dimension (channels added at the bottleneck)

    Architecture (with nif=32, input 32x32)
    ----------------------------------------
    Level 0    : 32x32,  nif    = 32  filters
    Level 1    : 16x16,  nif×2  = 64  filters
    Level 2    :  8x8,   nif×4  = 128 filters
    Level 3    :  4x4,   nif×8  = 256 filters
    Bottleneck :  2x2,   nif×16 = 512 filters
                  + embed_out          ← BoundaryEmbedding concat
                  → projected back to 512 filters before the (symmetric) decoder
    """
    def __init__(self, nif=32, n_in=3, n_out=3,
                use_cbam=True,
                hidden_layers_MLP = [32,64,128],
                embed_out=128,
                N_conv=3):
        """
        Parameters
        ----------
        nif               : int  — base number of filters (level 0).
        n_in              : int  — input channels (1 = density only).
        n_out             : int  — output channels (3 stress components).
        use_cbam          : bool — insert a CBAM block on the enriched bottleneck.
        hidden_layers_MLP : list[int] — embedding MLP hidden sizes.
        embed_out         : int  — embedding channels added at the bottleneck.
        N_conv            : int  — conv layers per level.
        """
        super().__init__()
        self.use_cbam = use_cbam
        f = nif

        # Encoder
        self.inc   = MultipleConv(n_in, f, N_conv=N_conv)
        self.down1 = Down(f,     f * 2, N_conv=N_conv)
        self.down2 = Down(f * 2, f * 4, N_conv=N_conv)
        self.down3 = Down(f * 4, f * 8, N_conv=N_conv)

        # Bottleneck
        self.bottleneck = Down(f * 8, f * 16, N_conv=N_conv)

        # Boundary embedding of the nodal tractions
        self.boundary_embedding = BoundaryEmbedding(
            in_channels  = 16,
            hidden_layers = hidden_layers_MLP,
            out_channels = embed_out
        )

        # Enriched bottleneck width (compute before using it)
        bottleneck_out = f * 16 + embed_out        # e.g. 512 + 64 = 576

        # CBAM on the enriched bottleneck channels
        if use_cbam:
            self.cbam = CBAM(bottleneck_out)

        # Project enriched → f*16 to restore the decoder's symmetry
        self.bottleneck_proj = nn.Conv2d(bottleneck_out, f * 16, kernel_size=1)

        # Decoder — symmetric; up1 receives f*16 after projection
        self.up1 = Up(f * 16, f * 8, N_conv=N_conv)
        self.up2 = Up(f * 8,  f * 4, N_conv=N_conv)
        self.up3 = Up(f * 4,  f * 2, N_conv=N_conv)
        self.up4 = Up(f * 2,  f, N_conv=N_conv)

        # Output head
        self.outc = nn.Conv2d(f, n_out, kernel_size=1)


    def forward(self, x, nodes=None):
        """
        Map a density field plus nodal tractions to the three stress fields.

        Parameters
        ----------
        x     : torch.Tensor [B, n_in, 32, 32] — density field.
        nodes : torch.Tensor [B, 16] — flattened nodal tractions for the embedding.

        Returns
        -------
        torch.Tensor [B, 3, 32, 32] — predicted (σx, σy, τxy).
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        xb = self.bottleneck(x4)              # [B, 512, 2, 2]

        # Boundary embedding — concatenated at the bottleneck
        e_T = self.boundary_embedding(nodes)  # [B, embed_out, 2, 2]
        xb  = torch.cat([xb, e_T], dim=1)    # [B, 512+embed_out, 2, 2]

        # CBAM on the enriched bottleneck
        if self.use_cbam:
            xb = self.cbam(xb)

        # Project back to 512 channels
        xb = self.bottleneck_proj(xb)         # [B, 512, 2, 2]

        # Decoder (with skip connections)
        x = self.up1(xb, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return self.outc(x)                   # [B, 3, 32, 32]