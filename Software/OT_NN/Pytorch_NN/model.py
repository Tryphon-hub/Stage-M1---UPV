# model.py
import torch
import torch.nn as nn


# ─── Blocs de base ───────────────────────────────────────────────────────────

class TripleConv(nn.Module):
    """3 conv par niveau"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            TripleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = TripleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─── Module CBAM ─────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Étape 1 du CBAM : quels canaux sont importants ?
    AvgPool + MaxPool globaux → MLP partagé → vecteur de poids Cx1x1.
    ratio : facteur de compression du MLP (hyperparamètre, 8 par défaut).
    """
    def __init__(self, channels, ratio=8):
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
        avg = self.mlp(self.avg_pool(x))   # [B, C]
        mx  = self.mlp(self.max_pool(x))   # [B, C]  — MLP partagé
        out = self.sigmoid(avg + mx)       # [B, C]
        return out.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]


class SpatialAttention(nn.Module):
    """
    Étape 2 du CBAM : où regarder dans la carte 2D ?
    Avg + Max sur les canaux → Conv 7x7 → carte HxW.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)       # [B, 1, H, W]
        mx, _ = x.max(dim=1, keepdim=True)      # [B, 1, H, W]
        out = torch.cat([avg, mx], dim=1)        # [B, 2, H, W]
        return self.sigmoid(self.conv(out))      # [B, 1, H, W]


class CBAM(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)   # pondère les canaux
        x = x * self.spatial_att(x)   # pondère les positions spatiales
        return x


# ─── U-Net principal ─────────────────────────────────────────────────────────

class UNetTopo(nn.Module):
    """
    U-Net pour optimisation topologique 2D.

    Paramètres
    ----------
    nif       : nombre de filtres au niveau 0 (hyperparamètre principal)
    n_in      : canaux d'entrée  — ρ, tx, ty, border_mask = 4
    n_out     : canaux de sortie — σx, σy, τxy = 3
    use_cbam  : active le module CBAM au bottleneck

    Architecture (avec nif=32, input 32x32)
    ----------------------------------------
    Niveau 0  : 32x32,  nif    = 32  filtres
    Niveau 1  : 16x16,  nifx2  = 64  filtres
    Niveau 2  :  8x8,   nifx4  = 128 filtres
    Niveau 3  :  4x4,   nifx8  = 256 filtres
    Bottleneck:  2x2,   nifx16 = 512 filtres  ← CBAM ici
    """
    def __init__(self, nif=32, n_in=4, n_out=3, use_cbam=True):
        super().__init__()
        self.use_cbam = use_cbam

        f = nif  # alias court

        # Encodeur — 4 niveaux comme dans l'article
        self.inc   = TripleConv(n_in, f)          # 32x32 → f
        self.down1 = Down(f,     f * 2)           # 16x16 → fx2
        self.down2 = Down(f * 2, f * 4)           #  8x8  → fx4
        self.down3 = Down(f * 4, f * 8)           #  4x4  → fx8

        # Bottleneck
        self.bottleneck = Down(f * 8, f * 16)     #  2x2  → fx16
        if use_cbam:
            self.cbam = CBAM(f * 16)

        # Décodeur — symétrique
        self.up1 = Up(f * 16, f * 8)             #  4x4  → fx8
        self.up2 = Up(f * 8,  f * 4)             #  8x8  → fx4
        self.up3 = Up(f * 4,  f * 2)             # 16x16 → fx2
        self.up4 = Up(f * 2,  f)                 # 32x32 → f

        # Tête de sortie — linéaire (pas d'activation : contraintes non bornées)
        self.outc = nn.Conv2d(f, n_out, kernel_size=1)

    def forward(self, x):
        # Encodeur
        x1 = self.inc(x)       # [B, f,    32, 32]
        x2 = self.down1(x1)    # [B, fx2,  16, 16]
        x3 = self.down2(x2)    # [B, fx4,   8,  8]
        x4 = self.down3(x3)    # [B, fx8,   4,  4]

        # Bottleneck
        xb = self.bottleneck(x4)          # [B, fx16,  2,  2]
        if self.use_cbam:
            xb = self.cbam(xb)

        # Décodeur
        x = self.up1(xb, x4)  # [B, fx8,   4,  4]
        x = self.up2(x,  x3)  # [B, fx4,   8,  8]
        x = self.up3(x,  x2)  # [B, fx2,  16, 16]
        x = self.up4(x,  x1)  # [B, f,    32, 32]

        return self.outc(x)    # [B, 3,    32, 32] — σx, σy, τxy
    


# ─── Boundary-Embedding ──────────────────────────────────────────────────────
class BoundaryEmbedding(nn.Module):
    """
    Embedding des tractions de bordure directement dans l'espace latent du U-Net, 
    pour mieux guider la reconstruction.
    Idée : concaténer les tractions de bordure (tx, ty) au bottleneck du U-Net,
    après les convolutions d'encodage, avant le CBAM et le décodage
    (au lieu de les fournir en entrée, ce qui peut être plus difficile à apprendre).
    Cela permet au réseau de "voir" les tractions de bordure 
    au moment où il doit reconstruire les contraintes, 
    et de les utiliser comme indices pour la reconstruction.


    Inputs: 
    - n1 : taille du premier layer du MLP (ex: 32)
    - out_channels : taille du second layer du MLP (ex: 64)
    """
    def __init__(self, in_channels, n1=32, out_channels=64):
        super().__init__()

        self.fclayers = nn.Sequential(
            nn.Linear(in_channels, n1),
            nn.ReLU(inplace=True),
            nn.Linear(n1, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels) # pas de Relu à la sortie, on veut des valeurs non bornées
            )
        
        # upscale 2x2 pour correspondre à la taille du bottleneck (2x2)
        self.upsample=nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2) 
        

    def forward(self, x):              # x : [B, 16]
        e = self.fclayers(x)                # [B, out_channels]
        e = e.unsqueeze(-1).unsqueeze(-1)   # [B, out_channels, 1, 1]
        e = self.upsample(e)           # [B, out_channels, 2, 2]
        return e




# ─── BE-UNet principal ─────────────────────────────────────────────────────────

class BE_UNetTopo(nn.Module):
    """
    U-Net pour optimisation topologique 2D.

    Paramètres
    ----------
    nif           : nombre de filtres au niveau 0 (hyperparamètre principal)
    n_in          : canaux d'entrée  — ρ seul = 1 
    n_out         : canaux de sortie — σx, σy, τxy = 3
    use_cbam      : active le module CBAM au bottleneck
    embed_n1      : taille du premier layer caché du MLP
    embed_out     : dimension de l'embedding (canaux ajoutés au bottleneck)

    Architecture (avec nif=32, input 32x32)
    ----------------------------------------
    Niveau 0   : 32x32,  nif    = 32  filtres
    Niveau 1   : 16x16,  nif×2  = 64  filtres
    Niveau 2   :  8x8,   nif×4  = 128 filtres
    Niveau 3   :  4x4,   nif×8  = 256 filtres
    Bottleneck :  2x2,   nif×16 = 512 filtres
                  + embed_out   = 64  filtres  ← BoundaryEmbedding concat ici
                  → total       = 576 filtres  ← CBAM ici
    """
    def __init__(self, nif=32, n_in=3, n_out=3,
                use_cbam=True, embed_n1=32, embed_out=64):
        super().__init__()
        self.use_cbam = use_cbam
        f = nif

        # Encodeur
        self.inc   = TripleConv(n_in, f)
        self.down1 = Down(f,     f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = Down(f * 8, f * 16)

        # BoundaryEmbedding
        self.boundary_embedding = BoundaryEmbedding(
            in_channels  = 16,
            n1           = embed_n1,
            out_channels = embed_out
        )

        # calcul de bottleneck_out AVANT de l'utiliser
        bottleneck_out = f * 16 + embed_out        # 512 + 64 = 576

        # CBAM sur les 576 canaux enrichis
        if use_cbam:
            self.cbam = CBAM(bottleneck_out)

        # Projection 576 → 512 pour restaurer la symétrie du décodeur
        self.bottleneck_proj = nn.Conv2d(bottleneck_out, f * 16, kernel_size=1)

        # Décodeur — symétrique, up1 reçoit f*16=512 après projection
        self.up1 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8,  f * 4)
        self.up3 = Up(f * 4,  f * 2)
        self.up4 = Up(f * 2,  f)

        # Tête de sortie
        self.outc = nn.Conv2d(f, n_out, kernel_size=1)


    def forward(self, x, nodes=None):
        # Encodeur
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        xb = self.bottleneck(x4)              # [B, 512, 2, 2]

        # BoundaryEmbedding — concat
        e_T = self.boundary_embedding(nodes)  # [B, 64,  2, 2]
        xb  = torch.cat([xb, e_T], dim=1)    # [B, 576, 2, 2]

        # CBAM sur les 576 canaux enrichis
        if self.use_cbam:
            xb = self.cbam(xb)

        # Projection 576 → 512
        xb = self.bottleneck_proj(xb)         # [B, 512, 2, 2]

        # Décodeur
        x = self.up1(xb, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return self.outc(x)                   # [B, 3, 32, 32]