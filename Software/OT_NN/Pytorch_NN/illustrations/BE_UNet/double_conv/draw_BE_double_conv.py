import sys
import os
import re
import shutil
from pathlib import Path

sys.path.insert(0, r'C:\Users\maxen\Documents\Stage\PlotNeuralNet')
from pycore.tikzeng import *
from pycore.blocks  import *

OUTPUT_DIR   = r'C:\Users\maxen\Documents\Stage\Software\OT_NN\Pytorch_NN\illustrations\BE_UNet\double_conv'
PROJECT_PATH = r'C:\Users\maxen\Documents\Stage\PlotNeuralNet'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)

img_rho = str(Path(OUTPUT_DIR) / 'Densities.png').replace('\\', '/')
img_sx  = str(Path(OUTPUT_DIR) / 'FEM_sigma_x.png').replace('\\', '/')
img_sy  = str(Path(OUTPUT_DIR) / 'FEM_sigma_y.png').replace('\\', '/')
img_txy = str(Path(OUTPUT_DIR) / 'FEM_tau_xy.png').replace('\\', '/')

# dimensions
# u1 = 32
# u2 = u1//2
# u3 = u2//2
# u4 = u3//2
# u5 = u4//2
# u  = u5//2
u1 = 40   # enc0, dec0  — résolution 32x32
u2 = 32   # enc1, dec1  — résolution 16x16
u3 = 24   # enc2, dec2  — résolution 8x8
u4 = 16   # enc3, dec3  — résolution 4x4
u5 = 8    # bottleneck  — résolution 2x2
u  = 4    # embedding   — résolution 1x1

off_enc = 1
off_dec = 2

arch = [
        to_head(PROJECT_PATH),
        to_cor(),
        to_begin(),

        # Input
        to_input(img_rho, to='(0, 0, 0)', width=8, height=8, name='input_rho'),

        # Encoder
        to_ConvConvRelu("enc0", " ", (" "," "),
                offset=f"({off_dec},0,0)", to="(0,0,0)",
                height=u1, depth=u1, width=(4,4)),

        to_Pool("pool0",
                offset="(0,0,0)", to="(enc0-east)",
                height=u2, depth=u2, width=1),

        to_ConvConvRelu("enc1", " ", (" "," "),
                offset=f"({off_enc},0,0)", to="(pool0-east)",
                height=u2, depth=u2, width=(5,5)),

        to_Pool("pool1",
                offset="(0,0,0)", to="(enc1-east)",
                height=u3, depth=u3, width=1),

        to_ConvConvRelu("enc2", " ", (" "," "),
                offset=f"({off_enc},0,0)", to="(pool1-east)",
                height=u3, depth=u3, width=(6,6)),

        to_Pool("pool2",
                offset="(0,0,0)", to="(enc2-east)",
                height=u4, depth=u4, width=1),

        to_ConvConvRelu("enc3", " ", (" "," "),
                offset=f"({off_enc},0,0)", to="(pool2-east)",
                height=u4, depth=u4, width=(7,7)),

        to_Pool("pool3",
                offset="(0,0,0)", to="(enc3-east)",
                height=u5, depth=u5, width=1),

        # Bottleneck
        to_ConvConvRelu("bottleneck", " ", (" "," "),
                offset=f"({off_enc},0,0)", to="(pool3-east)",
                height=u5, depth=u5, width=(8,8)),

        # ── BoundaryEmbedding ────────────────────────────────────────────
        to_Input("input_nodes", " ", " ",
                offset="(0,0,0)", to="(4, -6, 0)",
                height=u, depth=u, width=3,
                caption="nodal forces"),

        to_MLP("mlp", " ",
                offset="(4,0,0)", to="(input_nodes-east)",
                height=u, depth=u, width=6,
                caption="Multilayer Perceptron"),

        to_UnPool("be_up",
                offset="(4,0,0)", to="(mlp-east)",
                height=u5, depth=u5, width=6,
                caption=" "),

        to_connection("input_nodes", "mlp"),
        to_connection("mlp",         "be_up"),

        # CBAM
        to_CBAM("cbam", " ", " ",
                offset="(0,0,0)", to="(bottleneck-east)",
                height=u5, depth=u5, width=12),

        to_concat_embed(of="be_up", to="cbam"),

        # Decoder
        to_UnPool("unpool3",
                offset=f"({off_dec},0,0)", to="(cbam-east)",
                height=u4, depth=u4, width=1, opacity=0.8),

        to_ConvConvRelu("dec3", " ", (" "," "),
                offset="(0,0,0)", to="(unpool3-east)",
                height=u4, depth=u4, width=(7,7)),

        to_UnPool("unpool2",
                offset=f"({off_dec},0,0)", to="(dec3-east)",
                height=u3, depth=u3, width=1, opacity=0.8),

        to_ConvConvRelu("dec2", " ", (" "," "),
                offset="(0,0,0)", to="(unpool2-east)",
                height=u3, depth=u3, width=(6,6)),

        to_UnPool("unpool1",
                offset=f"({off_dec},0,0)", to="(dec2-east)",
                height=u2, depth=u2, width=1, opacity=0.8),

        to_ConvConvRelu("dec1", " ", (" "," "),
                offset="(0,0,0)", to="(unpool1-east)",
                height=u2, depth=u2, width=(5,5)),

        to_UnPool("unpool0",
                offset=f"({off_dec},0,0)", to="(dec1-east)",
                height=u1, depth=u1, width=1, opacity=0.8),

        to_ConvConvRelu("dec0", " ", (" "," "),
                offset="(0,0,0)", to="(unpool0-east)",
                height=u1, depth=u1, width=(4,4)),

        # Output
        to_output(img_sx,  to='(40.5, 0, 0)', width=8, height=8, name='output_sx' ),
        to_output(img_sy,  to='(41.5, 0, 0)', width=8, height=8, name='output_sy' ),
        to_output(img_txy, to='(42.5, 0, 0)', width=8, height=8, name='output_txy'),

        # Connections
        to_connection("pool0",       "enc1"),
        to_connection("pool1",       "enc2"),
        to_connection("pool2",       "enc3"),
        to_connection("pool3",       "bottleneck"),
        to_connection("cbam",        "unpool3"),
        to_connection("dec3",        "unpool2"),
        to_connection("dec2",        "unpool1"),
        to_connection("dec1",        "unpool0"),

        # Skip connections
        to_skip(of="enc3", to="unpool3", pos=1.25),
        to_skip(of="enc2", to="unpool2", pos=1.25),
        to_skip(of="enc1", to="unpool1", pos=1.25),
        to_skip(of="enc0", to="unpool0", pos=1.25),

        to_end()
]

to_generate(arch, 'BE_unet_double_conv.tex')

with open('BE_unet_double_conv.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# ── Nodal forces caption ─────────────────────────────────────────────────────
content = content.replace(
    r'caption=nodal forces',
    r'caption={\scalebox{1.5}{\parbox{3cm}{\vspace{0.5cm}nodal\\forces}}}'
)
content = content.replace(
    r'caption=Multilayer Perceptron',
    r'caption={\scalebox{1.5}{\parbox{4cm}{\vspace{0.5cm}Multilayer\\Perceptron}}}'
)


# ── Patch anyfontsize + graphicx ─────────────────────────────────────────────
content = content.replace(
    r'\usepackage{tikz}',
    r'\usepackage{tikz}' + '\n' + r'\usepackage{anyfontsize}' + '\n' + r'\usepackage{graphicx}'
)

# ── Patch captions ────────────────────────────────────────────────────────────
content = content.replace(
    r'caption=Output\\$(\sigma_{x},\,\sigma_{y},\,\tau_{xy})$',
    r'caption={Output\\$(\sigma_{x},\,\sigma_{y},\,\tau_{xy})$}'
)

# ── Patch line width ──────────────────────────────────────────────────────────
content = re.sub(
    r'\\pic\[shift=',
    r'\\pic[line width=0.2pt, shift=',
    content
)

content = content.replace(
    r'caption=nodes $[B,16]$',
    r'caption={nodes $[B,16]$}'
)

# ── Output images transparency ────────────────────────────────────────────────
for name in ['output_sx', 'output_sy', 'output_txy']:
    content = content.replace(
        rf'\node[canvas is zy plane at x=0, inner sep=0pt, outer sep=0pt] ({name})',
        rf'\node[canvas is zy plane at x=0, inner sep=0pt, outer sep=0pt, opacity=0.9] ({name})'
    )

# ── Arrow behind output images ────────────────────────────────────────────────
arrow_behind = r"\draw [connection] (dec0-east) -- node {\midarrow} (39, 0);" + "\n"

content = content.replace(
    r'\node[canvas is zy plane at x=0, inner sep=0pt, outer sep=0pt, opacity=0.9] (output_sx)',
    arrow_behind + r'\node[canvas is zy plane at x=0, inner sep=0pt, outer sep=0pt, opacity=0.9] (output_sx)'
)




# ── Labels + input arrow ──────────────────────────────────────────────────────
labels_arrows = r"""
% Input label + arrow
\node[left, align=center] (input_label) at (0, -7) {
    \scalebox{3}{\textbf{Input}} \\ \scalebox{2.5}{$\rho$}
};
\draw [connection] (0, 0) -- node {\midarrow} (enc0-west);

% Output label only (arrow already drawn behind images)
\node[right, align=center] (output_label) at (38, -7) {
    \scalebox{3}{\textbf{Output}} \\ \scalebox{2.5}{$(\sigma_{xx},\; \sigma_{yy},\; \tau_{xy})$}
};
"""

content = content.replace(
    r'\end{tikzpicture}',
    labels_arrows + r'\end{tikzpicture}',
    1
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend = r"""
\node[anchor=north] at (current bounding box.south) {
    \begin{tikzpicture}

        % Row 1 - Col 1
        \pic at (0,0) {RightBandedBox={name=leg_conv,
            fill=\ConvColor, bandfill=\ConvReluColor,
            height=12, width=2, depth=12,
            opacity=0.8, bandopacity=0.8,
            xlabel={{"",""}}, zlabel=, caption=}};
        \node[right] at (1.0, 0) {\scalebox{3}{Convolutional layer}};

        % Row 1 - Col 2
        \pic at (12,0) {Box={name=leg_pool, fill=\PoolColor,
            height=12, width=1, depth=12, opacity=0.8,
            xlabel={{"","","","","","","","","",""}},
            zlabel=, caption=}};
        \node[right] at (12.6, 0) {\scalebox{3}{Max Pooling operation}};

        % Row 2 - Col 1
        \pic at (0,-3.0) {Box={name=leg_unpool, fill=\UnpoolColor,
            height=12, width=1, depth=12, opacity=0.8,
            xlabel={{"","","","","","","","","",""}},
            zlabel=, caption=}};
        \node[right] at (0.6, -3.0) {\scalebox{3}{Up-conv. operation}};

        % Row 2 - Col 2
        \pic at (12,-3.0) {RightBandedBox={name=leg_cbam,
            fill=\SoftmaxColor, bandfill=\CBAMColor,
            height=12, width=2, depth=12,
            opacity=0.8, bandopacity=0.8,
            xlabel={{"",""}}, zlabel=, caption=}};
        \node[right] at (13.0, -3.0) {\scalebox{3}{CBAM}};

        % Row 1+2 - Col 3 : Skip connection
        \draw[copyconnection, -Stealth] (20, -3) -- ++(2.0, 0)
            node[right] {\scalebox{3}{Skip connection: Concatenation}};

    \end{tikzpicture}
};
"""

content = content.replace(
    r'\end{tikzpicture}',
    legend + r'\end{tikzpicture}',
    1
)

# ── Write & compile ───────────────────────────────────────────────────────────
with open('BE_unet_double_conv.tex', 'w', encoding='utf-8') as f:
    f.write(content)

os.system('pdflatex BE_unet_double_conv.tex')
os.system('start BE_unet_double_conv.pdf')