import sys
import os
import re
import shutil
from pathlib import Path

sys.path.insert(0, r'C:\Users\maxen\Documents\Stage\PlotNeuralNet')
from pycore.tikzeng import *
from pycore.blocks  import *

OUTPUT_DIR   = r'C:\Users\maxen\Documents\Stage\Software\OT_NN\Pytorch_NN\illustrations\U-Net\double_conv'
PROJECT_PATH = r'C:\Users\maxen\Documents\Stage\PlotNeuralNet'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)


img_rho = str(Path(OUTPUT_DIR) / 'Densities.png').replace('\\', '/')
img_tx  = str(Path(OUTPUT_DIR) / 'tx.png').replace('\\', '/')
img_ty  = str(Path(OUTPUT_DIR) / 'ty.png').replace('\\', '/')

img_sx  = str(Path(OUTPUT_DIR) / 'FEM_sigma_x.png').replace('\\', '/')
img_sy  = str(Path(OUTPUT_DIR) / 'FEM_sigma_y.png').replace('\\', '/')
img_txy = str(Path(OUTPUT_DIR) / 'FEM_tau_xy.png').replace('\\', '/')


arch = [
    to_head(PROJECT_PATH),
    to_cor(),
    to_begin(),

    to_input(img_ty,  to='(-2, 0, 0)', width=8, height=8, name='input_ty' ),
    to_input(img_tx,  to='(-1, 0, 0)', width=8, height=8, name='input_tx' ),
    to_input(img_rho, to='(0, 0, 0)', width=8, height=8, name='input_rho'),


    # Encoder
    to_ConvConvRelu("enc0", " ", (" "," "),
            offset="(2,0,0)", to="(0,0,0)",
            height=40, depth=40, width=(4,4)),

    to_Pool("pool0",
            offset="(0,0,0)", to="(enc0-east)",
            height=32, depth=32, width=1),

    to_ConvConvRelu("enc1", " ", (" "," "),
            offset="(1,0,0)", to="(pool0-east)",
            height=32, depth=32, width=(5,5)),

    to_Pool("pool1",
            offset="(0,0,0)", to="(enc1-east)",
            height=24, depth=24, width=1),

    to_ConvConvRelu("enc2", " ", (" "," "),
            offset="(1,0,0)", to="(pool1-east)",
            height=24, depth=24, width=(6,6)),

    to_Pool("pool2",
            offset="(0,0,0)", to="(enc2-east)",
            height=16, depth=16, width=1),

    to_ConvConvRelu("enc3", " ", (" "," "),
            offset="(1,0,0)", to="(pool2-east)",
            height=16, depth=16, width=(7,7)),

    to_Pool("pool3",
            offset="(0,0,0)", to="(enc3-east)",
            height=8, depth=8, width=1),

    # Bottleneck
    to_ConvConvRelu("bottleneck", " ", (" "," "),
            offset="(1,0,0)", to="(pool3-east)",
            height=8, depth=8, width=(8,8)),

    # CBAM
    to_CBAM("cbam", " ", " ",
            offset="(0,0,0)", to="(bottleneck-east)",
            height=8, depth=8, width=8),

    # Decoder
    to_UnPool("unpool3",
              offset="(2,0,0)", to="(cbam-east)",
              height=16, depth=16, width=1, opacity=0.8),

    to_ConvConvRelu("dec3", " ", (" "," "),
               offset="(0,0,0)", to="(unpool3-east)",
               height=16, depth=16, width=(7,7)),

    to_UnPool("unpool2",
              offset="(2,0,0)", to="(dec3-east)",
              height=24, depth=24, width=1, opacity=0.8),

    to_ConvConvRelu("dec2", " ", (" "," "),
               offset="(0,0,0)", to="(unpool2-east)",
               height=24, depth=24, width=(6,6)),

    to_UnPool("unpool1",
              offset="(2,0,0)", to="(dec2-east)",
              height=32, depth=32, width=1, opacity=0.8),

    to_ConvConvRelu("dec1", " ", (" "," "),
               offset="(0,0,0)", to="(unpool1-east)",
               height=32, depth=32, width=(5,5)),

    to_UnPool("unpool0",
              offset="(2,0,0)", to="(dec1-east)",
              height=40, depth=40, width=1, opacity=0.8),

    to_ConvConvRelu("dec0", " ", (" "," "),
               offset="(0,0,0)", to="(unpool0-east)",
               height=40, depth=40, width=(4,4)),

    # Output
    to_output(img_sx,  to='(40, 0, 0)', width=8, height=8, name='output_sx' ),
    to_output(img_sy,  to='(41, 0, 0)', width=8, height=8, name='output_sy' ),
    to_output(img_txy, to='(42, 0, 0)', width=8, height=8, name='output_txy'),
    # Connections

    to_connection("pool0",       "enc1"),
    to_connection("pool1",       "enc2"),
    to_connection("pool2",       "enc3"),
    to_connection("pool3",       "bottleneck"),
    to_connection("cbam",  "unpool3"),
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

to_generate(arch, 'unet_double_conv.tex')

with open('unet_double_conv.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# ── Patch anyfontsize ─────────────────────────────────────────────────────────
content = content.replace(
    r'\usepackage{tikz}',
    r'\usepackage{tikz}' + '\n' + r'\usepackage{anyfontsize}'
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

# ── Labels + input arrow (on top) ─────────────────────────────────────────────
labels_arrows = r"""
% Input label + arrow
\node[left, align=center] (input_label) at (0, -7) {
    \scalebox{3}{\textbf{Input}} \\ \scalebox{2.5}{$(\rho,\; t_x,\; t_y)$}
};
\draw [connection] (0, 0) -- node {\midarrow} (enc0-west);

% Output label only (arrow already drawn behind images)
\node[right, align=center] (output_label) at (38, -7) {
    \scalebox{3}{\textbf{Output}} \\ \scalebox{2.5}{$(\sigma_{xx},\; \sigma_{yy},\; \tau_{xy})$}
};
"""

content = content.replace(
    r'\end{tikzpicture}',
    labels_arrows + r'\end{tikzpicture}'
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

        % Row 1+2 - Col 3 : Skip connection (right column, vertically centered)
        \draw[copyconnection, -Stealth] (20, -3) -- ++(2.0, 0)
            node[right] {\scalebox{3}{Skip connection: Concatenation}};

    \end{tikzpicture}
};
"""

content = content.replace(
    r'\end{tikzpicture}',
    legend + r'\end{tikzpicture}'
)

# ── Write & compile ───────────────────────────────────────────────────────────
with open('unet_double_conv.tex', 'w', encoding='utf-8') as f:
    f.write(content)

os.system('pdflatex unet_double_conv.tex')
os.system('start unet_double_conv.pdf')