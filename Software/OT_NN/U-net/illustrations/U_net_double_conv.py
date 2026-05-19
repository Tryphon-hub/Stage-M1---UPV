import sys
import os
import re

sys.path.insert(0, r'C:\Users\maxen\Documents\Stage\PlotNeuralNet')
from pycore.tikzeng import *
from pycore.blocks  import *

OUTPUT_DIR   = r'C:\Users\maxen\Documents\Stage\Software\OT_NN\U-net\illustrations'
PROJECT_PATH = r'C:\Users\maxen\Documents\Stage\PlotNeuralNet'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)

arch = [
    to_head(PROJECT_PATH),
    to_cor(),
    to_begin(),

    # Input
    to_Input("input", " ", " ",
            offset="(0,0,0)", to="(0,0,0)",
            height=40, depth=40, width=1,
            caption=r"Input\\{$(\rho\;t_{x}\;t_{y})$}"),

    # Encoder
    to_ConvConvRelu("enc0", " ", (" "," "),
            offset="(2,0,0)", to="(input-east)",
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
    to_Output("output", " ", " ",
            offset="(2,0,0)", to="(dec0-east)",
            height=40, depth=40, width=1,
            caption=r"Output\\$(\sigma_{x},\,\sigma_{y},\,\tau_{xy})$"),

    # Connections
    to_connection("input",       "enc0"),
    to_connection("pool0",       "enc1"),
    to_connection("pool1",       "enc2"),
    to_connection("pool2",       "enc3"),
    to_connection("pool3",       "bottleneck"),
    to_connection("cbam",  "unpool3"),
    to_connection("dec3",        "unpool2"),
    to_connection("dec2",        "unpool1"),
    to_connection("dec1",        "unpool0"),
    to_connection("dec0",        "output"),

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

# ── Legend ────────────────────────────────────────────────────────────────────
legend = r"""
\node[anchor=north] at (current bounding box.south) {
    \begin{tikzpicture}

        % Colonne 1
        \pic at (0,0) {Box={name=leg_input, fill=\InputColor,
            height=12, width=1, depth=12, opacity=0.8,
            xlabel={{"","","","","","","","","",""}},
            zlabel=, caption=}};
        \node[right] at (0.6, 0) {\LARGE Input layer};

        \pic at (0,-3.0) {RightBandedBox={name=leg_conv,
            fill=\ConvColor, bandfill=\ConvReluColor,
            height=12, width=2, depth=12,
            opacity=0.8, bandopacity=0.8,
            xlabel={{"",""}}, zlabel=, caption=}};
        \node[right] at (1.0, -3.0) {\LARGE Convolutional layer};

        % Colonne 2
        \pic at (8,0) {Box={name=leg_pool, fill=\PoolColor,
            height=12, width=1, depth=12, opacity=0.8,
            xlabel={{"","","","","","","","","",""}},
            zlabel=, caption=}};
        \node[right] at (8.6, 0) {\LARGE Max Pooling operation};

        \pic at (8,-3.0) {Box={name=leg_unpool, fill=\UnpoolColor,
            height=12, width=1, depth=12, opacity=0.8,
            xlabel={{"","","","","","","","","",""}},
            zlabel=, caption=}};
        \node[right] at (8.6, -3.0) {\LARGE Up-conv. operation};

        % Colonne 3
        \pic at (17,0) {Box={name=leg_output, fill=\OutputColor,
            height=12, width=1, depth=12, opacity=0.8,
            xlabel={{"","","","","","","","","",""}},
            zlabel=, caption=}};
        \node[right] at (17.6, 0) {\LARGE Output layer};

        
        \pic at (17,-3.0) {RightBandedBox={name=leg_conv,
            fill=\SoftmaxColor, bandfill=\CBAMColor,
            height=12, width=2, depth=12,
            opacity=0.8, bandopacity=0.8,
            xlabel={{"",""}}, zlabel=, caption=}};
        \node[right] at (18, -3.0) {\LARGE CBAM};


        % Colonne 4      
        \draw[copyconnection, -Stealth] (22, -1.5) -- ++(2.0, 0)
            node[right] {\LARGE Skip connection: Concatenation};

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