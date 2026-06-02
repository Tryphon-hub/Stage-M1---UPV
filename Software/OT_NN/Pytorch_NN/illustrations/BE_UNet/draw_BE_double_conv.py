import sys
import os
import re

sys.path.insert(0, r'C:\Users\maxen\Documents\Stage\PlotNeuralNet')
from pycore.tikzeng import *
from pycore.blocks  import *

OUTPUT_DIR   = r'C:\Users\maxen\Documents\Stage\Software\OT_NN\Pytorch_NN\illustrations\BE_UNet\double_conv'
PROJECT_PATH = r'C:\Users\maxen\Documents\Stage\PlotNeuralNet'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)

# dimensions

u1 = 32
u2 = u1//2
u3 = u2//2
u4 = u3//2
u5 = u4//2
u  = u5//2

off_enc = 1
off_dec = 1


arch = [
    to_head(PROJECT_PATH),
    to_cor(),
    to_begin(),

    # Input
    to_Input("input", " ", " ",
            offset="(0,0,0)", to="(0,0,0)",
            height=u1, depth=u1, width=1,
            caption=r"Input $\rho$"),

    # Encoder
    to_ConvConvRelu("enc0", " ", (" "," "),
            offset=f"({off_dec},0,0)", to="(input-east)",
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
        
    to_Input("nodes", " ", " ",
            offset="(-2,-4,0)", to="(enc0-south)",   # ← ancré au nord du bottleneck
            height=u, depth=u, width=6,
            caption="Boundary conditions"),

    to_SoftMax("layer1", " ",
            offset=f"({off_dec},0,0)", to="(nodes-east)",  # ← juste au-dessus aussi
            height=u, depth=u, width=12,
            caption="layer 1"),

    to_SoftMax("layer2", " ",
            offset=f"({off_dec},0,0)", to="(layer1-east)",
            height=u, depth=u, width=24,
            caption="layer 2"),

    to_connection("nodes", "layer1"),   
    to_connection("layer1", "layer2"),

    to_UnPool("be_up",
            offset=f"({off_dec},0,0)", to="(layer2-east)",
            height=u5, depth=u5, width=24,
            caption="UnPool"),

    to_connection("layer2", "be_up"),
   

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
    to_Output("output", " ", " ",
            offset=f"({off_dec},0,0)", to="(dec0-east)",
            height=u1, depth=u1, width=1,
            caption=r"Output\\$(\sigma_{x},\,\sigma_{y},\,\tau_{xy})$"),

    # Connections
    to_connection("input",       "enc0"),
    to_connection("pool0",       "enc1"),
    to_connection("pool1",       "enc2"),
    to_connection("pool2",       "enc3"),
    to_connection("pool3",       "bottleneck"),
    to_connection("cbam",        "unpool3"),
    to_connection("dec3",        "unpool2"),
    to_connection("dec2",        "unpool1"),
    to_connection("dec1",        "unpool0"),
    to_connection("dec0",        "output"),

    # Skip connections
    to_skip(of="enc3", to="unpool3", pos=2),
    to_skip(of="enc2", to="unpool2", pos=2),
    to_skip(of="enc1", to="unpool1", pos=1.6),
    to_skip(of="enc0", to="unpool0", pos=1.25),

    to_end()
]

to_generate(arch, 'BE_unet_double_conv.tex')

with open('BE_unet_double_conv.tex', 'r', encoding='utf-8') as f:
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
        \node[right] at (8.6, -3.0) {\LARGE Unpool operation};

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
            node[right] {\LARGE Concatenation};

    \end{tikzpicture}
};
"""


content = content.replace(
    r'\end{tikzpicture}',
    legend + r'\end{tikzpicture}'
)

# ── Write & compile ───────────────────────────────────────────────────────────

# with open('BE_unet_double_conv.tex', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
# for i, line in enumerate(lines[30:45], start=31):
#     print(f"{i}: {line}", end='')




with open('BE_unet_double_conv.tex', 'w', encoding='utf-8') as f:
    f.write(content)

os.system('pdflatex BE_unet_double_conv.tex')
os.system('start BE_unet_double_conv.pdf')