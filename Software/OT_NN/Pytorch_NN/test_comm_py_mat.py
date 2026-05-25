import torch
from model import UNetTopo

#%% Import pytorch model and load weights

model = UNetTopo(nif=32, n_in=3, n_out=3, use_cbam=True)
state_dict = torch.load(
    r'C:\Users\maxen\Documents\Stage\Software\OT_NN\U-net\results\unet_topo_best.pth',
    map_location='cpu'
)
model.load_state_dict(state_dict)
model.eval()

#%% Export model and weights as ONNX

dummy_input = torch.zeros(1, 3, 32, 32)
torch.onnx.export(
    model,
    dummy_input,
    r'C:\Users\maxen\Documents\Stage\Software\OT_NN\U-net\results\unet_topo.onnx',
    input_names   = ['input'],
    output_names  = ['output'],
    opset_version = 11
)
print('Export ONNX réussi')