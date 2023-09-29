import torch
import torch.nn as nn
import torch.onnx

"""
When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also known as a “depthwise convolution”.
https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
"""

# in_ch = 16
# out_ch = 16
# size = 64
# batch = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, groups = 16, padding=0, bias=False)

    def forward(self,x):
        h = self.conv1(x)
        return h

torch.manual_seed(0)
model = Model()
x = torch.randn(1, 16, 146, 98, dtype=torch.float32)
torch.onnx.export(model, x, 'depthwise.onnx')
