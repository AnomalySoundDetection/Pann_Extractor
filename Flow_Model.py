import normflows as nf
import torch.nn as nn

base = nf.distributions.base.DiagGaussian(2)

num_layers = 32

class FlowModel(nn.Module):
    def __init__(self, num_layers):
        super(FlowModel, self).__init__()

        self.base = nf.distributions.base.DiagGaussian(2)
        self.num_layers = num_layers

    def create_flows(self):
        for i in range(self.num_layers):
            param_map = nf.nets.MLP([1024, ])

    def forward(self, x):
        pass