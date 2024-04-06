import torch
import torch.nn as nn
import torch.nn.functional as F

class LRPLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(LRPLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, input):
        self.input = input # Save input for use in relprop
        return F.linear(input, self.weight, self.bias)

    def relprop(self, R):
        # Assume R is the relevance of the output of this layer, we calculate the relevance of the input
        V = self.weight.clamp(min=0)
        Z = torch.mm(self.input, V.t()) + self.bias + 1e-9  # Adding bias for stability
        S = R / Z
        C = torch.mm(S, V)
        R = self.input * C
        return R
