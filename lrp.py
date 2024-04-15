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
        # Placeholder for storing the last input
        self.last_input = None

    def forward(self, input):
        self.last_input = input
        output = F.linear(input, self.weight, self.bias)
        self.last_output = output  # Capture output for LRP
        return output

    def relprop(self, R):
        Z = torch.mm(self.last_input, torch.clamp(self.weight, min=0).t()) + self.bias + 1e-9
        S = R / Z
        C = torch.mm(S, torch.clamp(self.weight, min=0))
        return self.last_input * C


    def relprop(self, R, epsilon=0.01):
        V = self.weight.clamp(min=0)
        Z = torch.mm(self.last_input, V.t()) + self.bias + epsilon  # Adding epsilon for stability
        S = R / Z
        C = torch.mm(S, V)
        return self.last_input * C


class LRPConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(LRPConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # Placeholder for storing the last input
        self.last_input = None

    def forward(self, input):
        self.last_input = input
        return self.conv(input)

    def relprop(self, R):
        print(f"Input Relevance Shape: {R.shape}")
        # For simplicity, use weight positivity and a simple redistribution rule as an example.
        # This should be adapted based on the specific LRP rules and scenarios.
        weight = torch.clamp(self.conv.weight, min=0)
        output = F.conv2d(self.last_input, weight, bias=None, stride=self.conv.stride, padding=self.conv.padding)
        
    #  
        Z = output + 1e-9  # Avoid division by zero
        print(f"Shape of self.last_input: {self.last_input.shape}")
        print(f"Shape of output: {output.shape}")
        print(f"Shape of R: {R.shape}")

        # Attempt to match the dimensions of R with Z before division
        R_summed = R.sum(dim=1, keepdim=True)  # Summing over channels
        # Reshape R_summed to have a shape compatible with Z for broadcasted division
        # Assuming the goal is to distribute relevance scores equally across all spatial dimensions and channels
        R_expanded = R_summed.unsqueeze(2).unsqueeze(3)  # Now R_expanded should have shape [64, 1, 1, 1]
        R_expanded = R_expanded.expand(-1, -1, 8, 8)  # Expand R_expanded to match Z's spatial dimensions, shape [64, 1, 8, 8]

        # Ensure channel dimension compatibility if necessary. Here's an example adjustment:
        R_expanded = R_expanded.expand_as(Z)  # This matches Z's shape, including the channel dimension

       
        print(f"Shape of R_summed: {R_summed.shape}")
        print(f"Shape of Z: {Z.shape}")

        # Now perform the division
        S = R_expanded / Z

        # This is a simplified approach; actual LRP for convolutions might involve more nuanced calculations.
        C = F.conv_transpose2d(S, weight, stride=self.conv.stride, padding=self.conv.padding)
        return self.last_input * C

def apply_lrp_to_last_conv_layer(model, input_data):
    output = model(input_data)
    _, target_class = torch.max(output, dim=1)
    R = torch.zeros_like(output)
    R[torch.arange(input_data.shape[0]), target_class] = output[torch.arange(input_data.shape[0]), target_class]
    
    # Start relevance propagation from the output back to the first relevant layer
    for layer in reversed(model.get_layers()):
        R = layer.relprop(R)
        if layer == model.conv2:
            print(f"Relevance at Conv2 layer: {R.shape}")
            break
    return R





