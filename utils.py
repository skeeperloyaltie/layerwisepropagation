# import torch
# import torch.nn.functional as F

# def manual_convolution(input_tensor, weight, bias=None, stride=1, padding=0):
#     # Get the number of channels, height, and width of each filter
#     out_channels, in_channels, kernel_height, kernel_width = weight.shape
    
#     # Padding the input tensor
#     input_tensor = F.pad(input_tensor, (padding, padding, padding, padding))
    
#     # Extract the height and width of the padded input
#     batch_size, _, height, width = input_tensor.shape
    
#     # Calculate output dimensions
#     out_height = (height - kernel_height) // stride + 1
#     out_width = (width - kernel_width) // stride + 1
    
#     # Create output tensor
#     output = torch.zeros((batch_size, out_channels, out_height, out_width)).to(input_tensor.device)
    
#     # Perform the convolution operation manually
#     for i in range(out_height):
#         for j in range(out_width):
#             h_start = i * stride
#             h_end = h_start + kernel_height
#             w_start = j * stride
#             w_end = w_start + kernel_width
            
#             region = input_tensor[:, :, h_start:h_end, w_start:w_end]
#             output[:, :, i, j] = torch.sum(region.unsqueeze(1) * weight, dim=[2, 3, 4])
    
#     # Add bias if not None
#     if bias is not None:
#         output += bias.view(1, -1, 1, 1)
    
#     return output

