import torch
import torch.nn as nn
import torch.nn.functional as F

def test_conv2d_padding(input_shape, kernel_shape):
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create original input and kernel tensors
    original_input = torch.randint(low=1, high=11, size=input_shape, dtype=torch.int32)
    original_kernel = torch.randint(low=1, high=11, size=kernel_shape, dtype=torch.int32)

    # Define the convolutional layer with "same" padding, no bias, and stride 1
    conv = nn.Conv2d(input_shape[1], kernel_shape[0], kernel_shape[2:], padding='same', bias=False, stride=1)

    # Perform convolution with original input and kernel
    with torch.no_grad():
        original_conv_result = conv(original_input)

    # Perform zero filling for the input tensor
    W = input_shape[3]
    padding_width = 32 - (W % 32)
    zero_padded_input = F.pad(original_input, (0, padding_width, 0, 0), mode='constant')

    # Perform convolution with zero-padded input and kernel
    with torch.no_grad():
        padded_conv_result = conv(zero_padded_input)

    # Check if the output height and width match the input height and width
    output_height, output_width = padded_conv_result.shape[2:]
    input_height, input_width = input_shape[2:]
    output_shape_match = (output_height == input_height) and (output_width == input_width)

    # Check if the convolution results match (only if output shape matches)
    conv_results_match = False
    if output_shape_match:
        conv_results_match = torch.allclose(original_conv_result, padded_conv_result, atol=1e-6)

    print("Output Shape Matches Input Shape:", output_shape_match)
    print("Convolution Results Match:", conv_results_match)

# Example usage:
input_shape = (1, 1, 4, 4)
kernel_shape = (1, 1, 3, 3)
test_conv2d_padding(input_shape, kernel_shape)
