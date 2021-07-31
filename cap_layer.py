import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
import constants

def v2v_non_linearity(vec: torch.Tensor, dim=-1):
    """
    This is the non-linear activation used in CapsNet that takes in a vector, squashes it to have length <= 1
    while retaining the directional information.
    """
    l2_norm = LA.norm(vec, dim=dim, keepdim=True)
    scale = (l2_norm**2)/(1 + l2_norm**2)
    unit_vec = vec/(l2_norm + constants.EPSILON) # handling division by 0.
    return unit_vec * scale

class DenseCapsule(nn.Module):
    """
    DenseCapsule layer is similar to the Dense FC layer except instead of taking scalar as inputs it operates on
    vectors. For each DenseCapsule layer input will be of the form (batch_size, #in_caps, in_dim) and output will
    be (batch_size, #out_caps, out_dim).

    :param in_num_caps: number of capsules in the previous layer.
    :param in_dim: dimension of the input vectors
    :param out_num_caps: number of capsules in the current layer.
    :param out_dim: dimension of the output vectors.
    :param num_iter: number of iterations to be used for the Dynamic Routing Algorithm. Default value = 3. Should
                    be > 0.
    """
    def __init__(self, in_num_caps: int, in_dim: int, out_num_caps: int, out_dim: int, is_cuda_available: bool, num_iter=3)-> None:
        # validate num_iter value.
        if type(num_iter) != int or num_iter < 0:
            raise ValueError("Invalid value for num_iter", num_iter)

        super(DenseCapsule, self).__init__()
        self.in_num_caps, self.in_dim = in_num_caps, in_dim
        self.out_num_caps, self.out_dim = out_num_caps, out_dim

        self.num_iter = num_iter
        self.weight = nn.Parameter(1e-2*torch.randn(out_num_caps, in_num_caps, out_dim, in_dim))
        self.is_cuda_available = is_cuda_available

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Computes forward pass for the DenseCapsule layer.

        :param x: torch.Tensor of size (batch_size, in_num_caps, in_dim)

        :return: output tensor of size(batch_size, out_num_caps, out_dim)
        """

        # expand input to (batch_size, 1, in_num_caps, in_dim, 1) and then (batch) matrix multiply it to the
        # weight. Resulting in output of size (batch_size, out_num_caps, in_num_caps, out_dim, 1). Squeeze out
        # the last dimension to get x_hat
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:,None,:,:,None]), dim=-1)

        # Dynamic Routing Algorithm. Use `no_grad` to prevent flow in back-propagation.
        with torch.no_grad():
            routing_logits = torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)
            if self.is_cuda_available:
                routing_logits = routing_logits.cuda()

            for i in range(self.num_iter-1):
                coeff = F.softmax(routing_logits, dim=1)
                output = v2v_non_linearity(torch.matmul(coeff[:,:,None,:], x_hat))
                routing_logits += torch.sum(output*x_hat, dim=-1)

        coeff = F.softmax(routing_logits, dim=1)
        output = v2v_non_linearity(torch.matmul(coeff[:,:,None,:], x_hat))

        return torch.squeeze(output, dim=-2)

class ConvCapsule(nn.Module):
    """
    ConvCapsule layer is similar to the Conv2D layer except it transforms the output shape from a 3D volume to
    multiple vectors each representing a capsule.

    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_capsule: dimension of the vector for each capsule
    :param kernel_size: kernel size
    :param stride: stride to be used by the kernel
    :param padding: padding to be applied on the input channel
    """
    def __init__(self, in_channels: int, out_channels: int, dim_capsule: int, kernel_size: int, stride=1, padding=0)-> None:
        super(ConvCapsule, self).__init__()
        self.dim_capsule = dim_capsule
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Computes the forward pass for the ConvCapsule layer.
        
        :param x: torch.Tensor of size (batch_size,height, width, in_channels)

        :return: output tensor of size (batch_size, num_capsules, dim_capsule)
        """
        output = self.conv2d(x)
        return v2v_non_linearity(output.view(x.size(0), -1, self.dim_capsule))


if __name__ == '__main__':
    DenseCapsule(10, 20, 30, 40, False, 1)
    ConvCapsule(10, 20, 30, 4)
