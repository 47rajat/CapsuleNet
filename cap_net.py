from PIL import Image
from cap_layer import DenseCapsule, ConvCapsule
from constants import *
from torch import nn
from typing import Tuple, Dict
from utils import combine_images
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


class CapsuleNet(nn.Module):
    """
    CapsuleNet architecture.

    :param input_dims: dimension of the input image (channel, height, width)
    :param num_classes: number of classes in the input dataset.
    :param architecture: map containing information about various components of the network.
    """
    def __init__(self, input_dims: Tuple[int,int,int], num_classes: int, architecture: Dict, is_cuda_available: bool)-> None:
        super(CapsuleNet, self).__init__()

        self.input_dims = input_dims
        self.num_classes = num_classes
        self.num_iter = architecture[DENSE_CAPSULE][NUM_ITER]
        self.is_cuda_available = is_cuda_available

        # initialize layers
        self.conv1 = nn.Conv2d(input_dims[0], architecture[CONV1][CHANNELS],
                kernel_size=architecture[CONV1][KERNEL_SIZE],
                stride=architecture[CONV1][STRIDE],
                padding=architecture[CONV1][PADDING])

        self.conv_capsule = ConvCapsule(architecture[CONV_CAPSULE][IN_CHANNELS],
                architecture[CONV_CAPSULE][OUT_CHANNELS],
                architecture[CONV_CAPSULE][CAP_DIM],
                architecture[CONV_CAPSULE][KERNEL_SIZE],
                stride=architecture[CONV_CAPSULE][STRIDE],
                padding=architecture[CONV_CAPSULE][PADDING])

        self.dense_capsule = DenseCapsule(
            in_num_caps=architecture[DENSE_CAPSULE][NUM_IN_CAPS],
            in_dim=architecture[DENSE_CAPSULE][IN_CAP_DIM],
            out_num_caps=num_classes,
            out_dim=architecture[DENSE_CAPSULE][OUT_CAP_DIM],
            num_iter=architecture[DENSE_CAPSULE][NUM_ITER],
            is_cuda_available=self.is_cuda_available)

        # decoder network.
        self.decoder = nn.Sequential(
                nn.Linear(architecture[DENSE_CAPSULE][OUT_CAP_DIM]*num_classes, architecture[DECODER][FC1]),
                nn.ReLU(),
                nn.Linear(architecture[DECODER][FC1], architecture[DECODER][FC2]),
                nn.ReLU(),
                nn.Linear(architecture[DECODER][FC2], input_dims[0]*input_dims[1]*input_dims[2]),
                nn.Sigmoid(),
                )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes forward pass for the CapsuleNet. Returns the class prediction + image reconstruction.

        :param x: torch.Tensor of size (batch_size, image_channels, image_height, image_width) representing
                    input image.
        :param y: torch.Tensor of size (batch_size, num_classes) representing the labels. Will not be provided 
                    during the testing phase.

        :return: Tuple of tensor (predictions, reconstruction)
        """
        x = self.relu(self.conv1(x))
        x = self.conv_capsule(x)
        x = self.dense_capsule(x)

        predictions = x.norm(dim=-1)

        if y is None:
            index = predictions.argmax(dim=-1, keepdim=True)
            y = torch.zeros(predictions.size()).scatter_(1, index.cpu().data, 1.)
            if self.is_cuda_available:
                y = y.cuda()

        reconstruction = self.decoder((x*y[:,:,None]).view(x.size(0),-1))
        return predictions, reconstruction.view(-1, *self.input_dims)

def cap_net_loss(y_true: torch.Tensor, y_pred: torch.Tensor, x: torch.Tensor, x_recon: torch.Tensor, lam_recon: float)-> torch.Tensor:
    """
    CasuleNetLoss = Margin loss + lam_recon * reconstruction loss.

    :param y_true: true labels, one hot encoded, size = (batch_size, num_classes)
    :param y_pred: predicted labels by the CapsuleNet, same size as `y_true`
    :param x: input (image) data, size = (batch_size, channels, width, height)
    :param x_recon: reconstructed input generated by the CapsuleNet, same size as `x`
    :param lam_recon: coefficient for the reconstruction loss.

    :return: Tensor containing a scalar loss value
    """
    loss = y_true * torch.clamp(M_PLUS - y_pred, min=0.)**2 + \
            0.5 * (1 - y_true) * torch.clamp(y_pred - M_MINUS, min=0.)**2

    loss_margin = loss.sum(dim=1).mean()

    loss_recon = nn.MSELoss()(x_recon, x)

    return loss_margin + lam_recon * loss_recon

def show_reconstruction(model, test_loader, num_images, file_args, is_cuda_available=False, show=False):
    """
    Displays the reconstructed images generated using the provided model for the images present in the
    test_loader.

    :param model: the CapsuleNet model.
    :param test_loader: torch.utils.data.DataLoader for the test data.
    :param num_images:
    :param file_args: dictionary contating details for recon file path (obtained from the experiment args)
    """
    model.eval()
    with torch.no_grad():
        for x, _ in test_loader:
            x = x[:min(num_images,x.size(0))]
            if is_cuda_available:
                x = x.cuda()
            _, x_recon = model(x, None)
            data = np.concatenate([x.cpu().data, x_recon.cpu().data])
            image = combine_images(np.transpose(data, [0,2,3,1]))
            save_path = os.path.join(file_args[SAVE_DIR], file_args[RECON_IMG_FILENAME])
            Image.fromarray(image).save(save_path)
            print('='*70)
            plt.figure()
            plt.imshow(plt.imread(save_path))
            if show:
                plt.show()
            break
