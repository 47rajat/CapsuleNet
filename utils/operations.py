import torch


def one_hot_encode(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Creates the one hot encoding of the provided class labels.

    :param y: class labels of size (batch_size,)

    :return y_one_hot: one hot encoding of size (batch_size, num_classes)
    """
    return torch.zeros(y.size(0), num_classes).scatter_(1, y.view(-1, 1), 1.)
