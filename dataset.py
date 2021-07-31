from torchvision import transforms, datasets
from constants import *
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict

def load_dataset(args: Dict)-> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct dataloder for training, validation and testing. Data augmentation is also added here.

    :param args: arguments (dictionary) containing information required to build the data loader.

    :return: train_loader, val_loader, test_loader
    """
    # create train and test dataset.
    train_val_dataset, test_dataset = get_dataset(args)

    # split train_val_dataset into train and validation dataset.
    len_val_dataset = int(args[TRAIN_VAL_SPLIT]*len(train_val_dataset))
    split = [len(train_val_dataset) - len_val_dataset, len_val_dataset]
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, split, generator)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args[BATCH_SIZE], shuffle=True, **args[KWARGS])
    val_loader = DataLoader(val_dataset, batch_size=args[BATCH_SIZE],  shuffle=True, **args[KWARGS])
    test_loader = DataLoader(test_dataset,batch_size=args[BATCH_SIZE], shuffle=True, **args[KWARGS])

    print()
    print(f"Loaded dataset {args[NAME]}:")
    print(f"\t Train dataset size: {len(train_loader.dataset)}")
    print(f"\t Val dataset size: {len(val_loader.dataset)}")
    print(f"\t Test dataset size: {len(test_loader.dataset)}")
    print()

    return train_loader, val_loader, test_loader

def get_dataset(args: Dict) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Construct dataset for training validation and testing. Data augmentation is also added here.

    :param args: arguments (dictionary) containing information required to build the data loader.

    :return: train_val_dataset, test_dataset
    """
    if args[NAME] == "MNIST":
        dataset = datasets.MNIST
    elif args[NAME] == "FASHION_MNIST":
        dataset = datasets.FashionMNIST
    elif args[NAME] == "CIFAR10":
        dataset = datasets.CIFAR10
    elif args[NAME] == "CIFAR100":
        dataset = datasets.CIFAR100
    else:
        raise ValueError(f"Invalid dataset provided = {args[NAME]}")

    train_val_dataset = dataset(args[PATH], train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomRotation(args[TRANSFORM][ROTATION]),
            transforms.RandomCrop(size=args[TRANSFORM][SIZE],padding=args[TRANSFORM][PADDING]),
            transforms.ToTensor()
            ])
        )
    test_dataset = dataset(args[PATH], train=False, download=True, transform=transforms.ToTensor())
    return train_val_dataset, test_dataset
