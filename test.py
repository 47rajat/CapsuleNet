from cap_net import CapsuleNet, show_reconstruction, cap_net_loss
from typing import Dict
import constants
from torch.autograd import Variable
from dataset import load_dataset
from utils import  one_hot_encode, load_experiment_args, Logger, BASE_EXPERIMENT
import argparse
import torch
import json

def test(model, test_loader, args, is_cuda_available=False):
    """
    Test the model on the provided dataset.

    :param model: the CapsuleNet model.
    :param test_loader: torch.utils.data.DataLoader for the test data.
    :param args: arguments for the experiment

    :return 
    """
    assert len(test_loader.dataset) <= 0, "Invalid length of test loader"

    model.eval()
    loss = 0
    correct = 0


    with torch.no_grad():
        for x, y in test_loader:
            y = one_hot_encode(y, args[constants.DATASET][constants.NUM_CLASSES])
            if is_cuda_available:
                x,y = x.cuda(), y.cuda()
            y_pred,x_recon = model(x, None)
            loss += cap_net_loss(y, y_pred, x, x_recon,
                                 args[constants.HYPERPARAMETERS][constants.LAMBDA_RECON]).item() * x.size(0)
            y_pred = y_pred.data.argmax(1)
            y_true = y.data.argmax(1)
            correct += y_pred.eq(y_true).cpu().sum()

    return loss/len(test_loader.dataset), correct/len(test_loader.dataset)

def init_model(logger, experiment_args: Dict):
    is_cuda_available = torch.cuda.is_available()

    # define model
    model = CapsuleNet(
        input_dims=experiment_args[constants.DATASET][constants.INPUT_DIMS],
        num_classes=experiment_args[constants.DATASET][constants.NUM_CLASSES],
        architecture=experiment_args[constants.ARCHITECTURE],
        is_cuda_available=is_cuda_available)

    # Enable CUDA if available.
    if is_cuda_available:
        model.cuda()

    logger.info(f"Model = {model}")

    return model

if __name__ == '__main__':
    logger = Logger()
    parser = argparse.ArgumentParser(description='[Testing] CapsuleNet')
    parser.add_argument('-e', '--experiment', default=BASE_EXPERIMENT,
            help='Which experiment to run from the experiments.json, default is BASE')
    parser.add_argument('-w', '--weights', default=None,
            help='Weights used to initialize the model for testing.')

    args = parser.parse_args()
    logger.info(f"Args = {args}")

    logger.info(f"Loading experiment: " \
            f"{constants.LOG_COLOR_OKBLUE}{args.experiment}{constants.LOG_COLOR_ENDC}")
    experiment_args = load_experiment_args(args.experiment)
    print(json.dumps(experiment_args, sort_keys=True, indent=4))

    # load data
    _, _, test_loader = load_dataset(experiment_args[constants.DATASET])

    model = init_model(logger, experiment_args)

    if args.weights is None:
        raise ValueError(f'{constants.LOG_COLOR_WARNING}[ERROR]{constants.LOG_COLOR_ENDC} '\
                         'Weights not provided for testing')

    model.load_state_dict(torch.load(args.weights))

    is_cuda_available = torch.cuda.is_available()

    loss, acc = test(model, test_loader, experiment_args, is_cuda_available)
    logger.info(f'accuracy = {acc:.5f}, loss = {loss:.5f}')
    show_reconstruction(model, test_loader, 50, experiment_args[constants.FILES], is_cuda_available=is_cuda_available)
