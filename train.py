from cap_net import CapsuleNet, cap_net_loss
import constants
from dataset import load_dataset
from test import test, init_model
from time import time
from torch.optim import Adam, lr_scheduler, SGD
from utils import one_hot_encode, load_experiment_args, Logger, BASE_EXPERIMENT, plot_log
import argparse
import csv
import json
import os
import torch

def get_optimizer(model, hyperparameters):
    """
    Returns an optimizer using the provided hyperparamters.

    :param model: The model which is to be trained using the optimizer.
    :param hyperparameters: hyperparameters to be used for selection.

    :return: Optimizer for the provided model.
    """

    if hyperparameters[constants.OPTIMIZER] == "SGD":
        return SGD(
            model.parameters(),
            lr=hyperparameters[constants.LR],
            momentum=hyperparameters[constants.MOMENTUM],
            nesterov=hyperparameters[constants.NESTEROV],
        )
    # By default use Adam.
    return Adam(model.parameters(), lr=hyperparameters[constants.LR])

def train(model, train_loader, val_loader, args):
    """
    Train a CapsuleNet.

    :param model: the CapsuleNet model.
    :param train_loader: torch.utils.data.DataLoader for the training data.
    :param val_loader: torch.utils.data.DataLoader for the validation data.
    :param args: arguments for the experiment

    :return: The trained CapsuleNet model.
    """
    logger = Logger()
    print('='*35, 'Begin Training', '='*35)
    log_file = open(os.path.join(
        args[constants.FILES][constants.SAVE_DIR],
        args[constants.FILES][constants.LOG_FILENAME]
    ), 'w')

    log_writer = csv.DictWriter(log_file, fieldnames=constants.LOG_FIELDS)
    log_writer.writeheader()

    hyperparameters = args[constants.HYPERPARAMETERS]

    t_start = time()
    optimizer = get_optimizer(model, hyperparameters)
    print('Optimizer:', optimizer)
    lr_decay = lr_scheduler.ExponentialLR(optimizer,
                                          gamma=hyperparameters[constants.LR_DECAY])

    best_val_acc = 0

    # will be used control whether or not cuda operations should be performed.
    is_cuda_available = torch.cuda.is_available()

    for epoch in range(hyperparameters[constants.EPOCHS]):
        model.train()  # set model in training mode
        t_curr = time()
        training_loss = 0.
        correct = 0
        for i, (x, y) in enumerate(train_loader):
            y = one_hot_encode(y, args[constants.DATASET][constants.NUM_CLASSES])
            # convert input data into GPU variable (if cuda available)
            if is_cuda_available:
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()  # reset gradients in the optimizer
            y_pred, x_recon = model(x, y)
            loss = cap_net_loss(y, y_pred, x, x_recon,
                                hyperparameters[constants.LAMBDA_RECON])
            loss.backward()  # computes gradients of loss w.r.t. all variables.
            optimizer.step()  # update trainable parameters

            training_loss += loss.item() * x.size(0)
            y_pred = y_pred.data.argmax(1)
            y_true = y.data.argmax(1)
            correct += y_pred.eq(y_true).cpu().sum()

        lr_decay.step()  # decay learning rate

        train_acc = correct/len(train_loader.dataset)
        training_loss /= len(train_loader.dataset)

        # compute validation loss and accuracy.
        val_loss, val_acc = test(model, val_loader, args, is_cuda_available)
        log_writer.writerow(
            dict(epoch=epoch+1, train_loss=training_loss,
                 train_acc=train_acc.item(), val_loss=val_loss,
                 val_acc=val_acc.item()
                 )
        )

        print(f"-- Epoch {epoch+1:02d}, train_loss = {training_loss:.5f},"
              f"training_acc = {train_acc:.5f},"
              f"val_loss = {val_loss:.5f}, val_acc = {val_acc:.5f},"
              f"epoch_time = {time()-t_curr:.1f}, elapsed_time = {time() - t_start:.1f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(
                           args[constants.FILES][constants.SAVE_DIR], f'epoch_{epoch+1:03d}.pkl'
                       ))
            logger.info(f"Updated best model with validation accuracy = "\
                        f"{best_val_acc:5f}")
    log_file.close()

    save_path = os.path.join(
        args[constants.FILES][constants.SAVE_DIR],
        args[constants.FILES][constants.FINAL_WEIGHTS_FILENAME]
    )
    torch.save(model.state_dict(), save_path)
    print(f'[INFO] Saved the final trained model at {save_path}')

    # plot train and validation logs.
    plot_log(args[constants.FILES])
    print('='*35, 'End Training', '='*35)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[Training] CapsuleNet')
    parser.add_argument('-e', '--experiment', default=BASE_EXPERIMENT,
                        help='Experiment to run from the experiments.json')
    parser.add_argument('-w', '--weights', default=None,
                        help='Weights used to initialize the model')

    args = parser.parse_args()

    logger = Logger()
    logger.info(f"Args = {args}")

    logger.info(f"Loading experiment: " \
            f"{constants.LOG_COLOR_OKBLUE}{args.experiment}{constants.LOG_COLOR_ENDC}")

    experiment_args = load_experiment_args(args.experiment)
    print(json.dumps(experiment_args, sort_keys=True, indent=4))

    # create save directory if it doesn't exist
    if not os.path.exists(experiment_args[constants.FILES][constants.SAVE_DIR]):
        os.makedirs(experiment_args[constants.FILES][constants.SAVE_DIR])

    # load data
    train_loader, val_loader, test_loader = \
        load_dataset(experiment_args[constants.DATASET])

    model = init_model(logger, experiment_args)
    if args.weights is not None:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    experiment_args[constants.FILES][constants.SAVE_DIR],
                    args.weights
                )
            )
        )

    train(model, train_loader, val_loader, experiment_args)
