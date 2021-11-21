import argparse
import logging
from configparser import ConfigParser

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.yolov2 import YOLO
from model.loss import get_loss
from utils.datasets import PascalVOCDataset


# import yaml


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train YOLOv2')

    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='./data/pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate',
                        help='',
                        default=0.1)
    parser.add_argument('--epoch', dest='epoch',
                        help='',
                        default=10)
    parser.add_argument('--validation_epoch', dest='validation_epoch',
                        help='',
                        default=10)
    parser.add_argument('--use_tensorboard', dest='use_tensorboard',
                        help='',
                        action='store_true')
    parser.add_argument('--min_valid_loss', dest='min_valid_loss',
                        help='',
                        default=1.0)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='',
                        default=1e-3)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='',
                        default=64)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='',
                        default=0)

    args, remaining_argv = parser.parse_known_args()

    if args.cfg_file:
        config_parser = ConfigParser()
        config_parser.read([args.cfg_file])
        defaults = dict(config_parser.items())

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    return args


def train(config):
    logger = logging.getLogger('train')

    pascal_train_dataset = PascalVOCDataset(
        image_set="train",
        root=config.dataset
    )

    pascal_valid_dataset = PascalVOCDataset(
        image_set="val",
        root=config.dataset
    )

    pascal_train_dataloader = DataLoader(
        pascal_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=pascal_train_dataset.collater
    )

    pascal_valid_dataloader = DataLoader(
        pascal_valid_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=pascal_valid_dataset.collater
    )

    train_features, train_labels = next(iter(pascal_train_dataloader))

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    label = train_labels[0]
    print(f"Label: {label}")

    device = "cuda" if config.cuda is True else "cpu"

    model = YOLO(is_training=True).to(device)
    # model = config.model.to(device)
    logger.info(model)

    log_dir = './logs'
    writer = SummaryWriter(log_dir)

    min_valid_loss = config.min_valid_loss
    criterion = get_loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    for epoch in range(config.start_epoch, config.epoch + 1):
        epoch_loss = []
        epoch_acc = []
        tepoch = tqdm(pascal_train_dataloader, unit="batch")

        step = 0
        for x_train, y_train in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            x_train = x_train.to(device)
            y_train = y_train.to(device)

            outputs = model(x_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            print(loss)
            loss.backward()
            optimizer.step()

            acc = (outputs.argmax(dim=1).cpu() == y_train.cpu()).numpy().sum() / len(outputs)
            epoch_acc.append(acc)
            epoch_loss.append(loss)

            if step % 10 == 0:
                writer.add_scalar('train_acc', acc, step)
                writer.add_scalar('train_loss', loss.item(), step)

            tepoch.set_postfix(loss=loss.item(), accuracy=100. * acc)
            step += 1

        print(f"Train. Epoch: {epoch :d}, Loss: {np.mean(epoch_loss):1.5f}, acc: {np.mean(epoch_acc) * 100 :1.5f}%")

        if epoch % config.validation_epoch == 0:
            validation_loss = []
            validation_acc = []
            model.eval()

            for x_valid, y_valid in pascal_valid_dataloader:
                with torch.no_grad():
                    x_valid = x_valid.to(device)
                    y_valid = y_valid.to(device)

                    outputs = model(x_valid)
                    val_loss = criterion(outputs, y_valid)
                    val_acc = (outputs.argmax(dim=1).cpu() == y_valid.cpu()).numpy().sum() / len(outputs)
                    validation_loss.append(val_loss.item())
                    validation_acc.append(val_acc)

            validation_loss = np.mean(validation_loss)
            validation_acc = np.mean(validation_acc)

            # print("Val. Epoch: %d , val_loss: % 1.5f, val_acc: %1.5f  \n" % (epoch, validation_loss, validation_acc))
            logger.info(
                "Val. Epoch: %d , val_loss: % 1.5f, val_acc: %1.5f  \n" % (epoch, int(validation_loss), int(validation_acc)))

            writer.add_scalar('val_acc', validation_acc, step)
            writer.add_scalar('val_loss', validation_loss, step)

            if validation_loss < min_valid_loss:
                min_valid_loss = validation_loss

                saved_model_dir = f'{log_dir}/epoch{epoch:d}_val_loss{min_valid_loss:.2f}.pth'
                torch.save(model.state_dict(), saved_model_dir)
                logger.info("The Validation loss is updated. A New trained model saved to ", saved_model_dir)

            model.train()

            pass


if __name__ == '__main__':
    train_args = parse_args()
    train(train_args)

    # train(
    #     model=YOLO(), writer=None, device=device, train_loader=pascal_train_dataloader,
    # )
