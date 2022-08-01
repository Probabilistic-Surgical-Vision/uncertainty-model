import argparse
import os

import torch
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms

import yaml

from loaders import DaVinciDataset, CityScapesDataset
from model import RandomlyConnectedModel, RandomDiscriminator

import train
from train.loss import GeneratorLoss


parser = argparse.ArgumentParser()

parser.add_argument('config', type=str, help='The config file path.')
parser.add_argument('dataset', choices=['da-vinci', 'cityscapes'],
                    help='The dataset to use for training (must be'
                    'either "da-vinci" or "cityscapes").')
parser.add_argument('--epochs', '-e', default=200, type=int,
                    help='The number of epochs to train the model for.')
parser.add_argument('--learning-rate', '-lr', default=1e-4, type=float,
                    help='The initial learning rate for training.')
parser.add_argument('--batch-size', '-b', default=8, type=int,
                    help='The batch size to train/evaluate the model with.')
parser.add_argument('--loss', '-l', choices=['monodepth', 'adversarial'],
                    default='monodepth', help='The loss function to use '
                    '(must be either "monodepth" or "adversarial").')
parser.add_argument('--training-size', default=None, nargs='?', type=int,
                    help='The number of samples to train with.')
parser.add_argument('--validation-size', default=None, nargs='?', type=int,
                    help='The number of samples to evaluate with.')
parser.add_argument('--workers', default=8, type=int,
                    help='The number of workers to use for the dataloader.')
parser.add_argument('--save-model-to', default=None, type=str,
                    help='The path to save models to.')
parser.add_argument('--save-evaluation-to', default=None, type=str,
                    help='The path to save evaluation images to.')
parser.add_argument('--save-model-every', default=10, type=int,
                    help='The number of epochs between saving the model.')
parser.add_argument('--evaluate-every', default=10, type=int,
                    help='The number of epochs between evaluations.')
parser.add_argument('--no-pbar', action='store_true', default=False,
                    help='Prevent program from printing the progress bar.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Prevent program from training model using cuda.')
parser.add_argument('--no-augment', action='store_true', default=False,
                    help='Prevent program from augmenting training images.')
parser.add_argument('--home', default=os.environ["HOME"], type=str,
                    help='Override the home directory (to find datasets).')


if __name__ == '__main__':
    args = parser.parse_args()

    val_label = 'test' if args.dataset == 'da-vinci' else 'val'

    dataset_path = os.path.join(args.home, 'datasets', args.dataset)
    dataset_class = DaVinciDataset if args.dataset == 'da-vinci' \
        else CityScapesDataset

    device = torch.device('cuda') if torch.cuda.is_available() \
        and not args.no_cuda else torch.device('cpu')

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    augment_transform = transforms.Compose([
        train.transforms.ResizeImage((256, 512)),
        train.transforms.RandomFlip(0.5),
        train.transforms.ToTensor(),
        train.transforms.RandomAugment(0.5, gamma=(0.8, 1.2),
                                       brightness=(0.5, 2.0),
                                       colour=(0.8, 1.2))])

    no_augment_transform = transforms.Compose([
        train.transforms.ResizeImage((256, 512)),
        train.transforms.ToTensor()])

    train_transform = no_augment_transform \
        if args.no_augment else augment_transform

    train_dataset = dataset_class(dataset_path, 'train',
                                  train_transform, args.training_size)
    val_dataset = dataset_class(dataset_path, val_label,
                                no_augment_transform, args.validation_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)

    print(f'Dataset size:'
          f'\n\tTrain: {len(train_dataset):,} images.'
          f'\n\tTest: {len(val_dataset):,} images.')

    model = RandomlyConnectedModel(config['model']).to(device)
    loss_function = GeneratorLoss().to(device)

    model_parameters = sum(p.numel() for p in model.parameters())
    print(f'Model has {model_parameters:,} learnable parameters.')
    print(f'Using CUDA? {next(model.parameters()).is_cuda}')

    if args.loss == 'adversarial':
        disc = RandomDiscriminator(config['discriminator']).to(device)
        disc_loss_function = BCELoss().to(device)

        disc_parameters = sum(p.numel() for p in disc.parameters())
        print(f'Discriminator has {disc_parameters:,} learnable parameters.')
        print(f'Using CUDA? {next(disc.parameters()).is_cuda}')

    else:
        disc = None
        disc_loss_function = None

    train.train_model(model, train_loader, loss_function, args.epochs,
                      args.learning_rate, disc, disc_loss_function,
                      val_loader=val_loader, save_model_to=args.save_model_to,
                      save_evaluation_to=args.save_evaluation_to,
                      save_every=args.save_every,
                      evaluate_every=args.evaluate_every,
                      device=device, no_pbar=args.no_pbar)
