"""
Handles data loader creation

"""
import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchtext.legacy import data, datasets


def _get_cifar_10_train_test_loaders(transform=None,
                                     batch_size=128,
                                     test_batch_size=None,
                                     data_location='/data',
                                     num_workers=4):
    """
    Creates CIFAR10 train and test loaders
    """

    num_test_examples = 10000
    output_dim = 10

    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(root=data_location,
                                            train=True,
                                            download=True,
                                            transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=data_location,
                                           train=False,
                                           download=True,
                                           transform=transform)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size if test_batch_size else batch_size,
        shuffle=False,
        num_workers=num_workers)

    return train_loader, test_loader, num_test_examples, output_dim


def _get_imdb_train_test_loaders(batch_size=128,data_location='/data'):
    """
    Create imdb train and test loader
    taken from https://github.com/bentrevett/pytorch-sentiment-analysis
    """

    MAX_VOCAB_SIZE = 25_000

    TEXT = data.Field(tokenize='spacy',
                      tokenizer_language='en_core_web_sm',
                      batch_first=True)

    LABEL = data.LabelField(dtype=torch.int)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root=data_location)

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors_cache='/data/vector_cache',
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_,
                     )

    LABEL.build_vocab(train_data)

    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data), batch_size=batch_size, device='cuda')

    return train_iterator, test_iterator, 12500, 2, (TEXT, LABEL)


def get_train_test_loaders(transform=None,
                           batch_size=128,
                           test_batch_size=None,
                           data_location='/data',
                           num_workers=4,
                           dataset='cifar10'):
    """
    Creates generic dataset loader 
    and provides dataset information
    """

    if dataset == 'cifar10':
        train_loader, test_loader, num_test_examples, output_dim = _get_cifar_10_train_test_loaders(
            transform=transform,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            data_location=data_location,
            num_workers=num_workers)
        extras = None

    elif dataset == 'imdb':
        train_loader, test_loader, num_test_examples, output_dim, extras = _get_imdb_train_test_loaders(
            batch_size=batch_size, data_location=data_location)

    return train_loader, test_loader, num_test_examples, output_dim, extras
