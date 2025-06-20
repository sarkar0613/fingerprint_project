import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from .pair_generator import TrainingPairGenerator
from .dataset import TrainingDataset
from utils.utils import expand_to_3_channels


def prepare_data(data_verify, data_enroll):

    data_verify['data'] = expand_to_3_channels(data_verify['data'])
    data_enroll['data'] = expand_to_3_channels(data_enroll['data'])

    imgs_verify, labels_verify = data_verify['data'].numpy(), data_verify['labels'].numpy()
    imgs_enroll, labels_enroll = data_enroll['data'].numpy(), data_enroll['labels'].numpy()

    imgs_all = np.concatenate((imgs_enroll, imgs_verify), axis=0)
    labels_all = np.concatenate((labels_enroll, labels_verify), axis=0)

    return imgs_all, labels_all


def create_data_loaders(imgs, labels, mean, std, batch_size=128, world_size=1, rank=0):
    label_tensor = torch.tensor(labels)

    generator = TrainingPairGenerator(label_tensor)
    pairs, pair_labels = generator.generate_training_pairs()
    pairs_train, labels_train, pairs_test, labels_test = generator.split_data(pairs, pair_labels)

    train_dataset = TrainingDataset(imgs, pairs_train, labels_train, mean, std)
    test_dataset = TrainingDataset(imgs, pairs_test, labels_test, mean, std)

    if world_size > 1:
        assert batch_size % world_size == 0
        per_device_batch_size = batch_size // world_size

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=per_device_batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=per_device_batch_size, sampler=test_sampler, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
