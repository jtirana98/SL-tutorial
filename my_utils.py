import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

import torch
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST, CIFAR100, ImageFolder, DatasetFolder, CelebA
from torchvision import datasets, transforms

from datasets import load_dataset
import flwr_datasets
from flwr_datasets.visualization import plot_label_distributions
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, SizePartitioner, ShardPartitioner, DistributionPartitioner, NaturalIdPartitioner
from flwr_datasets.visualization import plot_comparison_label_distribution
from datasets import Dataset
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler


def execute_partition_and_plot(partion_args):
    fds = FederatedDataset(
        dataset=partion_args['dataset_name'],
        partitioners={
            "train": generate_partitioner(partion_args)
        },
    )

    # fig, axes, df_list = plot_label_distributions(
    #     partitioner=fds.partitioners["train"],
    #     label_name=partion_args['partition_target'],
    #     title=f"{partion_args['dataset_name']} - {partion_args['partition_method']} - {partion_args['partition_target']}",
    #     legend=True,
    #     verbose_labels=True,
    # )
    # #plt.show()
    # plt.savefig('distribution.png')
    # print(df_list)


    def apply_transforms_train_mnist(batch):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        key_ = 'image'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch

    def apply_transforms_test_mnist(batch):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        key_ = 'image'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    for partition_id in range(partion_args['num_clients']):
        
        partition = fds.load_partition(partition_id, "train")
        partition = partition.train_test_split(train_size=0.8, seed=42)
        
        partition["train"] = partition["train"].with_transform(apply_transforms_train_mnist)
        partition["test"] = partition["test"].with_transform(apply_transforms_test_mnist)
        
        

        trainloaders.append(DataLoader(partition["train"], batch_size=64))
        valloaders.append(DataLoader(partition["test"], batch_size=64))

    testset = fds.load_split("test").with_transform(apply_transforms_test_mnist)    
    testloader = DataLoader(testset, batch_size=64)
    return trainloaders, valloaders, testloader, []


def generate_partitioner(partion_args):
    ## IID
    if partion_args['partition_method'] == 'iid':
        method = 'iid',
        partition_by=partion_args['partition_target'],    
        partitioner = IidPartitioner(num_partitions=partion_args['num_clients'])           
        
    ## DIRICHLET    
    elif partion_args['partition_method'] == 'dirichlet':
        method = 'dirichlet'
        partitioner = DirichletPartitioner(
            num_partitions=partion_args['num_clients'],
            partition_by=partion_args['partition_target'],
            alpha=partion_args['alpha'],
            seed=partion_args.get('dataset_seed', 42),
            min_partition_size=0)

    ## QUANTITY
    elif partion_args['partition_method'] == 'quantity':
        method = 'quantity'
        partitioner = SizePartitioner(
            num_partitions=partion_args['num_clients'],
            partition_id_to_size_fn=lambda x:x/10 ## How do we parametrize this?
        )
    ## ARCHETYPE
    else:
        method = 'archetype'
        partitioner = ShardPartitioner(
            num_partitions=partion_args['num_clients'],
            partition_by=partion_args['partition_target'],
            num_shards_per_partition=partion_args['num_shards_per_partition'],
            #shard_size = partion_args['shard_size']
        )
    return partitioner