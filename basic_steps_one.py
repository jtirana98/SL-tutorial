import argparse
import random
import numpy as np
import logging
import datetime
import time
import os

import torch
import my_utils


device_type = 'cpu' # alternatively this can be 'cuda' or 'mps' in case of a mac
device = torch.device(device_type)

torch.manual_seed(42) # in case you want to "seed" you experiments and ensure that they are the same in every run

# --- START initialize the dataset -- for this we can use the Flower framework
partition_args = {
    'dataset_name': 'mnist', ## Change this datasets for any dataset name available in https://huggingface.co/datasets.
    'num_clients': 1,
    'partition_method': 'IID', # TODO
    'partition_target': None,
    'alpha': None,    # for dirichlet
    'shard_size' : None,
    'num_shards_per_partition' : None
}


trainloaders, valloaders, testloader, df_list = my_utils.execute_partition_and_plot(partition_args, args) ## there is a single testloader

# --- START initialize the dataset -- for this we can use the Flower framework