import argparse
import random
import numpy as np
import logging
import datetime
import time
import os

import torch
import my_utils
import application.model as model


device_type = 'cpu' # alternatively this can be 'cuda' or 'mps' in case of a mac
device = torch.device(device_type)

torch.manual_seed(42) # in case you want to "seed" you experiments and ensure that they are the same in every run

# --- START initialize the dataset -- for this we can use the Flower framework
# the following dictionary defines the characteristics of the data partition, for this toy example we only have one client 
# and therefore it does not really matter.
partition_args = {
    'dataset_name': 'mnist', ## Change this datasets for any dataset name available in https://huggingface.co/datasets.
    'num_clients': 1,
    'partition_method': 'IID', # TODO
    'partition_target': None,
    'alpha': None,    # for dirichlet
    'shard_size' : None,
    'num_shards_per_partition' : None
}

trainloaders, valloaders, testloader, df_list = my_utils.execute_partition_and_plot(partition_args) ## note there is a single testloader

# --- END initialize the dataset -- for this we can use the Flower framework

# --- START initializing the model
input_dim=(16 * 4 * 4)
hidden_dims=[120, 84]
cut_layer = 3

client_model = model.SimpleCNNMNIST(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=10, first_cut=-1, last_cut=cut_layer) # this will return only the model part that starts from the first layer and ends at the third
server_model = model.SimpleCNNMNIST(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=10, first_cut=cut_layer, last_cut=-1) # return the second model part 
# in the application folder we will show a faster way to define the models for all entinies --> using the get_model functions
# ---- END 