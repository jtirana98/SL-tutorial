import argparse
import random
import numpy as np
import logging
import datetime
import time
import os
import torch
import torch.optim as optim

import my_utils
import application.model as model


device_type = 'cpu' # alternatively this can be 'cuda' or 'mps' in case of a mac
device = torch.device(device_type)

torch.manual_seed(42) # in case you want to "seed" you experiments and ensure that they are the same in every run

# --- START initialize the dataset -- for this we can use the Flower framework
# the following dictionary defines the characteristics of the data partition, for this toy example we only have one client 
# and therefore it does not really matter.
partition_args = {
    'dataset_name': 'ylecun/mnist', ## Change this datasets for any dataset name available in https://huggingface.co/datasets.
    'num_clients': 1,
    'partition_method': 'IID', # TODO
    'partition_target': 'label',
    'alpha': None,    # for dirichlet
    'shard_size' : 10,
    'num_shards_per_partition' : 10
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
# ---- END initializing the model

# initialize optimizers
client_optimizer = optim.SGD(client_model.parameters(), lr=0.05, weight_decay=0)
server_optimizer = optim.SGD(server_model.parameters(), lr=0.05, weight_decay=0)

myiter = iter(trainloaders[0])
total_batch = len(trainloaders)
batch_iter = 0

# let' start training
client_model.train()
server_model.train()

client_model.to(device)
server_model.to(device)
criterion = torch.nn.CrossEntropyLoss()

epoch_loss = 0
while batch_iter < total_batch:
    batch_iter += 1
    
    # client side -- forward propagation
    batch = next(myiter)
    key_ = 'image'
    label_ = 'label'
    inputs, labels = batch[key_], batch[label_]

    inputs = inputs.to(device)
    labels = labels.to(device)
    client_optimizer.zero_grad()

    my_outa = client_model(inputs)
    my_outa.requires_grad_(True)

    det_out_a = my_outa.clone().detach().requires_grad_(True) # this are the intermediate activations
    det_out_a.to(device)

    # server side
    # forward propagation
    server_optimizer.zero_grad()
    out = server_model(det_out_a)

    # backward propagation
    loss = criterion(out, labels)
    epoch_loss += loss.item()
    loss.backward()
    server_optimizer.step()

    grad_a = det_out_a.grad.clone().detach() # compute the intermediate gradients for the clients
    grad_a.to(device)

    # client side -- backward propagation
    my_outa.backward(grad_a) # take gradients from helper
    client_optimizer.step()

print(f'The total loss of the epoch is {epoch_loss/total_batch}')