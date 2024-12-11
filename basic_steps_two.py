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
cut_layer_first = 2
cut_layer_second = 3

client_model_a = model.SimpleCNNMNIST(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=10, first_cut=-1, last_cut=cut_layer_first) # this will return only the model part that starts from the first layer and ends at the third
server_model = model.SimpleCNNMNIST(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=10, first_cut=cut_layer_first, last_cut=cut_layer_second) # return the second model part 
client_model_c = model.SimpleCNNMNIST(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=10, first_cut=cut_layer_second, last_cut=-1) # return the third model part 

# in the application folder we will show a faster way to define the models for all entinies --> using the get_model functions
# ---- END initializing the model

# initialize optimizers
client_optimizer_a = optim.SGD(client_model_a.parameters(), lr=0.05, weight_decay=0)
server_optimizer = optim.SGD(server_model.parameters(), lr=0.05, weight_decay=0)
client_optimizer_c = optim.SGD(client_model_c.parameters(), lr=0.05, weight_decay=0)

myiter = iter(trainloaders[0])
total_batch = len(trainloaders)
batch_iter = 0

# let' start training
client_model_a.train()
server_model.train()
client_model_c.train()

client_model_a.to(device)
server_model.to(device)
client_model_c.to(device)

criterion = torch.nn.CrossEntropyLoss()

epoch_loss = 0
while batch_iter < total_batch:
    batch_iter += 1
    
    # client side -- forward propagation model part-a
    batch = next(myiter)
    key_ = 'image'
    label_ = 'label'
    inputs, labels = batch[key_], batch[label_]

    inputs = inputs.to(device)
    labels = labels.to(device)
    client_optimizer_a.zero_grad()
    client_optimizer_c.zero_grad()

    my_outa = client_model_a(inputs)
    my_outa.requires_grad_(True)

    det_out_a = my_outa.clone().detach().requires_grad_(True) # this are the intermediate activations client to server
    det_out_a.to(device)

    # server side forward propagation part-b
    server_optimizer.zero_grad()
    my_outb = server_model(det_out_a)
    my_outb.requires_grad_(True)

    det_out_b = my_outb.clone().detach().requires_grad_(True) # this are the intermediate activations server to client
    det_out_b.to(device)

    # client side -- forward propagation model part c
    out = client_model_c(det_out_b)

    # client backward propagation part-c
    loss = criterion(out, labels)
    epoch_loss += loss.item()
    loss.backward()
    client_optimizer_c.step()

    grad_b = det_out_b.grad.clone().detach() # compute the intermediate gradients for the server
    grad_b.to(device)

    # server side -- backward propagation part-b
    my_outb.backward(grad_b) # take gradients from helper
    server_optimizer.step()
    
    grad_a = det_out_a.grad.clone().detach() # compute the intermediate gradients for the client
    grad_a.to(device)

    # client side -- backward propagation part-a
    my_outa.backward(grad_a) # take gradients from helper
    client_optimizer_a.step()

print(f'The total loss of the epoch is {epoch_loss/total_batch}')