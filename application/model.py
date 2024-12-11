import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10, first_cut=-1, last_cut=-1):
        super(SimpleCNNMNIST, self).__init__()
        self.first_cut = first_cut
        self.last_cut = last_cut

        layer = 0
        start = False
        end = False
        
        if self.last_cut == -1:
            end = True

        if self.first_cut == -1:
            start = True

         # layer 0
        if start or ((not start) and (self.first_cut == layer)):
            if end or ((not end) and (self.last_cut > layer)):
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                start = True
            else:
                return 
        layer += 1

        # layer 1
        if start or (not start and (self.first_cut == layer)):
            if end or (not end and (self.last_cut > layer)):
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                start = True
            else:
                return 
        layer += 1    

        # layer 2
        if start or (not start and (self.first_cut == layer)):
            if end or (not end and (self.last_cut > layer)):
                self.fc1 = nn.Linear(input_dim, hidden_dims[0])
                start = True
            else:
                return 
        layer += 1    

        # layer 3
        if start or (not start and (self.first_cut == layer)):
            if end or (not end and (self.last_cut > layer)):
                self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
                start = True
            else:
                return 
        layer += 1

        # layer 4
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

        
    def forward(self, x):
        layer = 0
        start = False
        end = False
        
        if self.last_cut == -1:
            end = True

        if self.first_cut == -1:
            start = True

        # layer 0
        if start or ((not start) and (self.first_cut == layer)):
            if end or ((not end) and (self.last_cut > layer)):
                x = self.pool(F.relu(self.conv1(x)))
                start = True
            else:
                return x
        layer += 1

        # layer 1
        if start or (not start and (self.first_cut == layer)):
            if end or (not end and (self.last_cut > layer)):
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 4 * 4)
                start = True
            else:
                return x
        layer += 1    

        # layer 2
        if start or (not start and (self.first_cut == layer)):
            if end or (not end and (self.last_cut > layer)):
                x = F.relu(self.fc1(x))
                start = True
            else:
                return x
        layer += 1    

        # layer 3
        if start or (not start and (self.first_cut == layer)):
            if end or (not end and (self.last_cut > layer)):
                x = F.relu(self.fc2(x))
                start = True
            else:
                return x
        layer += 1    
        # layer 4
        x = self.fc3(x)
        return x



def get_model(model_name, dataset, cut, num_clients):
    input_dim=(16 * 4 * 4)
    hidden_dims=[120, 84]
    output_dim=10
    model_part_a = SimpleCNNMNIST(input_dim, hidden_dims, output_dim, -1, cut)
    model_part_b = SimpleCNNMNIST(input_dim, hidden_dims, output_dim, cut, -1)

    nets = []
    for i in range(num_clients):
        nets.append(SimpleCNNMNIST(input_dim, hidden_dims, output_dim, -1, cut))
    return (model_part_a, model_part_b), nets

def get_model_two_split(model_name, dataset, cut_a, cut_b, num_clients):
    input_dim=(16 * 4 * 4)
    hidden_dims=[120, 84]
    output_dim=10
    model_part_a = SimpleCNNMNIST(input_dim, hidden_dims, output_dim, -1, cut_a)
    model_part_b = SimpleCNNMNIST(input_dim, hidden_dims, output_dim, cut_a, cut_b)
    model_part_c = SimpleCNNMNIST(input_dim, hidden_dims, output_dim, cut_b, -1)

    nets_a = []
    nets_c = []
    for i in range(num_clients):
        nets_a.append(SimpleCNNMNIST(input_dim, hidden_dims, output_dim, -1, cut_a))
        nets_c.append(SimpleCNNMNIST(input_dim, hidden_dims, output_dim, cut_b, -1))
    
    return (model_part_a, model_part_b, model_part_c), (nets_a, nets_c)