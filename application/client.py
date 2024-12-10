import torch

class Client:
    def __init__(self, id, model, optimizer, dataloader, valloader, total_epoch, dataset_name, device):
        self.id = id

        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.dataset_name = dataset_name
        
        self.dataloader = dataloader
        self.valloader = valloader

        self.local_epoch_count = 0
        self.batch_count = 0
        self.prev_job = 1 # 0: for forward and 1: for backward

        self.total_batch = len(self.dataloader)
        self.total_epoch = total_epoch
        print(self.total_batch)
        
        # other status parameters
        self.labels = None
        self.my_outa = None
        self.det_out_a = None
        self.grad_a = None
        
        self.correct = 0
        self.total = 0
        self.epoch_loss = 0.0

        self.end = False
        self.myiter = iter(self.dataloader)

        self.start_init = False
    
    def start(self):
        self.myiter = iter(self.dataloader)
        self.total_batch = len(self.dataloader)
        self.prev_job = 1

    def next_job(self):
        self.model.train()
        self.prev_job = (self.prev_job + 1) % 2
        
        job = {
            'id' : self.id,
            'type' : self.prev_job
        }

        
        if self.prev_job == 0:
            # forward propagation job
            
            batch = next(self.myiter)
            key_ = 'image'
            label_ = 'label'
            
            inputs, labels = batch[key_], batch[label_]
                
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            self.labels = labels

            self.model.to(self.device)
            self.my_outa = self.model(inputs)
            self.my_outa.requires_grad_(True)
            self.det_out_a = self.my_outa.clone().detach().requires_grad_(True)
            self.det_out_a.to(self.device)
        else:
            # backward propagation job
            self.optimizer.zero_grad()
            self.my_outa.backward(self.grad_a) # take gradients from helper
            self.optimizer.step()

            self.batch_count = (self.batch_count + 1) % self.total_batch
    
            if self.batch_count == 0: # end of the epoch
                

                #self.epoch_loss /= len(self.dataloader.dataset)
                epoch_acc = self.correct / self.total
                # if verbose:
                #print(f"Client {self.id} >> Epoch {self.local_epoch_count+1}: train loss {self.epoch_loss}, accuracy {epoch_acc}")

                self.local_epoch_count += 1
                self.correct = 0
                self.total = 0
                self.epoch_loss = 0.0
                self.myiter = iter(self.dataloader)
                if self.local_epoch_count >= self.total_epoch:
                    self.end = True
                    self.local_epoch_count = 0

        return job