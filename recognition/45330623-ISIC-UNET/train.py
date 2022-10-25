from dataset import train_loader, val_loader
from modules import model
import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm as tq
from tqdm.notebook import tqdm
from datetime import datetime
import math
from torch.optim.optimizer import Optimizer, required

class Train():
    def __init__(self, model=model, num_epochs=10, batch_size=25, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = model
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate)
    
    def train_model(self):
        # keeping-track-of-losses 
        train_losses = []
        valid_losses = []

        for epoch in range(1, self.num_epochs + 1):
            # keep-track-of-training-and-validation-loss
            train_loss = 0.0
            valid_loss = 0.0
    
            # training-the-model
            model.train()
            for data, target in train_loader:
                # move-tensors-to-GPU 
                data = data.to(self.device)
                target = target.to(self.device)

                target = torch.argmax(target, dim=1)
        
                # clear-the-gradients-of-all-optimized-variables
                self.optimizer.zero_grad()
                # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
                output = model(data)
                # calculate-the-batch-loss
                loss = self.criterion(output, target)
                # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()
                # perform-a-ingle-optimization-step (parameter-update)
                self.optimizer.step()
                # update-training-loss
                train_loss += loss.item() * data.size(0)
        
            # validate-the-model
            model.eval()
            for data, target in val_loader:
        
                data = data.to(self.device)
                target = target.to(self.device)

                target = torch.argmax(target, dim=1)
        
                output = model(data)
        
                loss = self.criterion(output, target)
        
                # update-average-validation-loss 
                valid_loss += loss.item() * data.size(0)
    
            # calculate-average-losses
            train_loss = train_loss/len(train_loader.sampler)
            valid_loss = valid_loss/len(val_loader.sampler)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        
            # print-training/validation-statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))