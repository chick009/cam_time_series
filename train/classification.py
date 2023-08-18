
from torch import topk
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch
from typing import cast, Union, List
import time 

def train(model, 
          dataloader,
          criterion = nn.CrossEntropyLoss(),
          nb_epochs = 300, 
          learning_rate = 0.00001, 
          device = 'cpu'):

    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    loss_train_history = []
    accuracy_train_history = []

    for epoch in range(nb_epochs):
        time_start = time.time()

        mean_loss_train = []
        mean_accuracy_train = []
        total_sample_train = []

        for i, batch_data_train in enumerate(dataloader):
            # In each batch of data it contains (B x S x D) and labels
            ts_train, label_train = batch_data_train

            # ------------------- Training the data with CE Loss -------------------# 
            optimizer.zero_grad()
            ts_train = ts_train.permute(0, 2, 1)
            print(ts_train.shape)
            output_train = model(ts_train.float()).to(device)
            loss_train = criterion(output_train.float(), label_train.long())
            
            loss_train.backward()
            optimizer.step()

            # ------------------- Evaluate the data on training dataset -------------#
            total_train = label_train.size(0)
            _, pred_class = torch.max(output_train, 1)
            correct_train = (pred_class == label_train).sum().item()

            mean_loss_train.append(loss_train.item())
            mean_accuracy_train.append(correct_train)
            total_sample_train.append(total_train)

        time_end = time.time()

        if (epoch % 10 == 0):
            print('Epoch [{}/{}], Loss Train: {:.4f}, Accuracy Train: {:.2f}%'
                    .format(epoch + 1, 
                                nb_epochs,
                                    np.mean(mean_loss_train),
                                    (np.sum(mean_accuracy_train)/np.sum(total_sample_train)) * 100))
                                                                
    loss_train_history.append(np.mean(mean_loss_train))
    accuracy_train_history.append(np.sum(mean_accuracy_train)/np.sum(total_sample_train))
    
    # torch.save(model.state_dict(), './model/model.pth')
    return model
