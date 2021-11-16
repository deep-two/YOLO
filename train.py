import torch
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import yaml

from tensorboardX import SummaryWriter


log_dir = '/log'
writer = SummaryWriter(log_dir)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train YOLOv2')

    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
    
    parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
    
    parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
    
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)

    parser.add_argument('-f')
    args = parser.parse_args()
    return args


def train(num_epoch, start_epoch=1, learning_rate, step=1, log_dir
          model, writer, device, validation_epoch=3, train_loader, validation_loader):
    
    model = model.to(device)
    min_valid_loss = 1.0
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params=model.parameters(), lr= learning_rate, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    
    for epoch in range(start_epoch, num_epoch+1):
        epoch_loss = []
        epoch_acc = []
        tepoch = tqdm(train_loader, unit="batch")
        
        for x_train, y_train in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            x_train = x_train.to(device)
            y_train = y_train.to(device)


            outputs = model(x_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()


            acc = (outputs.argmax(dim=1).cpu() == y_train.cpu()).numpy().sum() / len(outputs)
            epoch_acc.append(acc)
            epoch_loss.append(loss.item())
            
            
            if step % 10 == 0:
                writer.add_scalar('train_acc', acc, step)
                writer.add_scalar('train_loss', loss.item(), step)
            
        
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * acc)
            step += 1
            
        print(f"Train. Epoch: {epoch :d}, Loss: {np.mean(epoch_loss):1.5f}, acc: {np.mean(epoch_acc)*100 :1.5f}%")
        
        
        if epoch % validation_epoch == 0:
            validation_loss = []
            validation_acc = []
            model.eval()
            
            for x_valid, y_valid in valid_loader:
                with torch.no_grad():
                
                    x_valid = x_valid.to(device)
                    y_valid = y_valid.to(device)

                    outputs = model(x_valid)
                    val_loss = criterion(outputs, y_valid)
                    val_acc = (outputs.argmax(dim=1).cpu() == y_valid.cpu()).numpy().sum() / len(outputs)
                    validation_loss.append(val_loss.item())
                    validation_acc.append(val_acc)
                    
            validation_loss = np.mean(validation_loss)
            validation_acc = np.mean(validation_acc)
            print("Val. Epoch: %d , val_loss: % 1.5f, val_acc: %1.5f  \n" % (epoch, validation_loss, validation_acc))
            writer.add_scalar('val_acc', validation_acc, step)
            writer.add_scalar('val_loss', validation_loss, step)
            
            if validation_loss < min_valid_loss:
                min_valid_loss = validation_loss
                torch.save(model.state_dict(), f'{log_dir}/epoch{epoch:d}_valloss{min_valid_loss:.2f}.pth')    
            
            model.train()
            
            pass
        
        


        
if __name__ == '__main__':
    args = parse_args() 
    train()