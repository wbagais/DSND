import argparse
import time
import os
import copy

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='flowers', type=str, help="data path ")
    parser.add_argument("--lr", default=0.001, type=float, help="learning_rate")
    parser.add_argument("--epochs", default=5 , type=int, help="number of epochs")
    parser.add_argument("--arch", default='vgg16', type=str, help="Model architecture: ether vgg16 or resnet18")
    parser.add_argument("--num_labels",default= 102, type=int, help="number of labels")
    parser.add_argument("--gpu",default= 'yes', type=str, help="write yes if you would like to use GPU")
    parser.add_argument("--hidden_units",default= 1000, type=int, help="write the number of hidden units")
    parser.add_argument("--save_dir", type=int, help="write the path where you want to save checkpoint.pth")
    
    args = parser.parse_args()
    return args
 
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir,  transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return(trainloader, testloader, validloader, train_data)

def validation(loader, model, criterion, device):
    test_loss = 0
    accuracy = 0
    num_images = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)
            
            num_images += inputs.size(0)
                    
            test_loss += batch_loss.item() * inputs.size(0)
                    
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.sum(equals.type(torch.FloatTensor)).item()
    
    final_accuracy = accuracy/num_images
    final_test_loss = test_loss/num_images
    return(final_accuracy, final_test_loss)

def train_model(trainloader,validloader, model, criterion, optimizer, num_epochs, device):
    steps = 0
    running_loss = 0
    num_images = 0
    print_every = 25
    
    train_losses, valid_losses = [], []
    for epoch in range(num_epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            num_images += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            
            if steps % print_every == 0:
                accuracy, valid_loss = validation(validloader, model, criterion, device)

                print(f"Epoch {epoch+1}/{num_epochs}.. "
                      f"Train loss: {running_loss/num_images:.3f}.. "
                      f"Test loss: {valid_loss:.3f}.. "
                      f"Test accuracy: {accuracy:.3f}")
                running_loss = 0
                model.train()

    return (model)
    
    
    
def main():
    args = parse_args()
    data_dir = args.data_dir
    lr = args.lr
    epochs = args.epochs
    arch = args.arch
    num_labels = args.num_labels
    gpu = args.gpu
    hidden_units = args.hidden_units
    save_dir = args.save_dir
   
    #Use GPU if available, otherwise use the CPU
    device = torch.device("cuda" if gpu=='yes' and torch.cuda.is_available() else "cpu")

    #dataloader
    trainloader, testloader, validloader, train_data = load_data(data_dir)
    
    #initate the model
    if arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        in_features = 512
        print('The model is resnet18')
    else:
        arch = 'vgg16'
        model = models.vgg16(pretrained=True)
        in_features = 25088
        print('The model is vgg16')
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('drpot', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(hidden_units, 500)),
                                            ('relu', nn.ReLU()),
                                            ('drpot', nn.Dropout(p=0.5)),
                                            ('fc3', nn.Linear(500, num_labels)),
                                            ('output', nn.LogSoftmax(dim=1))
                                           ]))
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    if arch == 'resnet18':
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr= lr)
    elif arch == 'vgg16':
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr= lr)
         
    criterion = nn.NLLLoss()
    
    model.to(device);
    #train the model
    model = train_model(trainloader,validloader, model, criterion, optimizer, epochs, device)  
    # Testing the network
    accuracy, test_loss = validation(testloader, model, criterion, device)
    
    print(f"Test loss: {test_loss:.3f}.. "
      f"Test accuracy: {accuracy:.3f}")
    
    #save the model chickpoints
    model.class_to_idx = train_data.class_to_idx
    
    if arch == 'resnet18':
        checkpoint = {'arch': arch,
                      'fc': model.fc,
                      'class_to_idx': model.class_to_idx,
                      'state_dict': model.state_dict()}
    elif arch == 'vgg16':
        checkpoint = {'arch': arch,
                      'classifier': model.classifier,
                      'class_to_idx': model.class_to_idx,
                      'state_dict': model.state_dict()}

    
    if (save_dir == None):
        save_dir = arch + '_checkpoint.pth'
        
    torch.save(checkpoint, save_dir)
    
main()