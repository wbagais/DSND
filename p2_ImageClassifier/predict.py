import argparse
import time
import os
import copy

import json
import torch
import matplotlib.image as mpimg
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np



import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='checkpoint.pth', type=str, help="model checkpoint")
    parser.add_argument("--image",  type=str, help="image path")
    parser.add_argument("--top_k", default=5 , type=int, help="number of top_k 3")
    parser.add_argument("--category_names", type=str, help="path of the category names in json")    
    parser.add_argument("--gpu",default= 'yes', type=str, help="write yes if you would like to use GPU")
    args = parser.parse_args()
    return args
    
def load_checkpoint(filepath, device):
    if (device == 'cuda'):
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
    model.to(device)
    for param in model.parameters(): 
        param.requires_grad = False
      
    if checkpoint['arch'] == 'vgg16':
        model.classifier = checkpoint['classifier']
    elif checkpoint['arch'] == 'resnet18':
        model.fc = checkpoint['fc']
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    return model

def process_image(image, device):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''     
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                            [0.229, 0.224, 0.225])])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    pil_image.unsqueeze_(0) 
    pil_image = pil_image.to(device)
            
    return pil_image


def predict(image, model, topk, category_names, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''   
    model.to(device)
    model.eval()
        
    image = image.to(device)  
    output = model.forward(image)   
    result, index = output.topk(topk)

    probs = torch.nn.functional.softmax(result.data, dim=1).cpu().numpy()[0]
    classes = index.data.cpu().numpy()[0]

    top_labels_data = index.data.cpu().numpy()
    
    if category_names != None:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
            
        labels = list(cat_to_name.values())
        classes = [labels[x] for x in classes]
    
    print(classes)
    print(probs)
    
    return(probs, classes)
def main():
    args = parse_args()
    checkpoint = args.checkpoint
    image_path = args.image
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    device = torch.device("cuda" if gpu=='yes' and torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(checkpoint, device)

    image = process_image(image_path, device)
    probs, classes = predict(image, model, top_k, category_names, device)
    
 
main()
