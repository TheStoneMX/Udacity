
# PROGRAMMER: Oscar A. Rangel
# DATE CREATED:  Nov 25 2018                                
# REVISED DATE: Dec 1 2018
# PURPOSE: It will train a new network on a dataset and save the model as a checkpoint
##
    
# Imports python modules
import argparse
import torch
import os
import traceback
import logging

import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architectures', dest='architectures', default='vgg19', action='store', help='model architectures')
    parser.add_argument('--data_dir', type=str, default='./flowers', help='dir to load images')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='dir to save checkpoints, default checkpoints')
    parser.add_argument('--hidden_units', type=int, default=500, help='hidden units, default 500')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate, default 0.005')
    parser.add_argument('--gpu', dest='gpu', default=True, help='training device, default gpu')
    parser.add_argument('--epochs', type=int, default=3, help='training epochs, default 3')
    parser.add_argument('--num_threads', type=int, default=8, help='thread to training with cpu')

    return parser.parse_args()

 ## ------------------------------------------------------##
 #  ------------------------------------------------------##
def get_dataloders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'training': transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),

        'validating': transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),

        'testing': transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    }

    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validating': datasets.ImageFolder(valid_dir, transform=data_transforms['validating']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing'])
    }

    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'validating': torch.utils.data.DataLoader(image_datasets['validating'], batch_size=64, shuffle=True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=30, shuffle=False)
    }

    class_to_idx = image_datasets['training'].class_to_idx
    return dataloaders, class_to_idx
## ---------------------------------------------------------------------------------------------##
#   ---------------------------------------------------------------------------------------------## 
def model_config(struc, hidden_units):

    model = models.vgg19(pretrained=True)
    classifier_input_size = model.classifier[0].in_features

   # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier_output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1',  nn.Linear(classifier_input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, classifier_output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model
## ---------------------------------------------------------------------------------------------##
#   ---------------------------------------------------------------------------------------------## 
def load_model(struc, learning_rate, hidden_units, class_to_idx):
    model = model_config(struc, hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.zero_grad()
    # Save class to index mapping
    model.class_to_idx = class_to_idx

    return model, optimizer, criterion
## ---------------------------------------------------------------------------------------------##
#  ---------------------------------------------------------------------------------------------##
def train_dataset(model, trainloader, epochs, criterion, optimizer, device='cpu'):
    epochs = epochs
    pass_count = 0

    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for i, (inputs, labels) in enumerate(trainloader):
            pass_count += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print("Epoch: {}/{}... ".format(e+1, epochs),
                    "Loss: {:.4f}".format(running_loss/pass_count))

            running_loss = 0
## ---------------------------------------------------------------------------------------------##
# ---------------------------------------------------------------------------------------------##
def save_checkpoint(file, model, optimizer, struc, learning_rate, epochs):
    checkpoint = {'input_size': 1024,
                  'architectures': 'densenet121',
                  'learing_rate': learning_rate,
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'output_size': 102,
                  'epochs': epochs,
                  'arch': 'densenet121',
                  'state_dict': model.state_dict(),
                  }

    torch.save(checkpoint, file)
## ---------------------------------------------------------------------------------------------##
# ---------------------------------------------------------------------------------------------##
def validation(model, dataloaders, criterion):
    correct = 0
    total = 0
    model.eval()  # turn off dropout
    with torch.no_grad():
        for data in dataloaders:
            images, labels = data
            gpu = torch.cuda.is_available()
            if gpu:
                images = Variable(images.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                images = Variable(images, volatile=True)
                labels = Variable(labels, volatile=True)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
## ---------------------------------------------------------------------------------------------##
# ---------------------------------------------------------------------------------------------##
def main():

    try:
        input_args = get_input_args()
        gpu = torch.cuda.is_available() and input_args.gpu

        dataloaders, class_to_idx = get_dataloders(input_args.data_dir)

        model, optimizer, criterion = load_model(
            input_args.architectures,
            input_args.learning_rate,
            input_args.hidden_units,
            class_to_idx
            )

        if gpu:
            model.cuda()
            criterion.cuda()
        else:
            torch.set_num_threads(input_args.num_threads)

        train_dataset(model, dataloaders['training'], input_args.epochs, criterion, optimizer, device='cpu')

        if input_args.save_dir:
            if not os.path.exists(input_args.save_dir):
                os.makedirs(input_args.save_dir)

            file_path = input_args.save_dir + '/' + input_args.architectures + '_checkpoint.pth'
        else:
            file_path = input_args.architectures + '_checkpoint.pth'
        
        # Now save everything
        save_checkpoint(file_path,model, optimizer, input_args.architectures, input_args.learning_rate,input_args.epochs)

        # And Validate
        validation(model, dataloaders['testing'], criterion)

    except Exception as e:
        logging.error(traceback.format_exc())
        print(e)



if __name__ == "__main__":
    main()