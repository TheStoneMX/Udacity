
# PROGRAMMER: Oscar A. Rangel
# DATE CREATED:  Nov 25 2018                                
# REVISED DATE: Dec 1 2018
# PURPOSE: It will train a new network on a dataset and save the model as a checkpoint
##
    
# Imports python modules
import argparse
from time import time
import json
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

import traceback
import logging

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./flowers/test/77/image_00114.jpg', type=str, help='Input image')
    parser.add_argument('--checkpoint', default='./checkpoints/vgg19_checkpoint.pth', type=str,help='checkpoint to predict')
    parser.add_argument('--top_k', type=int, default=5, help='top_k lasses')
    parser.add_argument('--gpu', dest='gpu',action='store_true', help='training device')
    parser.add_argument('--cat_names', default='cat_to_name.json', type=str,help='cat to names')
    parser.set_defaults(gpu=False)
    return parser.parse_args()

def model_config(struc, hidden_units):

    model = models.vgg19(pretrained=True)
    classifier_input_size = model.classifier[0].in_features

   # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier_output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, classifier_output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model

def model_create(struc, learning_rate, hidden_units, class_to_idx):
    # Load model
    model = model_config(struc, hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.zero_grad()
    # Save class to index mapping
    model.class_to_idx = class_to_idx

    return model, optimizer, criterion

def load_checkpoint(file):
    checkpoint = torch.load(file)
    class_to_idx = checkpoint['class_to_idx']
    learning_rate = checkpoint['learing_rate']
    model, optimizer, criterion = model_create('vgg19',learning_rate, 500, class_to_idx)
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.epochs = checkpoint['epochs']

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    width, height = image.size

    short = width if width < height else height
    long = height if height > width else width

    new_short, new_long = 256, int(256/short*long)

    im = image.resize((new_short, new_long))

    left, top = (new_short - 224) / 2, (new_long - 224) / 2
    area = (left, top, 224+left, 224+top)
    img_new = im.crop(area)
    np_img = np.array(img_new)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_img = (np_img / 255 - mean) / std
    image = np.transpose(np_img, (2, 0, 1))

    return image.astype(np.float32)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    gpu = torch.cuda.is_available()
    image = Image.open(image_path)
    np_image = process_image(image)
    model.eval()

    tensor_image = torch.from_numpy(np_image)

    if gpu:
        tensor_image = Variable(tensor_image.float().cuda())
    else:
        tensor_image = Variable(tensor_image)

    tensor_image = tensor_image.unsqueeze(0)
    output = model.forward(tensor_image)
    ps = torch.exp(output).data.topk(topk)

    probs = ps[0].cpu() if gpu else ps[0]
    classes = ps[1].cpu() if gpu else ps[1]

    inverted_class_to_idx = {model.class_to_idx[c]: c for c in model.class_to_idx}

    mapped_classes = list( inverted_class_to_idx[label] for label in classes.numpy()[0])

    return probs.numpy()[0], mapped_classes

def main():

    try:
        input_args = get_input_args()

        model = load_checkpoint(input_args.checkpoint)

        model.cuda()

        use_mapping_file = False

        if input_args.cat_names:
            with open(input_args.cat_names, 'r') as f:
                cat_to_name = json.load(f)
                use_mapping_file = True

        probs, classes = predict(input_args.input, model, input_args.top_k)

        for i in range(input_args.top_k):
            print("probability of class {}: {}".format(classes[i], probs[i]))

    except Exception as e:
        logging.error(traceback.format_exc())
        print(e)

if __name__ == "__main__":
    main()