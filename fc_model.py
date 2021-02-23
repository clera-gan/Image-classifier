import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image

def get_train_args():
    ''' Define command line arguments
    '''   
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,help='Path to dataset ')
    parser.add_argument('--save_dir', default="./", action="store",type=str,help='Directory to save training checkpoint file')
    parser.add_argument('--gpu', default= False, action='store', help='Use GPU if available')
    parser.add_argument('--epochs', default= 9, type=int, action='store',help='Number of epochs')
    parser.add_argument('--arch', default="vgg16", type=str, action='store',help='Model architecture')
    parser.add_argument('--learning_rate',default=0.001, type=float, action='store',help='Learning rate')
    parser.add_argument('--hidden_units', default= [4096], type=int, action='store',help='Number of hidden units')
    parser.add_argument('--checkpoint', default="checkpoint", type=str,action='store', help='Save trained model checkpoint to file')

    return parser.parse_args()

def get_predict_args():
    ''' Define command line arguments
    '''   
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image to predict')
    parser.add_argument('--checkpoint', default="checkpoint.pth", type=str,action='store',
                        help='Save trained model checkpoint to file')
    parser.add_argument('--topk', default=5, type=int, help='Return top K predictions')
    parser.add_argument('--category_names',default="cat_to_name.json",action="store",type=str,
                        help='Path to file containing the categories.')
    parser.add_argument('--gpu', default= False, action='store', help='Use GPU if available')

    return parser.parse_args()

def load_model_data(data_dir,batch_size=32):  
    ''' Load the data
    '''   
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    data_transforms = {'train':transforms.Compose([transforms.RandomRotation(45),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(), 
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'test': transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])}

    # Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']), 
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
                   'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True),
                   'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True)}

    return image_datasets,dataloaders

def build_classifier(model, input_size = 25088, output_size = 102, hidden_sizes = [4096], drop_p=0.5):    
    ''' build the classifier
    ''' 
    # Extend the existing architecture with new layers
    od = OrderedDict()
    for i in range(len(hidden_sizes)):
        if i==0:
            od['fc' + str(i + 1)] = nn.Linear(input_size, hidden_sizes[i]) 
        else:
            od['fc' + str(i + 1)] = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])

        od['relu' + str(i + 1)] = nn.ReLU()
        od['dropout' + str(i + 1)] = nn.Dropout(drop_p)

    od['fc' + str(len(hidden_sizes)+1)] = nn.Linear(hidden_sizes[-1], output_size)
    od['output'] = nn.LogSoftmax(dim=1)

    classifier = nn.Sequential(od)
    
    return classifier 

def load_model(arch='vgg16', hidden_sizes = [4096], output_size=102):
    ''' loads in a model
    ''' 
    # Load a pre-trained model
    if arch=='vgg13':
        model = models.vgg13(pretrained=True)
        input_size=model.classifier[0].in_features
    elif arch=='vgg16':
        model = models.vgg16(pretrained=True)
        input_size=model.classifier[0].in_features
    elif arch=='vgg19':
        model = models.vgg19(pretrained=True)
        input_size=model.classifier[0].in_features
    elif arch=='densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'densenet201':
        model = models.densenet201(pretrained=True)
        input_size = model.classifier.in_features

    # Freeze its parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = build_classifier(model, input_size, output_size, hidden_sizes, 0.5)

    return model

def train_model(dataloaders, model, criterion, optimizer, epochs=9, learning_rate=0.001, gpu=False):
    ''' train a model
    '''           
    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda:0")
    else:
        print('Using CPU for training')
        device = torch.device("cpu") 
  
    model.to(device)  
     
    start = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        # Each epoch has a training and validation phase
        for mode in ['train', 'valid']:
            if mode == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in iter(dataloaders[mode]):
                # Move input and label tensors to the GPU
                inputs, labels = inputs.to(device), labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(mode == 'train'):
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                    if mode == 'train':                        
                        loss.backward()
                        optimizer.step()  
                          
                # statistics
                running_loss += loss.item() 
                ps = torch.exp(outputs)
                equality = (labels.data == ps.max(dim=1)[1])
                running_corrects += equality.type(torch.FloatTensor).mean()
        
            epoch_loss = running_loss / len(dataloaders[mode])
            epoch_acc = running_corrects.double() / len(dataloaders[mode])           

            if mode == 'train':
                print("\nEpoch: {}/{} ".format(epoch+1, epochs),
                      "\nTraining Loss: {:.4f}  ".format(epoch_loss)) 
            else:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                print("Validation Loss: {:.4f}  ".format(epoch_loss),
                      "Accuracy: {:.4f}".format(epoch_acc))

    print("Validation Best Accuracy: {:.4f}".format(best_acc))  

    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60)) 

   
def save_checkpoint(filepath, image_datasets, model, optimizer, arch, learning_rate, hidden_units, epochs):   
    ''' Save the checkpoint
    ''' 	   
    model.class_to_idx = image_datasets['train'].class_to_idx

    torch.save({'arch': arch,
                'learning_rate': learning_rate,
                'hidden_units': hidden_units,
                'epochs': epochs,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'class_to_idx': model.class_to_idx},
               filepath)

def load_checkpoint(filepath, cat_to_name):
    ''' Load a checkpoint.
    '''
    chpt = torch.load(filepath) 
    
    arch = chpt['arch']
    hidden_units = chpt['hidden_units']
    output_size = len(cat_to_name) 
    
    model = load_model(arch, hidden_units, output_size)    
    model.class_to_idx = chpt['class_to_idx']
    model.load_state_dict(chpt['state_dict'])
    
    return model

# Implement the code to predict the class from an image file
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''  
    # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    size = 256 
    size_max =10000

    if image.size[0] > image.size[1]:
        image.thumbnail((size_max, 256))
    else:
        image.thumbnail((256, size_max))
        
    # crop out the center 224x224 portion of the image                               
    width, height = image.size
    new_width = 224
    new_height = 224
    left = (width - new_width)/2
    upper = (height - new_height)/2
    right = (width + new_width)/2
    lower = (height + new_height)/2
    image = image.crop((left, upper, right, lower))
     
    # convert color channels to floats 0-1                            
    np_image = np.array(image)/255
                                    
    # Normalize the image   
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
                      
    # color channel to be the first dimension                        
    np_image = np_image.transpose((2, 0, 1))    
                      
    return np_image

def predict(image_path, model, cat_to_name, gpu=False, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
     
    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda:0")
    else:
        print('Using CPU for training')
        device = torch.device("cpu") 
        
    model.to(device) 
    
    image = process_image(Image.open(image_path))  
    if gpu and torch.cuda.is_available():
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor) 
    else:
        image = torch.from_numpy(image).type(torch.FloatTensor)
    image.unsqueeze_(0) 
    
    image.to(device) 
    with torch.no_grad():
        probs = torch.exp(model.forward(image))
        top_probs, top_idcs = probs.topk(topk) # get the top 5 results  
        
    top_probs = top_probs.cpu().detach().numpy().tolist()[0] 
    top_idcs = top_idcs.cpu().detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = dict(map(reversed, model.class_to_idx.items()))     
    top_classes = [idx_to_class[idx] for idx in top_idcs]
    top_labels = [cat_to_name[x] for x in top_classes] 

    return top_probs, top_classes, top_labels

