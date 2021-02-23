import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim

import fc_model

#from train_func import get_train_arg

# Main program function defined below
def main():
    """
        Image Classification Network Trainer
    """

    # Define command line arguments
    args = fc_model.get_train_args()

    # Use command line values when specified
    if args.data_dir:
        data_dir = args.data_dir
        
    if args.save_dir:
        save_dir = args.save_dir
        
    if args.gpu:
        gpu = args.gpu
        
    if args.epochs:
        epochs = args.epochs
            
    if args.arch:
        arch = args.arch 
        
    if args.learning_rate:
        learning_rate = args.learning_rate
        
    if args.hidden_units:
        hidden_units = args.hidden_units
        
    if args.checkpoint:
        checkpoint = args.checkpoint   

    print('Data directory:', data_dir)
    print('Network architecture:', arch)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)
    print('Use GPU:', args.gpu)
    print('Checkpoint:', checkpoint)

    batch_size=32

    image_datasets, dataloaders = fc_model.load_model_data(data_dir,batch_size)     

    # load categories
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    output_size = len(cat_to_name) 
     
    model = fc_model.load_model(arch, hidden_units, output_size)
    # Defining criterion, optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # train a model
    fc_model.train_model(dataloaders, model, criterion, optimizer, epochs, learning_rate, gpu)
              
    # Save the checkpoint	
    save_path = save_dir + 'checkpoint.pth'
    fc_model.save_checkpoint(save_path, image_datasets, model, 
                       optimizer, arch, learning_rate, hidden_units,epochs)
  
# Call to main function to run the program
if __name__ == "__main__":
    main()
