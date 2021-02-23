import argparse
import json

import fc_model

def main():
    """
        Image Classification Prediction
    """
    # Define command line arguments
    args = fc_model.get_predict_args()

    # Use command line values when specified
    if args.image:
        image = args.image     
        
    if args.checkpoint:
        checkpoint = args.checkpoint

    if args.topk:
        topk = args.topk            

    if args.category_names:
        category_names = args.category_names

    #if args.gpu:
    gpu = args.gpu

    print('Image path:', image)
    print('Use GPU:', args.gpu)
    print('Topk:', topk)
    print('Checkpoint:', checkpoint)
    print('Category_names:', category_names)
        
    # Implement the code to predict the class from an image file
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    checkpoint = "./" + checkpoint
    model = fc_model.load_checkpoint(checkpoint,cat_to_name)
    print(model)
    
    probs, classes, labels = fc_model.predict(image, model, cat_to_name, gpu, topk)
    
    print('Category names: ', labels)
    print('Classes: ', classes)
    print('Probabilities: ', probs)
            
    #fc_model.display_image(image, cat_to_name, probs, classes)

# Call to main function to run the program
if __name__ == "__main__":
    main()