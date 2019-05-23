import argparse
import json
from PIL import Image
import torch
import numpy as np

from math import ceil
from train import check_gpu
from torchvision import models, transforms


def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    parser.add_argument('image_path', nargs='?', default='flowers/test/1/image_06743.jpg')

    parser.add_argument('checkpoint', 
                        nargs='?',
                        default='model.pth')
    
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.',
                        default=5)
    
    parser.add_argument('--category_names',
                       type=str,
                       default='cat_to_name.json')

    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')

    args = parser.parse_args()
    return args

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state'])
    
    return model

def process_image(image):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    image = Image.open(image) 
    image = test_transforms(image)
    return image


def predict(image_path, model, device, cat_to_name, top_k):
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.to(device)
    model = model.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(image)

    probs = torch.exp(output)
    probs, index = probs.topk(top_k)
    
    probs_top_list = np.array(probs)[0]
    index_top_list = np.array(index[0])
    
    class_to_idx = model.class_to_idx
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list


def print_probability(probs, flowers):
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))

if __name__ == '__main__':
    args = arg_parser()
    
    cat_to_names = args.category_names
    with open(cat_to_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    
    device = check_gpu(gpu_arg=args.gpu);
    
    top_probs, top_labels = predict(args.image_path, model, device, cat_to_name, args.top_k)
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    print_probability(top_flowers, top_probs)