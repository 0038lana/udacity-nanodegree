import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    parser.add_argument('data_dir', nargs='?', default='flowers')
    
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models as str',
                        default='vgg16')
    
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints as str. If not specified then model will be lost.',
                        default='.')
    
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate as float',
                        default=0.001)
    
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for DNN classifier as int',
                        default=2048)
    
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int',
                        default=5)

    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')

    args = parser.parse_args()
    return args

def train_transformer(train_dir):
   train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

   train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
   return train_data

def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    
def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=32)
    return loader

def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

def modelloader(architecture="vgg16"):
    if architecture == "vgg16":
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else:
        print("Not implemented for this arhitecture, use vgg16 instead")
        exit(-1)
    
    for param in model.parameters():
        param.requires_grad = False 
    return model

def classifier(hidden_units=2048):    
    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, 102, bias=True),
        nn.LogSoftmax(dim=1)
    )
    
    return classifier

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def network_trainer(model, trainloader, validloader, device, 
                   criterion, optimizer, print_every, epochs=5):
    steps = 0
    # Train Model
    for epoch in range(epochs):
        running_loss = 0
        model.train() 
        
        for inputs, labels in trainloader:
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)
            
                print(f'Epoch {epoch+1}/{epochs}')
                print(f'Train loss: {running_loss/print_every}')
                print(f'Validation loss: {valid_loss/len(validloader):.4f}')
                print(f'Validation accuracy: {100 * (accuracy/len(validloader)):.2f}%')
            
                running_loss = 0
                model.train()

    return model

def validate_model(model, testloader, device):
   # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

def checkpoint(model, classifier, optimizer, train_data, save_dir='.'):
    if isdir(save_dir):
        model.class_to_idx = train_data.class_to_idx

        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'class_to_idx': model.class_to_idx,
            'classifier': classifier,
        }

        torch.save(checkpoint, 'model.pth')

if __name__ == '__main__':
    args = arg_parser()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = test_transformer(train_dir)
    valid_data = train_transformer(valid_dir)
    test_data = train_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    model = modelloader(architecture=args.arch)
    
    model.classifier = classifier(args.hidden_units)
     
    device = check_gpu(gpu_arg=args.gpu);
    
    model.to(device);
   
    learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 30
    steps = 0
    
    trained_model = network_trainer(model, trainloader, validloader, 
                                    device, criterion, optimizer, print_every, 
                                    args.epochs)
    
    validate_model(trained_model, testloader, device)
    
    checkpoint(trained_model, model.classifier, optimizer, train_data, args.save_dir)