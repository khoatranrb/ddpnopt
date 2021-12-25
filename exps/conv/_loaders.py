import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

def get_data(ROOT = 'data', BATCH_SIZE=24):
    data_dir = os.path.join(ROOT, 'CUB_200_2011')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                               transforms.Resize(pretrained_size),
                               transforms.RandomRotation(5),
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomCrop(pretrained_size, padding = 10),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                           ])

    test_transforms = transforms.Compose([
                               transforms.Resize(pretrained_size),
                               transforms.CenterCrop(pretrained_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                           ])

    train_data = datasets.ImageFolder(root = train_dir, 
                                      transform = train_transforms)

    test_data = datasets.ImageFolder(root = test_dir, 
                                     transform = test_transforms)

    train_iterator = data.DataLoader(train_data, 
#                                      shuffle = True, 
                                     batch_size = BATCH_SIZE)

    valid_iterator = data.DataLoader(test_data, 
                                     batch_size = BATCH_SIZE)
    return train_iterator, valid_iterator, len(test_data.classes)
