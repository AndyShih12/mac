import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler

def get_dataset(config, distributed=False):
  kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}

  image_augmented_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
  ])

  image_transforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
  ])

  if config.dataset == 'CIFAR10':
    train_dataset = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=image_augmented_transforms)
    test_dataset = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=image_transforms)

  elif config.dataset == 'IMAGENET32':
    from .dataset_imagenet import ImageNetDownSample
    train_dataset = ImageNetDownSample(root=config.data_dir, train=True, transform=image_transforms)
    test_dataset = ImageNetDownSample(root=config.data_dir, train=False, transform=image_transforms)

  elif config.dataset == 'TEXT8':
    from .dataset_text8 import Text8
    data = Text8(root=config.data_dir, seq_len=config.seqlen)
    data_shape = (1,config.seqlen)
    num_classes = 27

    #train_dataset = torch.utils.data.ConcatDataset([data.train, data.valid])
    train_dataset = data.train
    test_dataset = data.test

  if distributed:  
    train_sampler = DistributedSampler(train_dataset, num_replicas=config.world_size, rank=config.local_rank, shuffle=True, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, **kwargs)
  else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)

  return train_loader, test_loader