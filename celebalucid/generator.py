import numpy as np
import pandas as pd
import cv2
import os
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torchvision import transforms

from celebalucid.utils import download_test_data


def build_generator(workdir, dataset_name, batch_size=32, shuffle=False, transform=None, **kwargs):
    verbose = kwargs.get('verbose', True)
    csv = download_test_data(workdir, dataset_name, verbose)
    dataset = Generator(csv, transform=transform)

    num_workers = kwargs.get('num_workers', 1)
    drop_last = kwargs.get('drop_last', True)
    pin_memory = kwargs.get('pin_memory', False)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             drop_last=drop_last)
    return data_loader


class Generator(Dataset):
    def __init__(self,
                 csv,
                 transform=None):

        # Read in csv and define root directory
        self.df = pd.read_csv(csv)
        self.root = os.path.split(csv)[0]

        # Convert relative paths to absolute
        self.df.img = self.df.img.apply(lambda x: os.path.join(self.root, x))

        # Transforms
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                lambda x: x * 255 - 117
            ])

    def __len__(self):
        ''' The length of the generator. Df size / batch_size '''
        return len(self.df)
        
    def __getitem__(self, i):
        ''' Get i^th item of the generator as gen[i] '''
        row = self.df.iloc[i]

        # Input
        x = cv2.imread(row.img)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)  
        if self.transform is not None:
            x = self.transform(x)

        # Output
        y = row.drop('img').astype(np.float32).values
        y = (y + 1) / 2. # [-1,1] data to [0,1]

        return x, y
