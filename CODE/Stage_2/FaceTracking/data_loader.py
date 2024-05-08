import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class CelebADataset(Dataset):
    def __init__(self, root_dir, image_path, split='train', transform=None):
        self.root_dir = root_dir
        self.image_path = image_path
        self.transform = transform
        self.attr_df = pd.read_csv(f'{root_dir}/list_attr_celeba.csv', index_col='image_id')
        self.split_df = pd.read_csv(f'{root_dir}/list_eval_partition.csv', index_col='image_id')
        self.attr_df = self.attr_df.map(lambda x: 0 if x == -1 else 1)
        self.attr_df['Facial_Hair'] = self.attr_df['5_o_Clock_Shadow'] | (1-self.attr_df['No_Beard']) | self.attr_df['Sideburns'] | self.attr_df['Goatee']
        self.attr_df = self.attr_df.drop(['5_o_Clock_Shadow', 'No_Beard', 'Sideburns', 'Goatee', 'Wearing_Necklace', 'Attractive', 'Wearing_Earrings'], axis=1)
        self.image_files = []
        for img_name in self.attr_df.index.tolist():
            img_path = os.path.join(self.image_path, img_name)
            if os.path.exists(img_path):
                if (split == 'train' and self.split_df['partition'][img_name] < 2) or (split == 'test' and self.split_df['partition'][img_name] == 2):
                    self.image_files.append(img_name)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_path, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        attributes = self.attr_df.loc[self.image_files[idx]].values.astype('float32')
        return image, torch.tensor(attributes)
