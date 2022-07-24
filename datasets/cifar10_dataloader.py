import torch
import pandas as pd
import os
import torch.utils.data as data
import natsort
from PIL import Image

import torchvision.transforms as transforms


class Cifar10Dataloader(torch.utils.data.Dataset):

    def __init__(self, main_dir: str, class_file: str, label_column: str, sub_images: tuple):

        all_imgs = os.listdir(main_dir)
        self.main_dir = main_dir
        self.X = natsort.natsorted(all_imgs)
        self.y = pd.read_csv(class_file)[label_column].values
        self.sub_images = sub_images

        self.base_transformers = transforms.Compose([
            transforms.PILToTensor()
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img_loc = os.path.join(self.main_dir, self.X[index])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.base_transformers(image)

        size_x = tensor_image.size(dim=1) // self.sub_images[0]
        size_y = tensor_image.size(dim=2) // self.sub_images[1]
        start_x = 0
        sub_tensors = [torch.rand(size_x * size_y * 3)]

        for cut_x in range(self.sub_images[0]):
            start_y = 0
            for cut_y in range(self.sub_images[1]):
                sub_tensors.append(
                    tensor_image[:, start_x:size_x*cut_x+size_x, start_y:size_y*cut_y+size_y].flatten() / 255)

                start_y += size_y
            start_x += size_x

        return torch.cat(sub_tensors).reshape(1+self.sub_images[0]*self.sub_images[1], size_x*size_y*3)
