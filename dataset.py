import json
import random
import sys
import os
import numpy as np
from torchvision.transforms import InterpolationMode, transforms
import math
import einops
import torchvision
from PIL import Image
#    torch.utils.data.DataLoader
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
def center_crop_arr(pil_image, image_size):
    """
    #(2560, 1440)->640,360
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    #(455, 256)
    arr = np.array(pil_image)#(256, 455, 3)
    crop_y = (arr.shape[0] - image_size) // 2#0
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])
TO_LMDB = False
CC3M_595K_PATH = '/mnt/data/user/lidehu/imagenet_data/lavis/CC3M_595K/ages'
file_path = '/mnt/data/user/lidehu/vas.json'
def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)
  # 将PIL Image或者numpy.ndarray转换为Tensor
    # 应用自定义的归一化到[-1, 1]
class CC3MDataset(Dataset):

    def __init__(self, root, img_shape=(256, 256)):
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        #self.filenames = sorted([f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        # with open(file_path, 'r') as f:
        #     self.filenames = json.load(f)
       ######可以提取前准备好数据路径
        with open("/mnt/data/user/lidehu/vae/z_finale_merged_filenames.json", 'r') as f:
            self.filenames = json.load(f)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
     #########添加异常判断
        while True:
            try:
                path = self.filenames[index]
                img = Image.open(path).convert('RGB')
                if img.size == (224, 224):
                    pipeline = transforms.Compose([
                transforms.Resize(self.img_shape),
                transforms.ToTensor(),
                #normalize_01_into_pm1#v
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)#V2
            ])
                else:
                    pipeline = transforms.Compose([
                  transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, 256)),
                  transforms.ToTensor(),
                    #normalize_01_into_pm1#v
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)#V2
    ])
                return pipeline(img)
            except Exception as e:
                index = random.randint(0, len(self.filenames) - 1)
def get_dataloader(type,
                   batch_size,
                   img_shape=None,
                   dist_train=False,
                   num_workers=16,
                   use_lmdb=False,
                   **kwargs):
    
    
    if img_shape is not None:
            kwargs['img_shape'] = img_shape
    if use_lmdb:
        dataset = CC3MDataset(CC3M_595K_PATH, **kwargs)
    else:
        dataset = CC3MDataset(CC3M_595K_PATH, **kwargs)

    if dist_train:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=num_workers)
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
        return dataloader

