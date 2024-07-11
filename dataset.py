import json
import sys
import os

import einops
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

# Set this tp `True` and run this script to convert dataset to LMDB format
TO_LMDB = False

# CELEBA_DIR = 'data/celebA/img_align_celeba'
# CELEBA_LMDB_PATH = 'data/celebA/img_align_celeba.lmdb'   /mnt/data/zhendingcheng/old_3090/lavis_dataset/lavis/CC3M_595K/images
CELEBA_HQ_DIR = '/mnt/data/jingli/vae/VQ-VAE/DL-Demos/dldemos/VQVAE/data/celeba_hq_256'
# CELEBA_HQ_LMDB_PATH = 'dldemos/VQVAE/data/celeba_hq_256.lmdb' /mnt/data/user/lidehu/lavis/CC3M_595K/images
CC3M_595K_PATH = '/mnt/data/user/lidehu/lavis/CC3M_595K/images'
file_path = '/mnt/data/user/lidehu/vae/filenames.json'
# CC3M_595K_PATH = '/mnt/workspace/jingli/vae/VQ-VAE/DL-Demos/dldemos/VQVAE/data/celeba_hq_256'

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)
  # 将PIL Image或者numpy.ndarray转换为Tensor
    # 应用自定义的归一化到[-1, 1]
def download_mnist():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    print('length of MNIST', len(mnist))
    id = 4
    img, label = mnist[id]
    print(img)
    print(label)

    # On computer with monitor
    # img.show()

    img.save('work_dirs/tmp_mnist.jpg')
    tensor = transforms.ToTensor()(img)
    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())


class CelebADataset(Dataset):

    def __init__(self, root, img_shape=(64, 64)):
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path)
        pipeline = transforms.Compose([
            
            transforms.Resize(self.img_shape),
            transforms.ToTensor(), normalize_01_into_pm1
        ])
        return pipeline(img)


if TO_LMDB:
    from dldemos.lmdb_loader import ImageFolderLMDB

    class CelebALMDBDataset(ImageFolderLMDB):

        def __init__(self, path, img_shape=(64, 64)):
            pipeline = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.Resize(img_shape),
                transforms.ToTensor(), normalize_01_into_pm1
            ])
            super().__init__(path, pipeline)


class MNISTImageDataset(Dataset):

    def __init__(self, img_shape=(28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.mnist = torchvision.datasets.MNIST(root='./data/mnist')

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index: int):
        img = self.mnist[index][0]
        pipeline = transforms.Compose(
            [transforms.Resize(self.img_shape),
             transforms.ToTensor(), normalize_01_into_pm1])
        return pipeline(img)


class CC3MDataset(Dataset):

    def __init__(self, root, img_shape=(256, 256)):
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        #self.filenames = sorted([f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        # with open(file_path, 'r') as f:
        #     self.filenames = json.load(f)

        with open("/mnt/data/user/lidehu/vae/filenames.json", 'r') as f:
            self.filenames = json.load(f)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path).convert('RGB')
        pipeline = transforms.Compose([
            transforms.Resize(self.img_shape),
           # transforms.RandomResizedCrop(256), 
            transforms.ToTensor(), normalize_01_into_pm1
        ])
        return pipeline(img)


def get_dataloader(type,
                   batch_size,
                   img_shape=None,
                   dist_train=False,
                   num_workers=16,
                   use_lmdb=False,
                   **kwargs):
    if type == 'CelebA':
        if img_shape is not None:
            kwargs['img_shape'] = img_shape
        if use_lmdb:
            dataset = CelebALMDBDataset(CELEBA_LMDB_PATH, **kwargs)
        else:
            dataset = CelebADataset(CELEBA_DIR, **kwargs)
    elif type == 'CelebAHQ':
        if img_shape is not None:
            kwargs['img_shape'] = img_shape
        if use_lmdb:
            dataset = CelebALMDBDataset(CELEBA_HQ_LMDB_PATH, **kwargs)
        else:
            dataset = CelebADataset(CELEBA_HQ_DIR, **kwargs)
    elif type == 'MNIST':
        if img_shape is not None:
            dataset = MNISTImageDataset(img_shape)
        else:
            dataset = MNISTImageDataset()
    elif type == 'CC3M':
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


if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)

    if os.path.exists(CELEBA_DIR):
        dataloader = get_dataloader('CelebA', 16)
        img = next(iter(dataloader))
        print(img.shape)
        N = img.shape[0]
        img = einops.rearrange(img,
                               '(n1 n2) c h w -> c (n1 h) (n2 w)',
                               n1=int(N**0.5))
        print(img.shape)
        print(img.max())
        print(img.min())
        img = transforms.ToPILImage()(img)
        img.save('work_dirs/tmp_celeba.jpg')
        if TO_LMDB:
            from dldemos.lmdb_loader import folder2lmdb
            folder2lmdb(CELEBA_DIR, CELEBA_LMDB_PATH)

    if os.path.exists(CELEBA_HQ_DIR):
        dataloader = get_dataloader('CelebAHQ', 16)
        img = next(iter(dataloader))
        print(img.shape)
        N = img.shape[0]
        img = einops.rearrange(img,
                               '(n1 n2) c h w -> c (n1 h) (n2 w)',
                               n1=int(N**0.5))
        print(img.shape)
        print(img.max())
        print(img.min())
        img = transforms.ToPILImage()(img)
        img.save('work_dirs/tmp_celebahq.jpg')
        if TO_LMDB:
            from dldemos.lmdb_loader import folder2lmdb
            folder2lmdb(CELEBA_HQ_DIR, CELEBA_HQ_LMDB_PATH)

    download_mnist()
