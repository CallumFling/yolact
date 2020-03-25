import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms.functional as TF
from skimage import io
import accimage

np.random.seed(0)

try:
    from data.config import cfg
except ImportError:

    class Config:
        pass

    cfg = Config()
    cfg.use_amp = False


class LarvaeDataset(Dataset):
    def __init__(self, root_dir):
        # Path here is the path to JPGS
        self.root_dir = root_dir
        structure = [
            sorted(
                [
                    os.path.join(root_dir, folder, file)
                    for file in os.listdir(os.path.join(root_dir, folder))
                ]
            )
            for folder in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, folder))
        ]
        index = []
        mapping = {}
        counter = 0
        for i in range(len(structure)):
            index.append([])
            for j in range(len(structure[i])):
                index[-1].append(counter)
                mapping[counter] = structure[i][j]
                counter = counter + 1
        self.mapping = mapping
        self.length = counter
        self.index = [np.array(i, dtype=np.int) for i in index]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_name = self.mapping[index]
        image = accimage.Image(img_name)
        # Get a square
        image = TF.center_crop(image, min(image.height, image.width))
        image = TF.to_tensor(image)
        if cfg.use_amp:
            return image.half()
        else:
            return image


class VideoSampler(Sampler):
    # This is a batch sampler
    def __init__(self, data_source, batch_size):
        self.batch_size = batch_size
        self.index = data_source.index

    def __iter__(self):
        random_index = [np.random.permutation(i) for i in self.index]
        random_batches = np.concatenate(
            [
                np.reshape(
                    batch[: (batch.size // self.batch_size * self.batch_size)],
                    (-1, self.batch_size),
                )
                for batch in random_index
            ]
        )
        np.random.shuffle(random_batches)
        for i in random_batches:
            yield list(i)

