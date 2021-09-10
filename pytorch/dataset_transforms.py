import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.datasets import load_wine

"""
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms:
https://pytorch.org/docs/stable/torchvision/transforms.html
On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale
On Tensors
----------
LinearTransformation, Normalize, RandomErasing
Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage
Generic
-------
Use Lambda
Custom
------
Write own class
Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
"""


class ToTensor:

    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)


class MulTransform:

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


class WineDataSet(Dataset):

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y.reshape(-1, 1)
        self.n_samples = self.X.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.X[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


loader = load_wine()
X, y = loader["data"], loader["target"]

"""
dataset = WineDataSet(X, y, transform=ToTensor())
"""
factor = 2
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(factor)])

dataset = WineDataSet(X, y, transform=composed)


first_data = dataset[0]

print(first_data)
