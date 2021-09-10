"""
One epoch -> forward & backward pass of all training samples.

batch_size -> number of training samples in one forward & backward pass.

number of iterations -> number of passes, each pass using [batch_size] number of samples.

e.g. 100 samples, batch_size=20 -> 100/20 = 5 iterations for 1 epoch.
"""
import math

import torch

from sklearn.datasets import load_wine
from torch.utils.data import Dataset, DataLoader


class WineDataSet(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        self.y = y.view(y.shape[0], 1)
        self.n_samples = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


loader = load_wine()
X, y = loader["data"], loader["target"]
dataset = WineDataSet(X, y)

dataloader = DataLoader(dataset=dataset, batch_size=4,
                        shuffle=True, num_workers=2)

dataiter = iter(dataloader)
data = dataiter.next()

features, labels = data
print(features, labels)


# training loop
num_epochs = 2
total_samples = len(dataset)
n_iteratios = math.ceil(total_samples / 4)
print(total_samples, n_iteratios)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if (i + 1) % 5 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iteratios}, inputs {inputs.shape}")

# some famous dataset
# torchvision.datasets.MNIST()
# torchvision.datasets.FashionMNIST()
