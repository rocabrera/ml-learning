import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

print(y.shape,X.shape)
y = y.view(y.shape[0], 1)
print(y.shape)
n_samples, n_features = X.shape

# 1-) model
input_dim = n_features
output_dim = 1
model = nn.Linear(input_dim, output_dim)

# 2-) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3-) training loop
num_epochs = 100

for epoch in range(num_epochs):
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward
    loss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")

    
# plot

predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()