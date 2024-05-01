import torch
import torch.nn as nn
import numpy as np

from sklearn import datasets

import matplotlib.pyplot as plt

# define/create a dataset
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
# convert to Torch tensor
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1) # reshape the output matrix
n_samples, n_features = x.shape
input_shape = n_features
output_shape = 1
# print(n_features)
# model = nn.Linear(input_shape, output_shape)

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_shape, output_shape)


loss = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(),lr = 0.04)

epochs = 100
epoch = []
loss_record = []
for i in range(epochs):
    y_pred = model(x)
    step_loss = loss(y_pred, y)

    step_loss.backward()
    optim.step()
    optim.zero_grad()
    i+=1
    epoch.append(i)
    loss_record.append(step_loss.item())
    if (i) % 10 == 0:
        print(f'Epoch {i+1} Loss {step_loss.item():.4f}')

predicted = model(x).detach().numpy()
plt.scatter(x_numpy, y_numpy)
plt.plot(x_numpy, predicted, 'g')
plt.show()
plt.scatter(epoch, loss_record)
plt.show()