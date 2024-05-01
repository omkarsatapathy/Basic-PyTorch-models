import torch
import torch.nn as nn
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
n_samples, n_features = X.shape
# w = torch.tensor(300.0, dtype=torch.float32, requires_grad=True)
x_test = torch.tensor([[30]], dtype=torch.float32)
learning_rate = 0.06
n_iters = 400

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, input_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(f'Pred value before training is {model(x_test).item()}')

for epoch in range(n_iters):
    # predict = forward pass with our model
    y_predicted = model(X)

    # loss
    l = loss(Y, y_predicted)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    # w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item(): 3f}, loss = {l: .8f}')


print(f'Pred value after training is {model(x_test).item()}')
