import torch
x = torch.ones(2,2, device = 'mps')
print(x.shape)
print(x)
import time
time1 = time.time()
x = torch.randn(4, device = 'mps', requires_grad=True)
# x1 = torch.rand(4000,4000, device = 'mps',requires_grad=True )
# y = x.view(-1,2000,2000)
z = (x*x*2)
# z = z.mean()
print('the Z matrix is : ', z)
v = torch.tensor([1,2,-1,3], dtype=torch.float32, device ='mps')
z.backward(v)
print(x.grad)

time2 = time.time()
time_taken = time2-time1
print(f'time taken to computer is {round(time_taken,7)}')

# Back Prop using Torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
# forward Pass
y_hat = w*x
loss = (y_hat-y)**2
print(loss)

loss.backward()
print(w.grad)