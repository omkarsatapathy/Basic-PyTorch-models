import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

#-------------------------Data Transformation-------------------------------#

transform = transforms.Compose([
    transforms.Resize((224,224)), # Give shape of 256 for resnet model
    transforms.ToTensor(),
    transforms. Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#-------------------------Inserting the Dataset-------------------------------#

train_dataset = datasets.ImageFolder(root='/Users/omkar/Python Dataset/horse-or-human/train', transform=transform)
test_dataset = datasets.ImageFolder(root='/Users/omkar/Python Dataset/horse-or-human/validation', transform=transform)

#-------------------------Creating data Loader-------------------------------#

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = False)

#-------------------------Make  custom CNN Model-------------------------------#

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, 16*53*53)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return  x

model = CNN()
# model.to(device='mps')

#-------------------------Optimize the models-------------------------------#
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

#-------------------------Train the model-------------------------------#

Epochs = 10

for epoch in range(Epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        output = model(images)
        loss = loss_func(output, labels)

        optimizer.zero_grad()
        optimizer.step()

        if i%100 == 0:
            print('Epoch [{} / {}], Step [{} / {}], Loss: {:.4f}'.format(epoch+1, 10, i+1, len(train_dataloader), loss))


#-------------------------Test the model-------------------------------#
#
# correct = 0,
# total = 0
#
# for i, (images, labels) in enumerate(test_dataloader):
#     output = model (images)
#     _, predicted = torch.max(output.data, 1)
#     correct += (predicted == labels).sum().item()
#     total += labels.size(0)
#
#
# accuracy = correct / total * 100
# print(accuracy)
