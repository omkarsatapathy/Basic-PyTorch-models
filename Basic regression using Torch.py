import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train. view(Y_train. shape [0], 1)
Y_test = Y_test. view(Y_test. shape [0], 1)

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.lin = nn.Linear(n_input_features, 1)

    def forward(self,x):
        y_pred = torch.sigmoid(self.lin(x))
        return y_pred


model = LogisticRegression(n_features)

Learning_rate = 0.05
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = Learning_rate)

epochs = 100

for epoch in range (epochs):
    #forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, Y_train)
    #Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch) % 10 == 0:
        print(f'Epoch {epoch+1} Loss {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f' accuracy = {acc: .4f}')






















