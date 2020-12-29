import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.fc1 = nn.Linear(10,64)
        self.fc5 = nn.Linear(64,32)
        self.fc8 = nn.Linear(32,2)

    def forward(self,x):
#import pdb; pdb.set_trace()
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.fc8(x)
        return x

    def predict(self,x):
        pred = F.softmax(self.forward(x), dim=1)
        return torch.max(pred, 1)[1]

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)


    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def predict(x):
   x = torch.from_numpy(x).type(torch.FloatTensor)
   ans = model.predict(x)
   return ans.numpy()

def load_data():

    df = pd.read_csv('VI_train.csv', index_col='id').iloc[:,1:]
    df_test = pd.read_csv('VI_test.csv', index_col='id')

    X = df.iloc[:,:-1]
    y = df.iloc[:, -1]
    X_test = df_test

    GENDER_MAPPING = {'Male': 0, 'Female': 1}
    VEHICLE_AGE_MAPPING = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    VEHICLE_DAMAGE = {'Yes': 0, 'No': 1}
    X = X.replace({'Gender': GENDER_MAPPING, 'Vehicle_Age': VEHICLE_AGE_MAPPING, 'Vehicle_Damage':VEHICLE_DAMAGE})
    X_test = X_test.replace({'Gender': GENDER_MAPPING, 'Vehicle_Age': VEHICLE_AGE_MAPPING, 'Vehicle_Damage':VEHICLE_DAMAGE})

    X = (X-X.mean()) / X.std()
    X_test = (X_test - X_test.mean()) / X_test.std()
    print('X.shape', X.shape)
    X = X
    y = y



    X_train, X_valid, y_train, y_valid = train_test_split(X,y,
                                                          test_size=0.25,
                                                          random_state=0)

    X_train_balance = pd.concat([X_train[y_train == 0][:27727], X_train[y_train == 1]])
    y_train_balance = pd.Series(np.concatenate([np.zeros(X_train[y_train == 0][:27727].shape[0]),
                                                np.ones(X_train[y_train == 1].shape[0])]))
    return tuple(torch.from_numpy(i.to_numpy()).float().cuda()
                    for i in [X_train_balance, X_valid, y_train_balance, y_valid])

def train():
    device = torch.device('cuda')
    X_train, X_valid, y_train, y_valid = load_data()

    if False:
        import sklearn.datasets
        X_train, y_train = sklearn.datasets.make_moons(200, noise=0.2)
        X_train = torch.from_numpy(X_train).float().cuda()
        y_train = torch.from_numpy(y_train).long().cuda()

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    SCHEDULER_STEP_SIZE = 3000
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHEDULER_STEP_SIZE, gamma=0.5)

    epochs = 30000
    for i in range(epochs):


        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train.long())
#import pdb; pdb.set_trace()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()
        if i % 100 == 0:
            #print(y_pred.std())
            #print(y_pred[:5], y_train[:5])
            #y_pred = model.predict(X_valid).cpu()
            y_pred = model.forward(X_valid).clone().detach().cpu()
            print(i, loss,
                  criterion(y_pred, y_valid.cpu().long()),
                  f1_score(torch.max(y_pred, 1)[1].numpy(), y_valid.cpu().numpy()))

    y_pred = model.predict(X_train).cpu()
    import pdb; pdb.set_trace()
    print('POSITIVE', y_pred.sum())
    print('F1 SCORE (TrainSet)', f1_score(y_pred.numpy(), y_train.cpu().numpy()))
    y_pred = model.predict(X_valid).cpu()
    print('F1 SCORE (ValidSet)', f1_score(y_pred.numpy(), y_valid.cpu().numpy()))

if __name__ == '__main__':
    train()
