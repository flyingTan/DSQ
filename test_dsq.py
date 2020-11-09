from DSQConv import DSQConv
import torch
import torch.nn as nn


class DsqTestModel(nn.Module):
    def __init__(self):
        super(DsqTestModel, self).__init__()
        self.conv1 = DSQConv(3, 8, 3, 1, 1, bias= None, QInput= False)
        self.fc = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    x = torch.randn((2,3, 5, 5))
    y = torch.randn((2, 10))
    test_model = DsqTestModel()
    out = test_model(x)
    loss = torch.abs(y - out).sum()
    loss.backward()
    print(out)
    
    