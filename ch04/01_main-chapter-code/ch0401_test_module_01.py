import torch
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


torch.manual_seed(123)
# x = torch.zeros(10)
# x = torch.randn(10)
x = torch.ones(10)
print(x)
print(x.shape)
model = MyModel()
y = model(x)  # 自动调用 forward
print(y)