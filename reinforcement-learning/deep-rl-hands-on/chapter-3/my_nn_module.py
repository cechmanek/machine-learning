
import torch
import torch.nn as nn


class MyNNModule(nn.Module):
  def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
    super().__init__()

    self.pipe = nn.Sequential(
      nn.Linear(num_inputs, 5),
      nn.ReLU(),
      nn.Linear(5,20),
      nn.ReLU(),
      nn.Linear(20, num_classes),
      nn.Dropout(p=dropout_prob),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    return self.pipe(x)


my_module = MyNNModule(3, 5, 0.4)

inputs = torch.tensor([[1,2,5]], dtype=torch.float32)
print(inputs)

output = my_module(inputs)
print(output)