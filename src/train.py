import torch
import torch.nn as nn
import torch.optim as optim

from model import DisasterCNN

model = DisasterCNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Model Ready ✅")
print("Loss Function:", criterion)
print("Optimizer:", optimizer)