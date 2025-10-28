import torch
from torch.utils.data import DataLoader
from cs172.datasets import ImageDataset
from cs172.networks import get_model

dataset = ImageDataset(''data'')
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
model = get_model(''resnet18'')
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.05)
criterion = torch.nn.CrossEntropyLoss()
losses = []
for step, (img, label) in enumerate(loader):
    if step >= 10:
        break
    optimizer.zero_grad()
    pred = model(img)
    logits = pred.view(-1, 10)
    targets = label.argmax(dim=-1).view(-1)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
print(losses)
