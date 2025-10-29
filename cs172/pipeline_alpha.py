import torch
from tqdm.notebook import tqdm
from torchmetrics import MetricCollection
from cs172.metrics_alpha import ImageAccuracy_alpha, DigitAccuracy_alpha


def train_alpha(model, device, dataloader, lr=1e-3, weight_decay=0.05, num_epoch=10):
    """
    Train the model on alphabetic CAPTCHA dataset (A–Z + a–z)
    Each image contains 5 letters, each letter has 52 classes.
    Output dimension = 5 * 52 = 260
    """
    metric_collection = MetricCollection(
        {
            "image_accuracy": ImageAccuracy_alpha(),
            "digit_accuracy": DigitAccuracy_alpha(),
        }
    ).to(device)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        sum_loss = 0
        metric_collection.reset()

        for img, label in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epoch}"):
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            pred = model(img)
            pred = pred.view(-1, 5, 52)

            targets = label.argmax(dim=-1).long()

            loss = loss_func(pred.permute(0, 2, 1), targets)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            metric_collection.update(pred, label)

        print(f"Epoch {epoch+1}: avg loss = {sum_loss/len(dataloader):.4f}")
        for key, value in metric_collection.compute().items():
            print(f"  {key}: {value.item():.4f}")


def test_alpha(model, device, dataloader):
    """
    Evaluate the model on alphabetic CAPTCHA dataset
    """
    metric_collection = MetricCollection(
        {
            "image_accuracy": ImageAccuracy_alpha(),
            "digit_accuracy": DigitAccuracy_alpha(),
        }
    ).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(dataloader, desc="Testing"):
            img, label = img.to(device), label.to(device)
            pred = model(img)  # [B, 260]
            pred = pred.view(-1, 5, 52)
            metric_collection.update(pred, label)

    return metric_collection.compute()
