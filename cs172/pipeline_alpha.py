import torch
from tqdm.notebook import tqdm
from torchmetrics import MetricCollection
from cs172.metrics_alpha import ImageAccuracy_alpha, DigitAccuracy_alpha


def train_alpha(model, device, dataloader, lr=1e-3, weight_decay=5e-4, num_epoch=30):

    metric_collection = MetricCollection(
        {
            "image_accuracy": ImageAccuracy_alpha(),
            "digit_accuracy": DigitAccuracy_alpha(),
        }
    ).to(device)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=1e-5
    )
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        sum_loss = 0
        metric_collection.reset()

        for img, label in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epoch}"):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(img).view(-1, 5, 52)
            targets = label.argmax(dim=-1).long()

            loss = 0.0
            for i in range(5):
                loss += loss_func(pred[:, i, :], targets[:, i])
            loss /= 5

            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            metric_collection.update(pred, label)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}: lr = {cur_lr:.6e}, avg loss = {sum_loss/len(dataloader):.4f}"
        )
        for key, value in metric_collection.compute().items():
            print(f"  {key}: {value.item():.4f}")

        scheduler.step()


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
            pred = model(img)
            pred = pred.view(-1, 5, 52)
            metric_collection.update(pred, label)

    return metric_collection.compute()
