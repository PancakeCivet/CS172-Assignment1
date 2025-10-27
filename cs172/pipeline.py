import torch
from tqdm.notebook import tqdm

from torchmetrics import MetricCollection
from cs172.metrics import ImageAccuracy, DigitAccuracy


def train(model, device, dataloader, lr=1e-3, weight_decay=0.05, num_epoch=10):
    metric_collection = MetricCollection(
        {"image_accuracy": ImageAccuracy(), "digit_accuracy": DigitAccuracy()}
    ).to(device)

    model.to(device)
    model.train()

    # ================== TO DO START ====================
    # define the optimizer and loss_func
    # Adam is a recommended optimizer, you can try different learning rate and weight_decay
    # You can use cross entropy as a loss function
    # ===================================================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        amsgrad=True,
    )
    loss_func = torch.nn.CrossEntropyLoss()
    # =================== TO DO END =====================

    # If you implement the previous code correctly, 10 epoch should be enough
    for epoch in range(num_epoch):
        sum_loss = 0
        for img, label in tqdm(dataloader):
            img, label = img.to(device), label.to(device)
            # ================== TO DO START ====================
            # Get the prediction through the model and call the optimizer
            # ===================================================
            pred = ...
            loss = ...
            # =================== TO DO END =====================
            sum_loss += loss.item()
            metric_collection.update(pred, label)
        print(f"loss for epoch {epoch}:", sum_loss / len(dataloader))
        for key, value in metric_collection.compute().items():
            print(f"{key} for epoch {epoch}:", value.item())


def test(model, device, dataloader):
    metric_collection = MetricCollection(
        {"image_accuracy": ImageAccuracy(), "digit_accuracy": DigitAccuracy()}
    ).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img, label = img.to(device), label.to(device)
            pred = model(img)
            metric_collection.update(pred, label)
    return metric_collection.compute()
