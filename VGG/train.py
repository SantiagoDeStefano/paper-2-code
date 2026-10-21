import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, val_loader
from model import VGG_D

def main():
    num_epoches = 90
    device = 'cpu'

    model = VGG_D().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        # weight_decay in optim.SGD is L2
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
    )
    for epoch in range(num_epoches):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # .size(0) is the first dimension in which is batch
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)

     # lr /10 if val_loss stops improving
    scheduler.step(val_loss)

    print(f"epoch {epoch}: train_loss {train_loss:.3f}, val_loss {val_loss:.3f}, val_acc {val_acc:.3f}")


if __name__ == "__main__":
    main()