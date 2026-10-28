from torch.optim import SGD, lr_scheduler
import torch
import torch.nn as nn
from dataset import train_loader, val_loader
from model import ResNet_LameAhh

def main():
    device = 'cpu'
    model = ResNet_LameAhh().to(device)
    epoches = 10
    lr = 0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        params=model.parameters(),
        lr=lr,
        weight_decay=1e-4,
        momentum=0.9
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=0.1
    )

    best_acc = -1.0
    for epoch in range(epoches):
        train_loss = 0.0
        model.train()
        # torch.Size([256, 3, 224, 224]) torch.Size([256])
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * image.size(0)

        train_loss /= len(train_loader.dataset)

        correct = 0.0
        val_acc = 0.0
        model.eval()
        with torch.no_grad():
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)

                output_logits = model(image)
                y_preds = (torch.argmax(output_logits, dim=1) == label)
                correct += y_preds.sum().item()
        val_acc = correct/len(val_loader.dataset)

        if(val_acc > best_acc):
            best_acc = val_acc
            torch.save(model.state_dict(), "ResNet-18/ResNet_LameAhh.pt")
        
        scheduler.step(val_acc)
        print(f"epoch {epoch + 1}/{epoches} | train_loss: {train_loss:.3f} | val_acc: {val_acc:.3f}")

if __name__ == '__main__':
    main()