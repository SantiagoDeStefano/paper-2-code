import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, val_loader
from model import AlexNet

def main():
    num_epoches = 90
    device = torch.device('cpu')

    model = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        # Uses the model parameters
        model.parameters(),
        # Control the size can make each weight update
        lr=0.01,
        # How much of the history of GD carries over to the current GD
        momentum=0.9,
        # Apply a small shrinking that pulls weight toward zero
        weight_decay=0.0005
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        # Use the current optimizer
        optimizer, 
        # The lower is better, it can be 'max' - for example: accuracy/f1/...
        mode='min', 
        # Mutiply LR by 0.1
        factor=0.1, 
        # Wait 3 epoches before cutting LR
        patience=3
    )

    for epoch in range(num_epoches):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate gradient of each weight
            loss.backward()

            # Applying gradient descent
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, y_pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (y_pred == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epoches} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {100*correct/total:.2f}%")

if __name__ == '__main__':
    main()