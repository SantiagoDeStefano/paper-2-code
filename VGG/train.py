import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, val_loader
from model import VGG_D_LameAhh

def main():
    num_epoches = 10
    device = 'cpu'

    model = VGG_D_LameAhh().to(device)
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

        # evaluation
        """
        print(images.shape)
        print(labels.shape)
        print(labels)
        
        (256, 3, 224, 224)
        (256)
        [7, 4, 2, 4, ....]
        """
        correct = 0.0
        val_loss = -1.0
        best_acc = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                outputs_logits = model(images)
                # outputs_logits: .shape = ([256, 10])
                # torch.argmax(outputs_logits): printed an index for a whole batch, e.g. 2531; .shape = ([])
                # as it is one scalar
                # torch.argmax(outputs_logits, dim=1): tensor of shape (256,) printed [1, 3, 4, 5, ...] (256 numbers)
                y_pred = (torch.argmax(outputs_logits, dim=1) == labels)
                # .sum() counts the True; .item() turns tensor to plain python number
                correct += y_pred.sum().item() 
        val_loss /= len(val_loader.dataset)
        val_acc = correct/len(val_loader.dataset)

        if(val_acc > best_acc):
            best_acc = val_acc
            torch.save(model.state_dict(), "VGG/VGG_D_LameAhh.pt")

        scheduler.step(val_loss)

        print(f"epoch {epoch+1}/{num_epoches} | train_loss {train_loss:.3f} | val_loss {val_loss:.3f} | val_acc {val_acc:.3f}")


if __name__ == "__main__":
    main()