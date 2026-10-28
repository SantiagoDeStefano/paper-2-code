from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('Datasets/imagenette2/train', transform=train_transform)
val_dataset = datasets.ImageFolder('Datasets/imagenette2/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=4)

def main():
    x, y = next(iter(train_loader))
    print(x.shape, y.shape)
    print(y)

if __name__ == '__main__':
    main()