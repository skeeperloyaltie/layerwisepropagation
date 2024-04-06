import torch
from torchvision import datasets, transforms

class MNISTDataLoader:
    def __init__(self, batch_size=64, transform=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transform

        self.batch_size = batch_size

    def load_data(self):
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
