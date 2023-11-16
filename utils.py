import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


#### Lisiting all Image files
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_data(path):
    hf = h5py.File(path,'r')
    XX = np.array(hf.get('jpdf'))
    YY = np.array(hf.get('label'))
    hf.close()

    # Convert the data to PyTorch tensor and add a channel dimension
    XX = torch.from_numpy(XX).float().unsqueeze(1)
    YY = torch.from_numpy(YY).float().unsqueeze(1)
    return XX, YY


class MaxPooling2D_3x3(nn.Module):
    def __init__(self):
        super(MaxPooling2D_3x3, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

    def forward(self, x):
        return self.pool(x)

class MaxPooling2D_2x2(nn.Module):
    def __init__(self):
        super(MaxPooling2D_2x2, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)


class UpSampling2D_2x2(nn.Module):
    def __init__(self):
        super(UpSampling2D_2x2, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

class UpSampling2D_3x3(nn.Module):
    def __init__(self):
        super(UpSampling2D_3x3, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=3, mode='nearest')


def conv2d_block(in_channels, out_channels, kernel_size=3, batchnorm=True):
    if batchnorm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=16, dropout=0.1, batchnorm=True):
        super(UNet, self).__init__()
        
        # Contracting Path
        self.c1 = conv2d_block(in_channels, n_filters, kernel_size=2, batchnorm=batchnorm)
        self.p1 = MaxPooling2D_2x2()
        self.d1 = nn.Dropout(dropout)

        self.c3 = conv2d_block(n_filters, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
        self.p3 = MaxPooling2D_3x3()
        self.d3 = nn.Dropout(dropout)

        self.c4 = conv2d_block(n_filters * 4, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

        # Expansive Path
        self.up1 = nn.ConvTranspose2d(n_filters * 4, n_filters * 4, kernel_size=3, stride=3)
        self.d4 = nn.Dropout(dropout)
        self.c5 = conv2d_block(n_filters * 4 *2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

        self.up3 = UpSampling2D_2x2()
        self.c8 = nn.Conv2d(n_filters * 4, n_filters , kernel_size=3, padding=1)
        self.d6 = nn.Dropout(dropout)
        self.c9 = conv2d_block(n_filters * 2, n_filters, kernel_size=3, batchnorm=batchnorm)

        self.outputs = nn.Conv2d(n_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.c1(x)
        x = self.p1(x1)
        x = self.d1(x)

        x3 = self.c3(x)
        x = self.p3(x3)
        x = self.d3(x)

        x4 = self.c4(x)

        x = self.up1(x4)

        x = torch.cat([x, x3], dim=1)
        x = self.d4(x)
        x5 = self.c5(x)

        x = self.up3(x5)
        x = self.c8(x)
        x = torch.cat([x, x1], dim=1)
        x = self.d6(x)
        x7 = self.c9(x)

        out = self.sigmoid(self.outputs(x7))

        return out

def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def validation(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss

    val_loss /= len(val_loader.dataset)
    return val_loss
