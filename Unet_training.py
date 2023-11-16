
import os, platform, sys, glob, time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import torchinfo

import h5py
from utils import *

###### Hyperparameter
learnrate = 0.0001    # Learning Rate
batchsize = 64       # Images per batch
ep_lim = 250      # Max number of epochs
imwidth = 36        # Image dimension used for training, smaller images computer faster 
imheight = 18                    # big impact on computation time, original image is 75x75
patience_t = 100
#imf.imwidth = imwidth # Set width for imf (image functions)
#imf.imheight=imheight

XX, YY = load_data('data_artif_0.55.h5')

X_train, X_val, y_train, y_val = train_test_split(XX, YY, test_size=0.2, random_state=42)

train_data = CustomDataset(X_train, y_train)
val_data = CustomDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batchsize, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = UNet(in_channels=1, out_channels=1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training and Validation
val_losses = []
train_losses = []
best_val_loss = float('inf')
torchinfo.summary(model,(100,1,36,18),col_names = ('input_size',"output_size","num_params","kernel_size"))


for epoch in range(1, ep_lim + 1):
    train(model, device, train_loader, optimizer, epoch, criterion)
    train_loss = validation(model, device, train_loader, criterion)
    val_loss = validation(model, device, val_loader, criterion)
    print(f'Epoch: {epoch}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_U_real_weights_045.pth')
        patience = 0
    val_losses.append(val_loss)
    train_losses.append(train_loss)
    patience += 1
    if (patience_t < patience):
        
        break

    
hf = h5py.File('history_045.h5','w')
hf.create_dataset('val_loss',data=np.array(val_losses))
hf.create_dataset('train_loss',data=np.array(train_losses))
hf.close()

# Plotting the losses
# plt.figure()
# plt.subplot(426)
# plt.plot(train_losses, label='Training loss')
# plt.plot(val_losses, label='Validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.yscale('log')
# plt.legend()
# import matplotlib.pyplot as plt
# import torch
# from scipy.spatial import ConvexHull
# import numpy as np

# # Assuming XX is already a PyTorch tensor
# XX_tensor = XX.float().to(device)

# # Add a batch dimension if necessary
# if len(XX_tensor.shape) == 3:
#     XX_tensor = XX_tensor.unsqueeze(0)

# model.eval()

# indices = [5]  
# results = []

# with torch.no_grad():
#     for idx in indices:
#         XX_tensor_slice = XX_tensor[idx].unsqueeze(0)  # add batch dimension back
#         output = model(XX_tensor_slice)
#         results.append(output)

# y_nn2 = torch.cat(results).cpu().numpy()


# plt.subplot(421)
# plt.title("dataset image prediction")
# plt.imshow(y_nn2[0,0,:,:].T,cmap='gray')
# plt.subplot(422)

# xox=YY[5,:,:].reshape(1,36,18)
# plt.imshow(xox[0,:,:].T,cmap='gray')
# plt.title("dataset image label")
# plt.subplot(423)

# xox2=XX[5,:,:].reshape(1,36,18)
# plt.title("dataset image")
# plt.imshow(xox2[0,:,:].T,cmap='gray')

# # Load the data
# name1='C:/Users/Austin/OneDrive/Documents/MSc_thesis/AI_data/universal_aspects_reproduction/CNN_Example/real_histogram_array_of_slice_0.npy'
# loaded_array = np.load(name1)

# # Add batch and channel dimensions
# loaded_array = loaded_array.reshape(1, 1, 36, 18)

# # Convert the input to tensor and put it to the correct device
# loaded_array_tensor = torch.from_numpy(loaded_array).float().to(device)

# with torch.no_grad():
#     y_nn = model(loaded_array_tensor)

# y_nn = y_nn.cpu().numpy()

# plt.subplot(424)
# plt.title("real image prediction")
# plt.imshow(y_nn[0,0,:,:].T,cmap='gray')
# plt.subplot(425)

# plt.title("real image")
# plt.imshow(loaded_array[0,0,:,:].T,cmap='gray')

# data=y_nn[0,0,:,:]

# plt.subplot(427)
# plt.title("all contours")
# contour = plt.contour(y_nn[0,0,:,:].T,cmap='gray')
# plt.imshow(loaded_array[0,0,:,:].T,cmap='gray')
