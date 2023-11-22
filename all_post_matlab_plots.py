import numpy as np
from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torchinfo
import h5py
from utils import *
import os
#import scipy
import scipy.io
from math import pi
#import pyvista as pv
import plotly.graph_objects as go
import plotly.io as pio
import cv2
import matplotlib.cm as cm
from skimage.measure import points_in_poly
from combined_plots_functions3 import *
import matplotlib.colors as mcolors
import colorsys
from scipy.io import savemat
import pandas as pd
from coordinate_comparer_test import *
import glob

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

#these should be updated for your velocity field
u = scipy.io.loadmat('u_data.mat')['u']
v= scipy.io.loadmat('v_data.mat')['v']
w = scipy.io.loadmat('w_data.mat')['w']

grid_size = len(u)
ranglen = 2
sample_fraction = 1

file_pattern = f"PCA_results_slice_{slice}_contour_*.mat"
file_list = glob.glob(file_pattern)
numstruc = len(file_list)  # This is the number of structures
inL=12 #integral length scale
#invol=64
numstructs=numstruc
numintlen=inL

fullindices=range(0,len(u.ravel()),int(1/sample_fraction))
usampled=u.ravel()[fullindices]
vsampled=v.ravel()[fullindices]
wsampled=w.ravel()[fullindices]
xyzmagx = np.sqrt(usampled**2 + vsampled**2 + wsampled**2)
uthreshx = np.sqrt(np.mean(usampled**2))*1.3*0+-1
u2sampled = usampled[xyzmagx > uthreshx]
v2sampled = vsampled[xyzmagx > uthreshx]
w2sampled = wsampled[xyzmagx > uthreshx]
total_kinetic_energy =np.sum( 0.5 * (usampled**2 + vsampled**2 + wsampled**2))

# Initialize empty lists to store the data for each subplot
points_list = []
density_list = []
kinetic_cluster_energy_list = []
rel_E_kin_list = []

for i in range(0, numstructs):
    filename = f'Merged_Grid07_{i}.mat'
    data = loadmat(filename)
    grid = data['grid']
    
    points_value = len(grid[grid>0])
    points_list.append(points_value)
    
    upoints = u[grid>0]
    vpoints = v[grid>0]
    wpoints = w[grid>0]
    
    density_value = (points_value/u2sampled.size)*100
    density_list.append(density_value)
    
    kinetic_cluster_energy_value = np.sum(0.5 * (upoints**2 + vpoints**2 + upoints**2))
    kinetic_cluster_energy_list.append(kinetic_cluster_energy_value)
    
    rel_E_kin_value = 100*kinetic_cluster_energy_value/total_kinetic_energy
    rel_E_kin_list.append(rel_E_kin_value)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
x_vals = np.arange(len(points_list))

label_font_size = 14  # Adjust as needed
title_font_size = 16  # Adjust as needed
tick_font_size = 12  # Adjust as needed

# Subplot 1: points
# ax[0, 0].bar(x_vals, points_list, color=colors[0])
# ax[0, 0].set_title("Points", fontsize=title_font_size)
# ax[0, 0].set_xlabel("Structure Index", fontsize=label_font_size)
# ax[0, 0].set_ylabel("Value", fontsize=label_font_size)
# ax[0, 0].tick_params(axis='both', labelsize=tick_font_size)

# Subplot 2: density
ax[0].bar(x_vals, density_list, color=colors[1])
ax[0].set_title("Volume Fraction", fontsize=title_font_size)
ax[0].set_xlabel("Structure Index", fontsize=label_font_size)
ax[0].set_ylabel("Value (%)", fontsize=label_font_size)
ax[0].tick_params(axis='both', labelsize=tick_font_size)

## Subplot 3: kinetic_cluster_energy
# ax[1, 0].bar(x_vals, kinetic_cluster_energy_list, color=colors[2])
# ax[1, 0].set_title("Kinetic Cluster Energy", fontsize=title_font_size)
# ax[1, 0].set_xlabel("Structure Index", fontsize=label_font_size)
# ax[1, 0].set_ylabel("Energy Value", fontsize=label_font_size)
# ax[1, 0].tick_params(axis='both', labelsize=tick_font_size)

# Subplot 4: rel_E_kin
ax[1].bar(x_vals, rel_E_kin_list, color=colors[3])
ax[1].set_title("Relative Kinetic Energy", fontsize=title_font_size)
ax[1].set_xlabel("Structure Index", fontsize=label_font_size)
ax[1].set_ylabel("Value (%)", fontsize=label_font_size)
ax[1].tick_params(axis='both', labelsize=tick_font_size)

# Adjust layout
plt.tight_layout()

plt.savefig('05General_Merged_Statistics.png')
plt.show()

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Feature names and corresponding colors for the plot
features = ['V', 'S', 'B', 'V3', 'T', 'W', 'L']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Single slice (as specified)
slice = 0




# Features for separate figure
separate_features = ['T', 'W', 'L']

# Define font sizes
title_font_size = 16
label_font_size = 14
tick_font_size = 12# Store data for each feature
feature_data = {feature: [0]*numstructs for feature in features}  # Initializing with zeros

# Loop over contours
for contour in range(numstructs):
    try:
        # Load the data
        file = f"High_Re_minkowski_results_slice_{slice}_contour_{contour}.mat"
        data = scipy.io.loadmat(file)
        key = next((k for k in data if not k.startswith('__')), None)

        if key is not None and key in data and data[key].size > 0:
            minkowski_results = data[key]
            if minkowski_results.ndim != 2 or minkowski_results.shape[0] < len(features):
                print(f"Unexpected data shape in file {file}: {minkowski_results.shape}.")
                continue

            for i, feature in enumerate(features):
                if minkowski_results[i, 0].size > 0:
                    value = minkowski_results[i, 0]
                    # Apply the given conditions
                    if feature in ['B', 'L', 'W', 'T']:
                        value /= numintlen
                    elif feature == 'V':
                        value /= numintlen**3
                    elif feature == 'S':
                        value /= 6*numintlen**2
                    feature_data[feature][contour] = value
                else:
                    print(f"No data for feature {feature} in file {file}.")

        else:
            print(f"No data found in file {file} under key {key}.")

    except FileNotFoundError:
        print(f"File {file} not found.")
        continue

x_vals = range(numstructs)
fig1, axs1 = plt.subplots(len(features) - len(separate_features), 1, figsize=(10, 10))
for i, feature in enumerate([f for f in features if f not in separate_features]):
    axs1[i].bar(x_vals, feature_data[feature], color=colors[i])
    # Updated y-axis labels
    ylabel_map = {
        'B': f"{feature}/L",
        'V': f"{feature}/L^3",
        'S': f"{feature}/6L^2"
    }
    axs1[i].set_ylabel(ylabel_map.get(feature, "E"), fontsize=label_font_size)
    axs1[i].set_xlim([0, numstructs-1])
    axs1[i].set_xlabel('Structure Index', fontsize=label_font_size)
    axs1[i].tick_params(axis='both', labelsize=tick_font_size)
plt.tight_layout()
#print(sum(feature_data['V'])*(inL**3)/(invol**3))
plt.savefig(f"05slice_{slice+1}_main_minkowski_results_distribution.png")
#plt.close(fig1)
plt.show()

# Separate plot for 'T', 'W', and 'L'
fig2, axs2 = plt.subplots(len(separate_features), 1, figsize=(10, 10))
for i, feature in enumerate(separate_features):
    axs2[i].bar(x_vals, feature_data[feature], color=colors[features.index(feature)])
    # Updated y-axis labels
    ylabel_map = {
        'T': f"{feature}/L",
        'W': f"{feature}/L",
        'L': "LM/L"
    }
    axs2[i].set_ylabel(ylabel_map.get(feature, "E"), fontsize=label_font_size)
    axs2[i].set_xlim([0, numstructs-1])
    axs2[i].set_xlabel('Structure Index', fontsize=label_font_size)
    axs2[i].tick_params(axis='both', labelsize=tick_font_size)
plt.tight_layout()
plt.savefig(f"05slice_{slice+1}_TWL_minkowski_results_distribution.png")
#plt.close(fig2)
plt.show()
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Feature names and corresponding colors for the plot
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Single slice (as specified)
slice = 0

# Create the subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Titles for the plots based on the content of the PCA columns
titles = ['Max Length PC1', 'Max Length PC2', 'Max Length PC3', 
          'Standard Deviation PC1', 'Standard Deviation PC2', 'Standard Deviation PC3']

# Store data for each PCA feature
pca_data = np.zeros((numstruc, 6))

# Loop over contours
for contour in range(numstruc):  # contours go from 0 to 115
    try:
        # Load the data
        file = f"PCA_results_slice_{slice}_contour_{contour}.mat"
        data = scipy.io.loadmat(file)

        # Retrieve the first non-special key
        key = next((k for k in data if not k.startswith('__')), None)

        # Make sure the key exists in the dictionary and the data is not empty
        if key is not None and key in data and data[key].size > 0:
            pca_results = data[key]

            # Check the dimensionality of the pca_results
            if pca_results.ndim != 2 or pca_results.shape[0] != 6:
                print(f"Unexpected data shape in file {file}: {pca_results.shape}. Expected a 2D array with 6 rows.")
                continue

            pca_data[contour, :] = pca_results[:, 0]

        else:
            print(f"No data found in file {file} under key {key}.")
            continue

    except FileNotFoundError:
        print(f"File {file} not found.")
        continue

# Generate bar plots for each PCA feature
x_vals = np.arange(numstruc)  # Indices go from 1 to 116

label_font_size = 14
title_font_size = 16
tick_font_size = 12


for idx, ax in enumerate(axs.ravel()):
    ax.bar(x_vals, pca_data[:, idx]/inL, color=colors[idx])
    ax.set_title(titles[idx], fontsize=title_font_size)
    ax.set_xlabel('Structure Index', fontsize=label_font_size)
    ax.set_ylabel('Value/L', fontsize=label_font_size)
    ax.tick_params(axis='both', labelsize=tick_font_size)

# Tighten the layout
plt.tight_layout()

# Save the figure to a file
plt.savefig(f"05slice_{slice+1}_PCA_results_distribution.png")

# Show the figure
plt.show()

