import torch
import torch.nn as nn
import torch.nn.functional as F
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
colors_rgb = generate_colors(50)

# For Matplotlib
colors_mpl = [(r/255, g/255, b/255) for r, g, b in colors_rgb]


device = torch.device( "cpu")
# Instantiate your model
model = UNet(in_channels=1, out_channels=1).to(device)

# Load the model weights
weights_path = 'best_model_U_real_weights_055.pth'
model.load_state_dict(torch.load(weights_path))

# Set the model to evaluation mode
model.eval()

#fig, ax = plt.subplots(5, 2, figsize=(10, 20))

cwd=os.getcwd()
features_t = ['tot E_k', 'rel E_k', 'rel rho']
data_statistics_t = {feature_t: [] for feature_t in features_t}


#change to fit your dataset
u = scipy.io.loadmat('u_data.mat')['u']  
v= scipy.io.loadmat('v_data.mat')['v']
w = scipy.io.loadmat('w_data.mat')['w']


cmap = cm.get_cmap('tab20')

grid_size = len(u)
ranglen = 2     #number of cubes to slice the domain into in each direction so 2 means 8 cubes in total
sample_fraction = 1  #amount of applied subsampling, 1 is factor 1

#initialise full velocity field
fullindices=range(0,len(u.ravel()),int(1/sample_fraction))
usampled=u.ravel()[fullindices]
vsampled=v.ravel()[fullindices]
wsampled=w.ravel()[fullindices]
xyzmagx = np.sqrt(usampled**2 + vsampled**2 + wsampled**2)
uthreshx = np.sqrt(np.mean(usampled**2))*1.3
u2sampled = usampled[xyzmagx > uthreshx]
v2sampled = vsampled[xyzmagx > uthreshx]
w2sampled = wsampled[xyzmagx > uthreshx]
coords_flat = np.arange(0, grid_size,1)
xvals_flat, yvals_flat, zvals_flat = np.meshgrid(coords_flat, coords_flat, coords_flat)

x2f = xvals_flat.ravel()[fullindices]
y2f = yvals_flat.ravel()[fullindices]
z2f = zvals_flat.ravel()[fullindices]

x1f = x2f[xyzmagx > uthreshx]
y1f = y2f[xyzmagx > uthreshx]
z1f = z2f[xyzmagx > uthreshx]

features_t = ['tot E_k', 'rel E_k', 'rel rho']
data_statistics_t = {feature_t: [] for feature_t in features_t}

import numpy as np




# Calculate the step size for each sub-volume
step = grid_size // ranglen

for i in range(ranglen):
    for j in range(ranglen):
        for k in range(ranglen):
            i1, i2 = i * step, (i + 1) * step
            j1, j2 = j * step, (j + 1) * step
            k1, k2 = k * step, (k + 1) * step
            
            # Taking a sub-volume from the grid
            usub_full = u[i1:i2, j1:j2, k1:k2]
            vsub_full = v[i1:i2, j1:j2, k1:k2]
            wsub_full = w[i1:i2, j1:j2, k1:k2]
            
            
            flat_size = usub_full.size
                        
            sample_indices = range(0,flat_size,int(1/sample_fraction))

            #apply the subsampling
            usub_sampled = usub_full.ravel()[sample_indices]
            vsub_sampled = vsub_full.ravel()[sample_indices]
            wsub_sampled = wsub_full.ravel()[sample_indices]
            
            usub = usub_sampled
            vsub = vsub_sampled
            wsub = wsub_sampled 

            #compute the angles for each of the four histograms as well as apply the velocity threshold
            u2,v2,w2,x1,y1,z1,azi2,phi2,azi_N, phi_N,azi_Z, phi_Z,azi3_shifted, velocity=velogen(usub,vsub,wsub,grid_size,ranglen,sample_indices)

            u2f,v2f,w2f,azi2f,phi2f,azi_Nf, phi_Nf,azi_Zf, phi_Zf,azi3_shiftedf, velocity=velogen2(u2sampled,v2sampled,w2sampled,grid_size,ranglen)

            #generate the 4 histograms
            H,xedges,yedges,X,Y=histogen(azi2,phi2,azi_N, phi_N,azi_Z, phi_Z,azi3_shifted,0)
            H_s,xedges_s,yedges_s,X_s,Y_s=histogen(azi2,phi2,azi_N, phi_N,azi_Z, phi_Z,azi3_shifted,1)
            H_N,xedges_N,yedges_N,X_N,Y_N=histogen(azi2,phi2,azi_N, phi_N,azi_Z, phi_Z,azi3_shifted,2)
            H_Z,xedges_Z,yedges_Z,X_Z,Y_Z=histogen(azi2,phi2,azi_N, phi_N,azi_Z, phi_Z,azi3_shifted,3)

            Hf,xedgesf,yedgesf,Xf,Yf           =histogen(azi2f,phi2f,azi_Nf, phi_Nf,azi_Zf, phi_Zf,azi3_shiftedf,0)
            H_sf,xedges_sf,yedges_sf,X_sf,Y_sf=histogen(azi2f,phi2f,azi_Nf, phi_Nf,azi_Zf, phi_Zf,azi3_shiftedf,1)
            H_Nf,xedges_Nf,yedges_Nf,X_Nf,Y_Nf=histogen(azi2f,phi2f,azi_Nf, phi_Nf,azi_Zf, phi_Zf,azi3_shiftedf,2)
            H_Zf,xedges_Zf,yedges_Zf,X_Zf,Y_Zf=histogen(azi2f,phi2f,azi_Nf, phi_Nf,azi_Zf, phi_Zf,azi3_shiftedf,3)                                                                                                     

            #Apply U-net and convert the prediction to contours
            sorted_contours,loaded_arrayx_rescaled=contourgen(H,model)
            sorted_contours_s,loaded_arrayx_rescaled_s=contourgen(H_s,model)
            sorted_contours_N,loaded_arrayx_rescaled_N=contourgen(H_N,model)
            sorted_contours_Z,loaded_arrayx_rescaled_Z=contourgen(H_Z,model)

            plt.tight_layout()
            
            #compute kinetic energy and the volume fraction of the entire volume
            total_kinetic_energy =np.sum( 0.5 * (usub**2 + vsub**2 + wsub**2))
            total_kinetic_energy2 =np.sum( 0.5 * (u2**2 + v2**2 + w2**2))
            density=u2.size/usub.size
            
            features = ['E_k', 'rel E_k', 'N', 'rho']
            data_statistics = {feature: [] for feature in features}
            data_statistics_t['tot E_k'].append(int(total_kinetic_energy))
            data_statistics_t['rel E_k'].append(int(100*total_kinetic_energy2/total_kinetic_energy))
            data_statistics_t['rel rho'].append(int(density*100))

            plt.close()
            fig, ax = plt.subplots(2,2)

            # Plot the original image
            ax[0,0].imshow(loaded_arrayx_rescaled, cmap='gray', origin='lower', extent=[-180, 180, -90, 90], aspect='auto')
              
            ax[0,0].axis('off')  # Hide axes    
            
            # Plot the original image
            ax[1,0].imshow(loaded_arrayx_rescaled_s, cmap='gray', origin='lower', extent=[-180, 180, -90, 90], aspect='auto') 
            
            ax[1,0].axis('off')  # Hide axes

            # Plot the original image
            ax[0,1].imshow(loaded_arrayx_rescaled_N, cmap='gray', origin='lower', extent=[-180, 180, -90, 90], aspect='auto')
              
            ax[0,1].axis('off')  # Hide axes    
            
            # Plot the original image
            ax[1,1].imshow(loaded_arrayx_rescaled_Z, cmap='gray', origin='lower', extent=[-180, 180, -90, 90], aspect='auto') 
            
            ax[1,1].axis('off')  # Hide axes

            #find bins inside the contours
            binlst,independent_contours=bingen(sorted_contours,xedges,yedges)
            binlst_s,independent_contours_s=bingen(sorted_contours_s,xedges,yedges)
            binlst_N,independent_contours_N=bingen(sorted_contours_N,xedges,yedges)
            binlst_Z,independent_contours_Z=bingen(sorted_contours_Z,xedges,yedges)


            bingrid=np.meshgrid(np.linspace(np.min(xedges), np.max(xedges), len(xedges)-1),np.linspace(np.min(yedges), np.max(yedges), len(yedges)-1))
            bingrid_flat_x = np.round(np.array(bingrid[0]).flatten(),0)
            bingrid_flat_y = np.round(np.array(bingrid[1]).flatten(),0)

            bingrid_s=np.meshgrid(np.linspace(np.min(xedges_s), np.max(xedges_s), len(xedges_s)-1),np.linspace(np.min(yedges_s), np.max(yedges_s), len(yedges_s)-1))
            bingrid_flat_x_s = reverse_shift(np.round(np.array(bingrid_s[0]).flatten(),0))
            bingrid_flat_y_s = np.round(np.array(bingrid_s[1]).flatten(),0)

            bingrid_N=np.meshgrid(np.linspace(np.min(xedges_N), np.max(xedges_N), len(xedges_N)-1),np.linspace(np.min(yedges_N), np.max(yedges_N), len(yedges_N)-1))
            #bingrid_flat_x_N = np.round(np.array(bingrid_N[0]).flatten(),0)
            #bingrid_flat_y_N = np.round(-np.array(bingrid_N[1]).flatten(),0)
            bingrid_flat_y_N,bingrid_flat_x_N=reverse_transformed_angles(np.array(bingrid_N[1]).flatten(),np.array(bingrid_N[0]).flatten())

            bingrid_Z=np.meshgrid(np.linspace(np.min(xedges_Z), np.max(xedges_Z), len(xedges_Z)-1),np.linspace(np.min(yedges_Z), np.max(yedges_Z), len(yedges_Z)-1))
            #bingrid_flat_x_Z = np.round(-np.array(bingrid_Z[0]).flatten(),0)
            #bingrid_flat_y_Z = np.round(-np.array(bingrid_Z[1]).flatten(),0)
            bingrid_flat_y_Z,bingrid_flat_x_Z=reverse_transformed_angles_south(np.array(bingrid_Z[1]).flatten(), np.array(bingrid_Z[0]).flatten())

            xblst=[]
            yblst=[]
            
            for xix in range((len(binlst)+len(binlst_s)+len(binlst_N)+len(binlst_Z))):
                
                if xix<len(binlst) and len(binlst[xix])>0:
                   xblst.append(bingrid_flat_x[binlst[xix]])
                   yblst.append(bingrid_flat_y[binlst[xix]])
                          
                elif xix<len(binlst)+len(binlst_s) and len(binlst_s[xix-len(binlst)])>0:
                   xblst.append(bingrid_flat_x_s[binlst_s[xix-len(binlst)]])
                   yblst.append(bingrid_flat_y_s[binlst_s[xix-len(binlst)]])
                   #print('pic 2:',xix,'_coordinates:',bingrid_flat_x_s[binlst_s[xix-len(binlst)]],bingrid_flat_y_s[binlst_s[xix-len(binlst)]])
                elif xix<len(binlst)+len(binlst_s)+len(binlst_N) and len(binlst_N[xix-len(binlst)-len(binlst_s)])>0:
                   xblst.append(bingrid_flat_x_N[binlst_N[xix-len(binlst)-len(binlst_s)]])
                   yblst.append(bingrid_flat_y_N[binlst_N[xix-len(binlst)-len(binlst_s)]])
                   #print('pic 3:',xix,'_coordinates:',bingrid_flat_x_N[binlst_N[xix-len(binlst)-len(binlst_s)]],bingrid_flat_y_N[binlst_N[xix-len(binlst)-len(binlst_s)]])

                elif xix<len(binlst)+len(binlst_s)+len(binlst_N)+len(binlst_Z) and len(binlst_Z[xix-len(binlst)-len(binlst_s)-len(binlst_N)])>0:
                   xblst.append(bingrid_flat_x_Z[binlst_Z[xix-len(binlst)-len(binlst_s)-len(binlst_N)]])
                   yblst.append(bingrid_flat_y_Z[binlst_Z[xix-len(binlst)-len(binlst_s)-len(binlst_N)]])
                   #print('pic 4:',xix,'_coordinates:',bingrid_flat_x_Z[binlst_Z[xix-len(binlst)-len(binlst_s)-len(binlst_N)]],bingrid_flat_y_Z[binlst_Z[xix-len(binlst)-len(binlst_s)-len(binlst_N)]])

            #remove duplicate contours
            filtered_x, filtered_y, remaining_indices = compare_and_filter_blobs(xblst, yblst)


            #find points inside bins
            bin_indices=[]
            bin_indices_s=[]
            bin_indices_N=[]
            bin_indices_Z=[]
            ii=1
            
            true_indices =[]
            threshpoint=2000
            for xix in remaining_indices:
                if xix<len(binlst):
                    bin_indices.append(xix)
                    point_indicesx = find_points_in_bins(azi2, phi2, xedges, yedges, binlst[xix])
                    if len(point_indicesx)>threshpoint:
                        point_indices = find_points_in_bins(azi2f, phi2f, xedgesf, yedgesf, binlst[xix])
                        true_indices.append(np.where(point_indices)[0])
                                     
                elif xix<len(binlst)+len(binlst_s):
                    xix2=xix-len(binlst)
                    bin_indices_s.append(xix2)
                    point_indices_sx= find_points_in_bins(azi3_shifted, phi2, xedges_s, yedges_s, binlst_s[xix2])
                    if len(point_indices_sx)>threshpoint:
                        point_indices_s = find_points_in_bins(azi3_shiftedf, phi2f, xedges_sf, yedges_sf, binlst_s[xix2])

                        true_indices.append(np.where(point_indices_s)[0])
                                   
                elif xix<len(binlst)+len(binlst_s)+len(binlst_N):
                    xix3=xix-len(binlst)-len(binlst_s)
                    bin_indices_N.append(xix3)
                    point_indices_Nx = find_points_in_bins(azi_N, phi_N, xedges_N, yedges_N, binlst_N[xix3])
                    if len(point_indices_Nx)>threshpoint:
                        point_indices_N = find_points_in_bins(azi_Nf, phi_Nf, xedges_Nf, yedges_Nf, binlst_N[xix3])

                        true_indices.append(np.where(point_indices_N)[0])
                       
                        
                elif xix<len(binlst)+len(binlst_s)+len(binlst_N)+len(binlst_Z):
                    xix4=xix-len(binlst)-len(binlst_s)-len(binlst_N)
                    bin_indices_Z.append(xix4)
                    point_indices_Zx = find_points_in_bins(azi_Z, phi_Z, xedges_Z, yedges_Z, binlst_Z[xix4])
                    if len(point_indices_Zx)>threshpoint:
                        point_indices_Z = find_points_in_bins(azi_Zf, phi_Zf, xedges_Zf, yedges_Zf, binlst_Z[xix4])

                        true_indices.append(np.where(point_indices_Z)[0])

                ii+=1

            #plot and save the points as well as compute the statistics
            ii=0    
            newpointlst=compare_and_filter_indices(true_indices)
            for xixx in newpointlst:
                xix=remaining_indices[xixx]
                trueind=true_indices[xixx]
                point_indices=np.full(len(x1f),False)
                point_indices[trueind]=True
                if xix<len(binlst):  
                    contour=independent_contours[xix]
                    xcoord = contour[:, 1] * (360 / 35) - 180
                    ycoord = ((17 - contour[:, 0]) * (180 / 17) - 90)             
                    color = cmap(ii )  
                    ax[0,0].plot(xcoord, ycoord, color=colors_mpl[ii-1], label=f'Contour {ii}') 
                    #print('pic 1:',xix,'_coordinates: X',bingrid_flat_x[binlst[xix]],'Y',bingrid_flat_y[binlst[xix]])
                    
                    if len(point_indices)>0:
                        
                        pointgen(point_indices,u,x1f,y1f,z1f,ranglen,i,j,k,ii)
                        density, total_kinetic_energy,kinetic_cluster_energy,density2=statgen(point_indices,usub,vsub,wsub,u2sampled,v2sampled,w2sampled,trueind)               
                        data_statistics['E_k'].append(int(kinetic_cluster_energy))
                        data_statistics['rel E_k'].append(int(100*kinetic_cluster_energy/total_kinetic_energy))
                        data_statistics['N'].append(int(len(trueind)))
                        data_statistics['rho'].append(int(density*100)) 
                     
                elif xix<len(binlst)+len(binlst_s):
                    xix2=xix-len(binlst) 
                    contour=independent_contours_s[xix2]
                    xcoord = contour[:, 1] * (360 / 35) - 180
                    ycoord = ((17 - contour[:, 0]) * (180 / 17) - 90) 
                    color = cmap(ii )  
                    ax[1,0].plot(xcoord, ycoord, color=colors_mpl[ii-1], label=f'Contour {ii}')
                    #print('pic 2:',xix,'_coordinates: X',bingrid_flat_x_s[binlst_s[xix-len(binlst)]],'Y',bingrid_flat_y_s[binlst_s[xix-len(binlst)]])
                    
                    if len(point_indices)>0:    
                        pointgen(point_indices,u,x1f,y1f,z1f,ranglen,i,j,k,ii)
                        density, total_kinetic_energy,kinetic_cluster_energy,density2=statgen(point_indices,usub,vsub,wsub,u2sampled,v2sampled,w2sampled,trueind)           
                        data_statistics['E_k'].append(int(kinetic_cluster_energy))
                        data_statistics['rel E_k'].append(int(100*kinetic_cluster_energy/total_kinetic_energy))
                        data_statistics['N'].append(int(len(trueind)))
                        data_statistics['rho'].append(int(density*100)) 
                   
                elif xix<len(binlst)+len(binlst_s)+len(binlst_N):
                    xix3=xix-len(binlst)-len(binlst_s)        
                    contour=independent_contours_N[xix3]
                    xcoord = contour[:, 1] * (360 / 35) - 180
                    ycoord = ((17 - contour[:, 0]) * (180 / 17) - 90) 
                    color = cmap(ii ) 
                    ax[0,1].plot(xcoord, ycoord, color=colors_mpl[ii-1], label=f'Contour {ii}')
                    #print('pic 3:',xix,'_coordinates: X',bingrid_flat_x_N[binlst_N[xix-len(binlst)-len(binlst_s)]],'Y',bingrid_flat_y_N[binlst_N[xix-len(binlst)-len(binlst_s)]])
                    
                    if len(point_indices)>0:            
                        pointgen(point_indices,u,x1f,y1f,z1f,ranglen,i,j,k,ii)
                        density, total_kinetic_energy,kinetic_cluster_energy,density2=statgen(point_indices,usub,vsub,wsub,u2sampled,v2sampled,w2sampled,trueind)                         
                        data_statistics['E_k'].append(int(kinetic_cluster_energy))
                        data_statistics['rel E_k'].append(int(100*kinetic_cluster_energy/total_kinetic_energy))
                        data_statistics['N'].append(int(len(trueind)))
                        data_statistics['rho'].append(int(density*100)) 
                        
                elif xix<len(binlst)+len(binlst_s)+len(binlst_N)+len(binlst_Z):
                    xix4=xix-len(binlst)-len(binlst_s)-len(binlst_N)  
                    contour=independent_contours_Z[xix4]
                    xcoord = contour[:, 1] * (360 / 35) - 180
                    ycoord = ((17 - contour[:, 0]) * (180 / 17)) - 90
                    color = cmap(ii )  
                    ax[1,1].plot(xcoord, ycoord, color=colors_mpl[ii-1], label=f'Contour {ii}')
                    #print('pic 4:',xix,'_coordinates: X',bingrid_flat_x_Z[binlst_Z[xix-len(binlst)-len(binlst_s)-len(binlst_N)]],'Y',bingrid_flat_y_Z[binlst_Z[xix-len(binlst)-len(binlst_s)-len(binlst_N)]])
                    
                    if len(point_indices)>0:
                        pointgen(point_indices,u,x1f,y1f,z1f,ranglen,i,j,k,ii)
                        density, total_kinetic_energy,kinetic_cluster_energy,density2=statgen(point_indices,usub,vsub,wsub,u2sampled,v2sampled,w2sampled,trueind)            
                        data_statistics['E_k'].append(int(kinetic_cluster_energy))
                        data_statistics['rel E_k'].append(int(100*kinetic_cluster_energy/total_kinetic_energy))
                        data_statistics['N'].append(int(len(trueind)))
                        data_statistics['rho'].append(int(density*100)) 
                ii+=1

            ax[0,1].legend(loc="upper left")                        
            ax[0,0].legend(loc="upper left")
            ax[1,0].legend(loc="upper left")
            ax[1,1].legend(loc="upper left")

            plt.savefig(f'High_Re_10_slice_plot_{i}_{j}_{k}.png', dpi=300)

            #This figure is used to show the direct U-net detection without any contour selection
            fig2, ax2 = plt.subplots(2,2)

            # Plot the original image
            ax2[0,0].imshow(loaded_arrayx_rescaled, cmap='gray', origin='lower', extent=[-180, 180, -90, 90], aspect='auto')
              
            ax2[0,0].axis('off')  # Hide axes    
            
            # Plot the original image
            ax2[1,0].imshow(loaded_arrayx_rescaled_s, cmap='gray', origin='lower', extent=[-180, 180, -90, 90], aspect='auto') 
            
            ax2[1,0].axis('off')  # Hide axes

            # Plot the original image
            ax2[0,1].imshow(loaded_arrayx_rescaled_N, cmap='gray', origin='lower', extent=[-180, 180, -90, 90], aspect='auto')
              
            ax2[0,1].axis('off')  # Hide axes    
            
            # Plot the original image
            ax2[1,1].imshow(loaded_arrayx_rescaled_Z, cmap='gray', origin='lower', extent=[-180, 180, -90, 90], aspect='auto') 
            
            ax2[1,1].axis('off')  # Hide axes

            
            ii=0
            for contour_idx, contour in enumerate(independent_contours):
                if len(contour) > 0:
                    #for xo in range(len(contour)):
                        #ontour0 = contour[contour_idx]
                        xcoord = contour[:, 1] * (360 / 35) - 180
                        ycoord = ((17 - contour[:, 0]) * (180 / 17) - 90) 
                        #ycoord = np.clip(ycoord, -90, 90)
                        color = cmap(ii ) 
                        ax2[0,0].plot(xcoord, ycoord, color=colors_mpl[ii-1], label=f'Contour {ii}') 
                        ii+=1
            ii=0
            for contour_idx, contour in enumerate(independent_contours_s):
                if len(contour) > 0:
                    #for xo in range(len(contour)):
                        #contour0 = contour[contour_idx]
                        xcoord = contour[:, 1] * (360 / 35) - 180
                        ycoord = ((17 - contour[:, 0]) * (180 / 17) - 90) 
                        #ycoord = np.clip(ycoord, -90, 90)
                        color = cmap(ii ) 
                        ax2[1,0].plot(xcoord, ycoord, color=colors_mpl[ii-1], label=f'Contour {ii}')
                        ii+=1
            ii=0
            for contour_idx, contour in enumerate(independent_contours_N):
                if len(contour) > 0:
                    #for xo in range(len(contour)):
                        #contour0 = contour[contour_idx]
                        xcoord = contour[:, 1] * (360 / 35) - 180
                        ycoord = ((17 - contour[:, 0]) * (180 / 17) - 90) 
                        #ycoord = np.clip(ycoord, -90, 90)
                        color = cmap(ii ) 
                        ax2[0,1].plot(xcoord, ycoord, color=colors_mpl[ii-1], label=f'Contour {ii}')
                        ii+=1
            ii=0
            for contour_idx, contour in enumerate(independent_contours_Z):
                if len(contour) > 0:
                    #for xo in range(len(contour)):
                        #contour0 = contour[contour_idx]
                        xcoord = contour[:, 1] * (360 / 35) - 180
                        ycoord = ((17 - contour[:, 0]) * (180 / 17)) - 90
                        #ycoord = np.clip(ycoord, -90, 90)
                        color = cmap(ii )  
                        ax2[1,1].plot(xcoord, ycoord, color=colors_mpl[ii-1], label=f'Contour {ii}')
                        ii+=1
            

            #plt.show()                      
            plt.savefig(f'High_Re_10_slice_unfiltered_plot_{i}_{j}_{k}.png', dpi=300)
            #plt.savefig(f'High_Re_plot_{i}.png', dpi=300)
            df = pd.DataFrame(data_statistics, columns=features, index=range(len(data_statistics['E_k'])))
            #df = pd.DataFrame(data_statistics, columns=features,index=range(ii) )
            df.to_csv(f'High_Re_10_slice_Statistics_Slice_{i}_{j}_{k}.csv')
            
df_t = pd.DataFrame(data_statistics_t, columns=features_t,index=range(len(data_statistics_t['tot E_k'])))
df_t.to_csv(f'High_Poles_Re_General_Statistics.csv')
#plt.show()

#plt.show()
