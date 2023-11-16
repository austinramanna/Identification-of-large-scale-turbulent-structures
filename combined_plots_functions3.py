import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors as mcolors
import colorsys
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import savemat
# List of colors in RGB format
colors_rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), 
              (255, 0, 255), (192, 192, 192), (128, 0, 0), (128, 128, 0), (0, 128, 0), 
              (128, 0, 128), (0, 128, 128), (0, 0, 128), (255, 165, 0), (255, 20, 147)]




def reverse_shift(azi3_shifted_array):
    return np.where(azi3_shifted_array >= 180, azi3_shifted_array - 360, azi3_shifted_array)

def remove_similar_from_lists(*list_groups):
    for lists in list_groups:
        i = 0
        while i < len(lists):
            j = i + 1
            while j < len(lists):
                # Compute intersection of lists[i] and lists[j]
                intersection = len(np.intersect1d(lists[i], lists[j]))

                # Check for the 60% criteria on both lists
                if intersection > 0.6 * len(lists[i]) or intersection > 0.6 * len(lists[j]):
                    print('yes')
                    # If lists[i] is shorter or equal, remove it, else remove lists[j]
                    if len(lists[i]) <= len(lists[j]):
                        lists.pop(i)
                        i -= 1  # Decrement i to check the next list against the previous lists
                        break  # Break out of the inner loop and recheck with the outer loop
                    else:
                        lists.pop(j)
                        continue  # Continue with the same i in the outer loop
                j += 1
            i += 1

    return list_groups

def remove_similar_from_lists(*list_groups):
    for lists in list_groups:
        i = 0
        while i < len(lists):
            j = i + 1
            while j < len(lists):
                # Compute intersection of lists[i] and lists[j]
                intersection = len(np.intersect1d(lists[i], lists[j]))

                # Check for the 60% criteria on both lists
                if intersection > 0.6 * len(lists[i]) or intersection > 0.6 * len(lists[j]):
                    # If lists[i] is shorter or equal, remove it, else remove lists[j]
                    if len(lists[i]) <= len(lists[j]):
                        lists.pop(i)
                        i -= 1  # Decrement i to check the next list against the previous lists
                        break  # Break out of the inner loop and recheck with the outer loop
                    else:
                        lists.pop(j)
                        continue  # Continue with the same i in the outer loop
                j += 1
            i += 1

    return list_groups

def shift_indices_x(indices, shift_amount, hist_shape):
    # Reshape the indices to a 2D array
    y_indices, x_indices = np.unravel_index(indices, hist_shape)
    
    # Apply the shift in x-direction and use modulo operation to wrap around
    x_indices = (x_indices + shift_amount) % hist_shape[1]
    
    # Convert 2D indices back to flattened index
    shifted_indices = np.ravel_multi_index((y_indices, x_indices), hist_shape)
    
    return shifted_indices

def shift_indices_y(indices, shift_amount, hist_shape):
    # Reshape the indices to a 2D array
    y_indices, x_indices = np.unravel_index(indices, hist_shape)
    
    # Apply the shift in y-direction and use modulo operation to wrap around
    y_indices = (y_indices + shift_amount) % hist_shape[0]
    
    # Convert 2D indices back to flattened index
    shifted_indices = np.ravel_multi_index((y_indices, x_indices), hist_shape)
    
    return shifted_indices


def compute_transformed_angles(u2, v2, w2):
    # Original azimuthal and elevation angles
    azi = np.arctan2(v2, u2)
    xymag = np.sqrt(u2**2 + v2**2)
    phi = np.arctan2(w2, xymag)

    # Cartesian coordinates
    x = xymag * np.cos(azi)
    y = xymag * np.sin(azi)
    z = w2

    # For North Pole moved to the equator
    # Rotation about y-axis by pi/2
    x_prime_north = z
    y_prime_north = y
    z_prime_north = -x
    azi_prime_north = np.arctan2(y_prime_north, x_prime_north)*(180/pi)
    phi_prime_north = np.arctan2(np.sqrt(x_prime_north**2 + y_prime_north**2), z_prime_north)*(180/pi)

    # For South Pole moved to the equator
    # Rotation about y-axis by -pi/2
    x_prime_south = -z
    y_prime_south = y
    z_prime_south = x
    azi_prime_south = np.arctan2(y_prime_south, x_prime_south)*(180/pi)
    phi_prime_south = np.arctan2(np.sqrt(x_prime_south**2 + y_prime_south**2), z_prime_south)*(180/pi)

    return (azi_prime_north, phi_prime_north), (azi_prime_south, phi_prime_south)
def compactplot2(X,Y,H):
        #name0=str('compact_histogram_of_slice_')+str(i)
        #name0=str('full_histogram')+str(i)
        name0=str('low_filter_histogram_of_slice_')+str(i)
        name1=name0+str('.png')
        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(X, Y, H, cmap='gray')#, vmin=np.min(H), vmax=np.max(H))
        #np.save('full_histogram.npy', H.T)
        
        # set limits and ticks
        #ax.set_ylim(-90, 90)
        #ax.set_xlim(-180, 180)
        ax.set_xticks([])
        ax.set_yticks([])

        # remove borders and save
        fig.patch.set_visible(False)
        ax.axis('off')
        #np.save(f'low_filter_histogram_of_slice_{i}.npy', H.T)
        #plt.savefig(name1, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)

import numpy as np

def velogen(usub, vsub, wsub, grid_size, ranglen,sample_indices):
    xyzmag = np.sqrt(usub**2 + vsub**2 + wsub**2)
    uthresh = np.sqrt(np.mean(usub**2))*1.3

    u2 = usub[xyzmag > uthresh]
    v2 = vsub[xyzmag > uthresh]
    w2 = wsub[xyzmag > uthresh]

    div = 1
    cube_size = grid_size // ranglen
    # Using flattened coordinates for indexing
    # Using flattened coordinates for indexing
    coords_flat = np.arange(0, cube_size)
    xvals_flat, yvals_flat, zvals_flat = np.meshgrid(coords_flat, coords_flat, coords_flat)
    xvals_flat = xvals_flat.ravel()[sample_indices]
    yvals_flat = yvals_flat.ravel()[sample_indices]
    zvals_flat = zvals_flat.ravel()[sample_indices]
    # Using the mask to index the flattened coordinates
    x1 = xvals_flat[xyzmag.ravel() > uthresh]
    y1 = yvals_flat[xyzmag.ravel() > uthresh]
    z1 = zvals_flat[xyzmag.ravel() > uthresh]

    


    # computing the angles
    azi = np.arctan2(v2, u2)
    xymag = np.sqrt(u2**2 + v2**2)
    phi = np.arctan(w2 / xymag)
    azi2 = azi * 180 / np.pi
    phi2 = phi * 180 / np.pi

    # Assuming the function compute_transformed_angles exists and works as intended
    (azi_N, phi_N), (azi_Z, phi_Z) = compute_transformed_angles(u2, v2, w2)
    azi3_shifted = (azi2 + 360) % 360
    velocity = np.sqrt(u2**2 + v2**2 + w2**2)

    return u2, v2, w2, x1, y1, z1, azi2, phi2, azi_N, phi_N, azi_Z, phi_Z, azi3_shifted, velocity
#creating the histogram
def velogen2(usub, vsub, wsub, grid_size, ranglen):
    #xyzmag = np.sqrt(usub**2 + vsub**2 + wsub**2)
    #uthresh = np.sqrt(np.mean(usub**2))*1.3

    #u2 = usub[xyzmag > uthresh]
    #v2 = vsub[xyzmag > uthresh]
    #w2 = wsub[xyzmag > uthresh]
    u2=usub
    v2=vsub
    w2=wsub

    # computing the angles
    azi = np.arctan2(v2, u2)
    xymag = np.sqrt(u2**2 + v2**2)
    phi = np.arctan(w2 / xymag)
    azi2 = azi * 180 / np.pi
    phi2 = phi * 180 / np.pi

    # Assuming the function compute_transformed_angles exists and works as intended
    (azi_N, phi_N), (azi_Z, phi_Z) = compute_transformed_angles(u2, v2, w2)
    azi3_shifted = (azi2 + 360) % 360
    velocity = np.sqrt(u2**2 + v2**2 + w2**2)

    return u2, v2, w2, azi2, phi2, azi_N, phi_N, azi_Z, phi_Z, azi3_shifted, velocity
#creating the histogram
def histogen(azi2,phi2,azi_N, phi_N,azi_Z, phi_Z,azi3_shifted,OO):
        
    weights=1/(np.cos(phi2/(180/pi)))
    if OO==0:
        H,xedges,yedges=np.histogram2d(azi2,phi2,weights=weights,bins=[36,18])
    elif OO==1:
        

        # update the histogram creation
        H, xedges, yedges = np.histogram2d(azi3_shifted, phi2, weights=weights, bins=[36,18])
    elif OO==2:
        #weights=1/(np.cos(phi_N/(180/pi)))
        #print(len(azi_N),len( phi_N),len(weights))
        H, xedges, yedges = np.histogram2d(azi_N, phi_N, weights=weights, bins=[36,18])
    elif OO==3:
        #weights=1/(np.cos(phi_Z/(180/pi)))
        H, xedges, yedges = np.histogram2d(azi_Z, phi_Z, weights=weights, bins=[36,18])

        
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    xcenter = (xedges[1:] + xedges[:-1])/2.
    ycenter = (yedges[1:] + yedges[:-1])/2.
    xcenter = xcenter*np.pi/180
    ycenter = ycenter*np.pi/180

    H = H / (np.sum(H)*dx*np.pi/180.*dy*np.pi/180.) 
    #H = H.T
    H_flat=H.flatten()
    
    
    X, Y = np.meshgrid(xcenter, ycenter)
    return H,xedges,yedges,X,Y,
def bingen(sorted_contours,xedges,yedges):
    ii = 0
    independent_bins=[]
    independent_contours=[]
    independent_contours2=[]
    binlst=[]
    normal_areas=[]
    for contour_idx, contour in enumerate(sorted_contours):
        if len(contour) > 0:
            for xo in range(len(contour)):
                contour0 = contour[xo]
                contour0_cv2_format = contour0.reshape((-1, 1, 2)).astype(np.int32)  # Convert to cv2 format
                                
                is_current_contour_touching_boundary=is_contour_touching_boundary(contour0)

                if not is_current_contour_touching_boundary:
                    #is_current_contour_inside_any_previous = any(cv2.pointPolygonTest(previous_contour, tuple(point), False) >= 0
                    #                                        for previous_contour in independent_contours
                    #                                        for point in contour0)
 
                    #if not is_current_contour_inside_any_previous:
                        independent_contours.append(contour0_cv2_format)
                    
                        bin_indices = find_bins_inside_contour(contour0, xedges, yedges)
                        if len(bin_indices)>0:
                            normal_area= calculate_contour_area(contour[xo])
                            normal_areas.append(normal_area)
                            independent_bins.append(bin_indices)
                            binlst.append(bin_indices)
                            independent_contours2.append(contour0)
                            ii += 1
    return binlst,independent_contours2

def contourgen(H,model):
    loaded_arrayx=H
    loaded_array = loaded_arrayx.reshape(1, 1, 36, 18)  # Add batch and channel dimensions
    input_image = torch.from_numpy(loaded_array).float()
    output = model(input_image)
    output_numpy = output.detach().numpy()

    contoursxx = plt.contour(output_numpy[0, 0, :, :]).allsegs  # Extract contour coordinates
    filtered_contoursxx = [contour for contour in contoursxx if len(contour) > 0]
    flattened_contoursxx = [contour for sublist in filtered_contoursxx for contour in sublist]
    sorted_contoursxx = sorted(filtered_contoursxx, key=calculate_contour_area0, reverse=True)
    loaded_arrayx_rescaled =np.flipud(loaded_arrayx.T) * (360 / 35) - 180
    sorted_contours = sorted(sorted_contoursxx, key=len, reverse=True)
    return sorted_contours,loaded_arrayx_rescaled

def pointgen(point_indices,u,x1,y1,z1,ranglen,i,j,k,ii):
    #isomatrix[point_indices] = ii+1
    #isomatrix2=x1*0                                 

        #isomatrix2[point_indices] = 1
    #isomatrix3=isomatrix2[point_indices]
    x2=x1[point_indices]
    y2=y1[point_indices]
    z2=z1[point_indices]
    mdict = {'x': x2, 'y': y2, 'z': z2}#, 'Q': isomatrix3}
    step=int((len(u)))
    grid = np.zeros((step,step,step))
    points = np.vstack((x2, y2, z2)).T
    for point in points:
        x, y, z = point
        # Assuming x, y, z are integer indices that fit within the grid dimensions
        grid[int(x), int(y), int(z)] = 1
    mdict = {'grid': grid}
    savemat(f'High_Re_{ranglen}_Slice_{i}_{j}_{k}_Contour_{ii}.mat', mdict)

def statgen(point_indices,usub,vsub,wsub,u2,v2,w2,trueind):
    density=len(trueind)/usub.size
    total_kinetic_energy =np.sum( 0.5 * (usub**2 + vsub**2 + wsub**2))
    kinetic_cluster_energy=np.sum(0.5 * (u2[point_indices]**2 + v2[point_indices]**2 + w2[point_indices]**2))
    density2=len(trueind)/u2.size
    return density, total_kinetic_energy,kinetic_cluster_energy,density2


def read_mat_file(filename):
    data = scipy.io.loadmat(filename)
    # Extract the array from the dictionary returned by loadmat
    for key in data:
        if "__" not in key and "readme" not in key.lower():
            return data[key]

def compactplot(i,X,Y,H):
        #name0=str('compact_histogram_of_slice_')+str(i)
        #name0=str('full_histogram')+str(i)
        name0=str('high_Re_low_filter_histogram_of_slice_')+str(i)
        name1=name0+str('.png')
        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(X, Y, H, cmap='gray')#, vmin=np.min(H), vmax=np.max(H))
        #np.save('full_histogram.npy', H.T)
        np.save(f'high_Re_low_filter_histogram_of_slice_{i}.npy', H.T)
        # set limits and ticks
        #ax.set_ylim(-90, 90)
        #ax.set_xlim(-180, 180)
        ax.set_xticks([])
        ax.set_yticks([])

        # remove borders and save
        fig.patch.set_visible(False)
        ax.axis('off')
        plt.savefig(name1, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)


def is_contour_touching_boundary(contour):
    
        #sub_contour = np.array(sub_contour)
        
        contour_x = contour[:, 1] * (360 / 35) - 180  # assuming the x-values are in the range of -180 to 180
        contour_y = (17 - contour[:, 0]) * (180 / 17) - 90
        #boundary_points = np.sum((contour_x <= (low_boundary + boundary_range)) | 
        #                         (contour_x >= (high_boundary - boundary_range)))
        xox=0
        boundary_points = np.sum((contour_x <= -179.999-xox) | 
                                 (contour_x >= 179.999 +xox)|
                                 (contour_x >= 89.999 +xox)|
                                (contour_y <= -89.999 -xox) )

        if boundary_points >= 1:
            #print('touch',contour_x)
            return True
        return False

def is_shifted_contour_touching_boundary(contour):        
        contour_x = contour[:, 1] * (360 / 35) #shift_value # assuming the x-values are in the range of -180 to 180
        xox=0
        boundary_points = np.sum((contour_x <= 0.2-xox) | 
                                 (contour_x >= 359.8+xox ))
        if boundary_points >= 1:
            #print('touch_shift',contour_x)
            return True
        return False



def find_shifted_bins_inside_contour(contour, H, x_edges, y_edges):
    # Adjust the transformation for the shifted azimuth
    contour_x = contour[:, 1] * (360 / 35)
    contour_y = (17 - contour[:, 0]) * (180 / 17) - 90
    
    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    y_centers = (y_edges[1:] + y_edges[:-1]) / 2
    
    X, Y = np.meshgrid(x_centers, y_centers)
    bin_centers = np.column_stack((X.ravel(), Y.ravel()))
    
    mask = points_in_poly(bin_centers, np.column_stack((contour_x, contour_y)))
    
    bin_indices = np.where(mask)[0]
    
    return bin_indices

def find_bins_inside_contour(contour, x_edges, y_edges):
    contour_x = contour[:, 1] * (360 / 35) - 180
    contour_y = (17 - contour[:, 0]) * (180 / 17) - 90
    
    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    y_centers = (y_edges[1:] + y_edges[:-1]) / 2
    
    X, Y = np.meshgrid(x_centers, y_centers)
    bin_centers = np.column_stack((X.ravel(), Y.ravel()))
    
    mask = points_in_poly(bin_centers, np.column_stack((contour_x, contour_y)))
    
    bin_indices = np .where(mask)[0]
    
    return bin_indices

def find_shifted_points_in_bins(azi2, phi2, xedges, yedges, bin_indices):
    # Shift the azimuth angles from (-180, 180) to (0, 360)
    azi2_shifted = (azi2 + 360) % 360

    # Get bin indices for a ll points
    bin_x = np.clip(np.digitize(np.round(azi2_shifted, decimals=-1), xedges) - 1, 0, len(xedges)-2)
    bin_y = np.clip(np.digitize(np.round(phi2, decimals=-1), yedges) - 1, 0, len(yedges)-2)

    # Convert 2D bin indices to 1D (flatten the bins)
    bin_x_y = (len(yedges)-2-bin_y) * (len(xedges) -1)+ bin_x

    # Get the indices of points inside the bins
    point_indices = np.isin(bin_x_y, bin_indices)

    return point_indices





def find_points_in_bins(azi2, phi2, xedges, yedges, bin_indices):
    # Get bin indices for all points
    bin_x = np.clip(np.digitize(np.round(azi2 ,decimals=-1), xedges) - 1, 0, len(xedges)-2)
    bin_y = np.clip(np.digitize(np.round(phi2, decimals=-1), yedges) - 1, 0, len(yedges)-2)


    # Convert 2D bin indices to 1D (flatten the bins)
    bin_x_y = (len(yedges)-2-bin_y) * (len(xedges) -1)+ bin_x

    # Get the indices of points inside the bins
    point_indices = np.isin(bin_x_y, bin_indices)

    return point_indices
def getdata():
    mat1=scipy.io.loadmat('DNS_ucomp.mat')
    mat2=scipy.io.loadmat('DNS_vcomp.mat')
    mat3=scipy.io.loadmat('DNS_wcomp.mat')
    
    u=mat1['u']
    v=mat2['v']
    w=mat3['w']
    
    return u,v,w
def load_filtered_contours(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)
def calculate_contour_area(contour):
        # Try to convert the contour to an array and compute the area
        try:
            return abs(cv2.contourArea(np.array(contour[0], dtype=np.float32).reshape((-1, 1, 2))))
        except ValueError:
            print('error',contour[0])
            return 0

def contour_area_diff_within_threshold(contour1, contour2, threshold):
    area1 = cv2.contourArea(contour1)
    area2 = cv2.contourArea(contour2)
    if area1 == 0 and area2 == 0:  
        return True
    elif area1 == 0 or area2 == 0: 
        return False
    else:
        return abs(area1 - area2) / ((area1 + area2) / 2) < threshold

def generate_colors(n):
    # Use a golden ratio to try to spread out the colors as much as possible
    golden_ratio_conjugate = 0.618033988749895
    h = 0.5 # start with an arbitrary hue
    
    colors = []
    for _ in range(n):
        h += golden_ratio_conjugate
        h %= 1  # ensure hue is in [0, 1)
        
        r, g, b = colorsys.hsv_to_rgb(h, 0.99, 0.99)  # high saturation and value to get bright colors
        
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
        
    return colors
def calculate_contour_area0(contour):
        # Try to convert the contour to an array and compute the area
        try:
            return abs(cv2.contourArea(np.array(contour[0], dtype=np.float32).reshape((-1, 1, 2))))
        except ValueError:
            print('error',contour[0])
            return 0

def calculate_contour_average2(contour):
        flattened_contour = np.array([item for sublist in contour for item in sublist])
        return np.average(flattened_contour, axis=0)
def calculate_contour_area2(contour):
        #areas = []
        #for contour in contours:
            # Now 'contour' is a 2D numpy array and you can apply your area calculation
            contour=contour[0]
            x = contour[:, 1]
            y = contour[:, 0]
            area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
            #areas.append(area)
            return area
def calculate_contour_area(contour):
        x = contour[:, 1]
        y = contour[:, 0]
        return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
    
def calculate_contour_average(contour):
        flattened_contour = np.array(contour).reshape(-1, 2)
        return np.average(flattened_contour, axis=0)

def calculate_overlap_percentage(indices1, indices2):
    indices1_set = set(indices1)
    indices2_set = set(indices2)
    overlap = indices1_set & indices2_set  # This gets the intersection of the two sets
    percentage_overlap = len(overlap) / len(indices1_set)  # Calculates the percentage overlap
    return percentage_overlap
def check_encapsulation_bin_indices(indices1, indices2):
    """
    Function to check if all bin indices of indices2 are within indices1
    :param indices1: A list of bin indices describing the first contour.
    :param indices2: A list of bin indices describing the second contour.
    :return: True if all indices of indices2 are within indices1, otherwise False.
    """
    indices1_set = set(indices1)
    indices2_set = set(indices2)
    
    return indices2_set.issubset(indices1_set)
