import numpy as np
from numpy import array

def distance(point1, point2):
    """Compute the distance between two points."""
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def compare_and_filter_blobs(x_coords, y_coords, threshold=2):
    n = len(x_coords)
    
    # Create a list to hold lists of points (x,y) for each blob
    blobs = [list(zip(x_coords[i], y_coords[i])) for i in range(n)]
    
    to_remove = set()  # indices of blobs to be removed
    
    for i in range(n):
        for j in range(i+1, n):
            if j in to_remove:  # Skip blobs already marked for removal
                continue
            
            # Check how many points of blob j are close to blob i
            close_points = sum(1 for point_j in blobs[j] for point_i in blobs[i] if distance(point_j, point_i) < threshold)
            
            common_ratio_i = close_points / len(blobs[i])
            common_ratio_j = close_points / len(blobs[j])
            thresh=0.5
            if common_ratio_i >= thresh or common_ratio_j >= thresh:
                # Add the smaller blob's index to the removal set
                if len(blobs[i]) < len(blobs[j]):
                    to_remove.add(i)
                else:
                    to_remove.add(j)
    
    # Compute the list of remaining indices
    remaining_indices = [i for i in range(n) if i not in to_remove]
    
    # Filter out the blobs to be removed
    filtered_x_coords = [x_coords[i] for i in remaining_indices]
    filtered_y_coords = [y_coords[i] for i in remaining_indices]
    
    return filtered_x_coords, filtered_y_coords, remaining_indices
import numpy as np

import numpy as np

def reverse_transformed_angles(phi_north, azi_north):
    phi_north = np.array(phi_north)
    azi_north = np.array(azi_north)
    
    # Handle poles
    at_north_pole = (phi_north == 90)
    at_south_pole = (phi_north == 0)

    # Calculate Cartesian coordinates based on North-Pole centered projection
    x_north = np.sin(np.radians(phi_north)) * np.cos(np.radians(azi_north))
    y_north = np.sin(np.radians(phi_north)) * np.sin(np.radians(azi_north))
    z_north = np.cos(np.radians(phi_north))

    # Inverse the rotation about the y-axis by pi/2 to get back the original coordinates
    x = -z_north
    y = y_north
    z = x_north

    # Convert the Cartesian coordinates back to spherical coordinates
    azi = np.round(np.arctan2(y, x) * (180/np.pi),0)
    xymag = np.sqrt(x**2 + y**2)
    phi = np.round(np.arctan2(z, xymag) * (180/np.pi),0)

    # Adjustments for poles
    azi[at_north_pole] = np.nan
    phi[at_north_pole] = 90
    azi[at_south_pole] = np.nan
    phi[at_south_pole] = -90

    return phi, azi
import numpy as np

def reverse_transformed_angles_south(phi_south, azi_south):
    phi_south = np.array(phi_south)
    azi_south = np.array(azi_south)
    
    # Handle poles
    at_north_pole = (phi_south == 0)
    at_south_pole = (phi_south == 90)

    # Calculate Cartesian coordinates based on South-Pole centered projection
    x_south = np.sin(np.radians(phi_south)) * np.cos(np.radians(azi_south))
    y_south = np.sin(np.radians(phi_south)) * np.sin(np.radians(azi_south))
    z_south = -np.cos(np.radians(phi_south))

    # Inverse the rotation about the y-axis by -pi/2 to get back the original coordinates
    x = z_south
    y = y_south
    z = -x_south

    # Convert the Cartesian coordinates back to spherical coordinates
    azi = np.round(np.arctan2(y, x) * (180/np.pi),0)
    xymag = np.sqrt(x**2 + y**2)
    phi = np.round(np.arctan2(z, xymag) * (180/np.pi),0)

    # Adjustments for poles
    azi[at_north_pole] = np.nan
    phi[at_north_pole] = 90
    azi[at_south_pole] = np.nan
    phi[at_south_pole] = -90

    return phi, azi


import numpy as np

def compare_and_filter_indices(indices_list):
    n = len(indices_list)
    
    # Create a list to hold sets for each array for quick intersection operations
    indices_sets = [set(indices_array) for indices_array in indices_list]
    
    to_remove = set()  # indices of arrays to be removed
    
    for i in range(n):
        for j in range(i+1, n):
            if j in to_remove:  # Skip arrays already marked for removal
                continue
                
            # Compute the intersection of the two arrays
            common_indices = indices_sets[i].intersection(indices_sets[j])
            
            # Check if they share 60% or more of their indices
            len_i = len(indices_sets[i])
            len_j = len(indices_sets[j])
            
            # Guard against division by zero
            common_ratio_i = len(common_indices) / len_i if len_i != 0 else 0
            common_ratio_j = len(common_indices) / len_j if len_j != 0 else 0
            
            thresh = 0.5
            if common_ratio_i >= thresh or common_ratio_j >= thresh:
                # Add the smaller array's index to the removal set
                if len_i < len_j:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
    
    # Compute the list of indices of the remaining arrays
    remaining_indices = [i for i in range(n) if i not in to_remove]
    
    return remaining_indices


#indices_list = [np.array([0, 3, 5]), np.array([9, 5]), np.array([0, 3, 4, 5])]
#filtered_list = compare_and_filter_indices(indices_list)
#print(filtered_list)


'''
# Test
print(reverse_transformed_angles_south(90, 90))  # South pole, expecting (-90, NaN)
print(reverse_transformed_angles_south(0, 0))    # Equatorial, pointing East, expecting (0, 0)
print(reverse_transformed_angles_south(0, 90))   # Equatorial, pointing North, expecting (0, 90)
print(reverse_transformed_angles_south(0, 180))  # Equatorial, pointing West, expecting (0, 180 or -180)
print(reverse_transformed_angles_south(0, -90))  # Equatorial, pointing South, expecting (0, -90)

# Test
print(reverse_transformed_angles(90, 90))  # North pole, expecting (90, NaN)
print(reverse_transformed_angles(0, 0))    # Equatorial, pointing East, expecting (0, 0)
print(reverse_transformed_angles(0, 90))   # Equatorial, pointing North, expecting (0, 90)
print(reverse_transformed_angles(0, 180))  # Equatorial, pointing West, expecting (0, 180 or -180)
print(reverse_transformed_angles(0, -90))  # Equatorial, pointing South, expecting (0, -90)
'''

# Example:
x_coords = [array([-130., -120., -110., -140., -130., -120., -140., -130.]), array([-100.]), array([-160., -150., -170., -160., -150., -180., -170., -160.]), array([-150., -170., -160., -150., -180., -170.]), array([-170., -160., -180., -170.]), array([-160., -150., -140., -170., -160., -150., -180., -170., -160.]), array([-160.]), array([-140., -160., -150., -140., -180., -170., -160., -150.,  180.,
       -180., -170., -160.,  180., -180.]), array([-160., -150., -140., -170., -160., -150., -140., -180., -170.,
       -160., -150., -140.,  180., -180., -170., -160.,  170.,  180.,
       -180., -170.]), array([-120., -110., -130., -120., -110., -100.,  -90., -140., -130.,
       -120., -110., -150., -140., -130.]), array([-50., -40., -60., -50., -40.]), array([-140., -130., -120., -110., -140., -130., -120., -110., -100.,
        -90.,  -60.,  -50.,  -40.,  -30., -140., -130., -120., -110.,
       -100.,  -90.,  -80.,  -70.,  -60.,  -50.,  -40.,  -30., -150.,
       -140., -130., -120., -110., -100.,  -70.,  -60.,  -50., -160.,
       -150., -140., -130., -120.]), array([-130., -120., -130., -120., -110., -100.,  -60.,  -50.,  -40.,
        -30., -140., -130., -120., -110., -100.,  -90.,  -80.,  -70.,
        -60.,  -50.,  -40., -150., -140., -130., -120., -110., -100.,
        -60.,  -50., -160., -150., -140., -130., -120.])]
y_coords = [array([-38., -38., -38., -28., -28., -28., -18., -18.]), array([-77.]), array([-18., -18.,  -8.,  -8.,  -8.,   2.,   2.,   2.]), array([-18.,  -8.,  -8.,  -8.,   2.,   2.]), array([-8., -8.,  2.,  2.]), array([-18., -18., -18.,  -8.,  -8.,  -8.,   2.,   2.,   2.]), array([-8.]), array([-28., -18., -18., -18.,  -8.,  -8.,  -8.,  -8.,  -8.,   2.,   2.,
         2.,   2.,  11.]), array([-28., -28., -28., -18., -18., -18., -18.,  -8.,  -8.,  -8.,  -8.,
        -8.,  -8.,   2.,   2.,   2.,   2.,   2.,  11.,  11.]), array([-18., -18.,  -8.,  -8.,  -8.,  -8.,  -8.,   2.,   2.,   2.,   2.,
        11.,  11.,  11.]), array([-18., -18.,  -8.,  -8.,  -8.]), array([-28., -28., -28., -28., -18., -18., -18., -18., -18., -18., -18.,
       -18., -18., -18.,  -8.,  -8.,  -8.,  -8.,  -8.,  -8.,  -8.,  -8.,
        -8.,  -8.,  -8.,  -8.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,
         2.,   2.,  11.,  11.,  11.,  11.,  11.]), array([-28., -28., -18., -18., -18., -18., -18., -18., -18., -18.,  -8.,
        -8.,  -8.,  -8.,  -8.,  -8.,  -8.,  -8.,  -8.,  -8.,  -8.,   2.,
         2.,   2.,   2.,   2.,   2.,   2.,   2.,  11.,  11.,  11.,  11.,
        11.])]

x_coords=[array([-140., -130., -120. ,-160. ,-150. ,-140. ,-130., -120. ,-170., -160. ,-150. ,-140.
 -130. ,-120.,  180., -180. ,-170., -160. ,-150., -140. ,-130. , 170. , 180. ,-180.
 -170. ,-160., -150., -140. , 170. , 180. ,-180., -170. ,-160.]),array([-140. -130. ,-120. ,-110., -160., -150., -140. ,-130., -120., -110., -170. ,-160.,
 -150., -140., -130., -120. ,-110.  ,-90. ,-180., -170., -160., -150. ,-140. ,-130.,
 -110., -100.,  -90.,  180. ,-180. ,-170. ,-160., -150. , 180. ,-180., -170.]),array([-170. ,-180. , 180. ,-160. ,-170. ,-180. ,-140. ,-150. ,-160. ,-170., -110., -120.,
 -130., -140., -150., -160. ,-120. ,-130., -140., -150.])]

y_coords= [array([-77. ,-77. ,-77. ,-68., -68. ,-68. ,-68., -68., -58., -58., -58., -58. ,-58., -58.,
 -58., -48. ,-48. ,-48., -48. ,-48., -48., -48., -48. ,-38., -38., -38., -38., -38.,
 -38., -38., -28., -28., -28.]),array([ -65.,  -65. , -65.  ,-65. , -75. , -75. , -75.  ,-75.  ,-75. , -75. , -84. , -84.,
  -84.,  -84.  ,-84.,  -84.  ,-84.,  -84. , -93.  ,-93. , -93. , -93. , -93. , -93.,
  -93. , -93. , -93. , -93., -102., -102., -102., -102., -102. ,-112. ,-112.]),array([-50., -50. ,-59. ,-59. ,-59. ,-59. ,-68., -68., -68. ,-68. ,-78. ,-78. ,-78., -78.,
 -78., -78., -87., -87., -87., -87.])]




#filtered_x, filtered_y, remaining_indices = compare_and_filter_blobs(x_coords, y_coords)
#print(filtered_x)
#print(filtered_y)
#print(remaining_indices)
