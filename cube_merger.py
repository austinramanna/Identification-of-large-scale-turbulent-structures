import numpy as np
from scipy.io import loadmat, savemat

def is_similar(array1, array2, threshold=0.6):
    array1_int = array1.astype(int)
    array2_int = array2.astype(int)
    
    shared_points = np.sum(array1_int & array2_int)  # Count of common '1' points
    
    # Calculate points in the smaller array
    smaller_array_points = min(np.sum(array1_int), np.sum(array2_int))
    
    # Check if shared points are at least 65% of the smaller array's points
    return shared_points / smaller_array_points >= threshold

def merge_arrays(array1, array2):
    return np.logical_or(array1, array2).astype(int)

unique_arrays = []

for i in range(3):
    for j in range(3):
        for k in range(3):
            for ii in range(10):  # Assuming maximum 100 files for ii, adjust as needed
                try:
                    filename = f'High_Re_2_Slice_{i}_{j}_{k}_Contour_{ii}.mat'
                    #filename = f'High_Re_10_slice_Complete_Points_Slice_{i}__{j}_{k}_Contour_{ii}.mat'
                    data = loadmat(filename)
                    grid = data['grid']
                    
                    merged = False
                    for index, unique in enumerate(unique_arrays):
                        if is_similar(grid, unique):
                            unique_arrays[index] = merge_arrays(grid, unique)
                            merged = True
                            break
                            
                    if not merged:
                        unique_arrays.append(grid)
                        
                except FileNotFoundError:
                    if ii > 0:  # Move on if we've passed the 0th file and encountered an error
                        break

merged_indices = set()  # This set will keep track of which indices have been merged
for i in range(len(unique_arrays)):
    if i in merged_indices:
        continue
    for j in range(i+1, len(unique_arrays)):
        if j in merged_indices:
            continue
        if is_similar(unique_arrays[i], unique_arrays[j]):
            unique_arrays[i] = merge_arrays(unique_arrays[i], unique_arrays[j])
            merged_indices.add(j)  # Mark the second array as merged

# Filter out the merged arrays
final_arrays = [unique_arrays[i] for i in range(len(unique_arrays)) if i not in merged_indices]

# Save the final arrays
for index, grid in enumerate(final_arrays):
    mdict = {'grid': grid}
    savemat(f'Merged_Grid07_{index}.mat', mdict)
