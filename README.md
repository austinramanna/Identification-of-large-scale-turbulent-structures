# Identification-of-large-scale-turbulent-structures



This code allows the user to acquire large-scale structures in turbulence flows directly from a velocity field using U-net. A flow diagram of the complete program is presented below. To train the U-net from scratch run the gen_histogram.py and Unet_training.py files back to back. Alternatively, there are several uploaded pre-trained weight files from which best_model_U_real_weights_055.pth turned out superior. 

To acquire the structures, update the code in Pole_test2.py slightly to link your velocity field, then run this file as well as cube_merger.py to get the structures as a set of point. Execute minkowskimergerd.m to acquire the structures as continuous isosurfaces. To get statistics (structure kinetic energy, volume fraction, minkowski length scales and PCA length scales), run all_post_matlab_plots.py. Make sure to update run all_post_matlab_plots.py for the velocity field as well as the integral length scale of the velocity field. run_code.py allows the user to run all the files back to back automatically, however, make sure to insert the right velocity field as well as integral length scale in the individual code files beforehand.

![Detection flow diagram](detection_flowdiagram.png)

