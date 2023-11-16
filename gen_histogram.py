import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import h5py
from tqdm import trange

def add_peak(scaling=0.2,scaling_shift=0.2):
	ns = 10000
	azi_vals = np.arange(36)*2.*np.pi/36
	phi_vals = np.arange(18)*np.pi/18 - np.pi/2.

	dazi = 2.*np.pi/36.
	dphi = np.pi / 18.
	a = np.random.rand()*2.
	b = 2.-a
	L = np.random.rand()*scaling+scaling_shift
	cov = sp.stats.random_correlation.rvs((a, b))*L

	m_azi = random.sample(list(azi_vals),1)
	m_phi = random.sample(list(phi_vals),1)
	mean = np.array([m_azi[0], m_phi[0]])

	dis = sp.stats.multivariate_normal(mean,cov)

	x, y = np.mgrid[-np.pi:3.*np.pi:np.pi/180., -np.pi:np.pi:np.pi/180.]
	pos = np.dstack((x, y))
	val = dis.pdf(pos)
	nx = int( (x.shape[0]-1)/2) 
	nx_half = int(nx/2)
	ny = int( (x.shape[1]-1)/2)
	ny_half = int(ny/2)

	## enforce cyclic periodicity for the given peak
	val2 = val[nx_half:nx_half+nx,ny_half:ny_half+ny].copy()
	val2[nx_half:nx_half+nx_half,:] += val[:nx_half,ny_half:ny_half+ny]
	val2[:nx_half,:] += val[-nx_half:,ny_half:ny_half+ny]
	val2[:,ny_half:ny_half+ny_half] += val[nx_half:nx_half+nx:,:ny_half]
	val2[:,:ny_half] += val[nx_half:nx_half+nx:,-ny_half:]
	val2[-nx_half:,-ny_half:] += val[:nx_half,:ny_half]
	val2[-nx_half:,:ny_half] += val[:nx_half,-ny_half:]
	val2[:nx_half,-ny_half:] += val[-nx_half:,:ny_half]
	val2[:nx_half,:ny_half] += val[-nx_half:,-ny_half:]
	x2 = x[nx_half:nx_half+nx,ny_half:ny_half+ny]
	y2 = y[nx_half:nx_half+nx,ny_half:ny_half+ny]

	# get the associated label
	#the 0.6 determines the label radius
	label = (val2>0.6*np.max(val2))*1.0

	return x2,y2,val2,label

def add_noise(val,noise_ratio):
	noise = np.random.rand(val.shape[0],val.shape[1])*noise_ratio*np.max(val)
	val_noise = val + noise
	return val_noise

def generate_artif_histograms(nb_images=1,max_peaks=5,scaling=0.1,scaling_shift=0.2):
	all_label = []
	all_val = []
	for iim in trange(nb_images): #tqdm(range(nb_images)):
		scaling = scaling
		scaling_shift = scaling_shift
		nb_peaks = np.random.randint(1,max_peaks)
		x,y,val_all,label_all = add_peak(scaling,scaling_shift)
		dx = x[0,0]-x[1,0]
		dy = y[0,0] - y[0,1]
		for i in range(1,nb_peaks):
			x,y,val,label = add_peak(scaling,scaling_shift)
			peak_scale = np.random.rand()+0.1
			val_all += val*peak_scale
			label_all += label

		# coarsen it to the (36,18) format
		skip = 10
		nxrange = int(label_all.shape[0]/skip+1)
		nyrange = int(label_all.shape[1]/skip+1)
		label_coarse = np.zeros((nxrange,nyrange))
		val_coarse = np.zeros((nxrange,nyrange))
		for i in range(nxrange):
			for j in range(nyrange):
				val_coarse[i,j] = np.sum(val_all[skip*i:skip*(i+1),skip*j:skip*(j+1)])
				label_coarse[i,j] = np.sum(label_all[skip*i:skip*(i+1),skip*j:skip*(j+1)])

		# re-normalization
		val_coarse = val_coarse / (np.sum(val_coarse) * dx*dy*skip*skip)
		label_coarse[label_coarse>1] = 1.		

		# add noise and renormalization
		noise_ratio = 0.1
		val_coarse = add_noise(val_coarse,noise_ratio)
		val_coarse = val_coarse / (np.sum(val_coarse) * dx*dy*skip*skip)

		all_label.append(label_coarse)
		all_val.append(val_coarse)

	all_label = np.array(all_label)
	all_val = np.array(all_val)

	return x[::skip,::skip], y[::skip,::skip], all_label, all_val



x, y, label_all, val_all = generate_artif_histograms(10000,5)

hf = h5py.File('data_artif_0.45.h5','w')
hf.create_dataset('jpdf',data=val_all)
hf.create_dataset('label',data=label_all)
hf.close()


# label = label_all[0,:,:]
# val = val_all[0,:,:]
# skip = 10
# nxrange = int(label.shape[0]/skip+1)
# nyrange = int(label.shape[1]/skip+1)
# label_coarse = np.zeros((nxrange,nyrange))
# val_coarse = np.zeros((nxrange,nyrange))
# for i in range(nxrange):
# 	for j in range(nyrange):
# 		val_coarse[i,j] = np.sum(val[skip*i:skip*(i+1),skip*j:skip*(j+1)])
# 		label_coarse[i,j] = np.sum(label[skip*i:skip*(i+1),skip*j:skip*(j+1)])

# print(val_all.shape)
# print(label_all.shape)
idx = 0
plt.figure()
plt.contourf(x,y,val_all[idx,:,:])
plt.colorbar()
plt.figure()
plt.contourf(x,y,label_all[idx,:,:])
plt.colorbar()
plt.show()


#  





