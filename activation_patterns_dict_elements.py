# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 23:00:59 2015

@author: asangari
"""
from hexagonal_sampling import Retina_model, Retina_sample
from dic_learning_rotUpdate import sec_rot_invar, code_rot_invar_dic

from time import time
import pylab as plt
import numpy as np
import cv2

def vis_filters(D, patch_size):
    m, k = D.shape
    V = D.T
    plt.figure(figsize=(8.4, 8))
    for i, comp in enumerate(V[:k]):
        plt.subplot(20,24, i + 1)
        plt.imshow(comp.reshape(patch_size), interpolation='nearest',cmap='Greys') #  vmin=vmin, vmax=vmax, )
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23) 

    
from scipy.interpolate import griddata
def vis_sectional_filters(D, sp_coord, n_sections, patch_size, resampling = 'linear'):
    
    m, k = D.shape
    cortical_sizes = sp_coord.shape
    grid_y, grid_x = np.mgrid[0:float(patch_size[0]), 0:float(patch_size[1])]
    
    grid_y -= float(patch_size[0]-1)/2
    grid_x -= float(patch_size[1]-1) /2
    V = D.T
    plt.figure()
    for i, comp in enumerate(V[:k]):
        plt.subplot(5,16, i + 1)
        resampled_el = griddata(sp_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2), 
                           comp, (grid_y, grid_x), method=resampling, fill_value=0)    
                           
        plt.imshow(resampled_el.reshape(patch_size), interpolation='nearest', cmap='Greys')#, vmin=vmin, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)   
    
## DATASET GENERATION
if __name__ == '__main__':
    n_samples = 500
    win_r = 4
    
    hexagonal_sp_coord, R_i = Retina_model(M_p = 0, M_f = win_r, n_t = 6)
    retina_size = hexagonal_sp_coord.shape[0:2]
    
    
    # Mask for square sampling
    patch_size = (2*win_r+1, 2*win_r+1)
   
    X_hex = []
   
    image_name = './image_samples/barbara.bmp'    
    orig = cv2.imread(image_name)
    h, w, ch = orig.shape
    img1 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    barbara_hex = []
    for s in range(n_samples):
        x_c = np.random.randint(win_r, w-win_r-1)
        y_c = np.random.randint(win_r, h-win_r-1)
        fovea_xy = [y_c, x_c]
        
        retina_sample = Retina_sample(img1, fovea_xy, hexagonal_sp_coord)
        X_hex.append(retina_sample.flatten())
        barbara_hex.append(retina_sample.flatten())
        
    
        
    image_name = './image_samples/lena.bmp'
    orig = cv2.imread(image_name)
    h, w, ch = orig.shape
    img2 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    lena_hex = []
    for s in range(n_samples):
        x_c = np.random.randint(win_r, w-win_r-1)
        y_c = np.random.randint(win_r, h-win_r-1)
        fovea_xy = [y_c, x_c]
        
        retina_sample = Retina_sample(img2, fovea_xy, hexagonal_sp_coord)
        X_hex.append(retina_sample.flatten())
        lena_hex.append(retina_sample.flatten())
        

    image_name = './image_samples/sower.png'
    orig = cv2.imread(image_name)
    h, w, ch = orig.shape
    img3 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) 
    boat_hex = []
    for s in range(n_samples):
        x_c = np.random.randint(win_r, w-win_r-1)
        y_c = np.random.randint(win_r, h-win_r-1)
        fovea_xy = [y_c, x_c]
        
        retina_sample = Retina_sample(img3, fovea_xy, hexagonal_sp_coord)
        X_hex.append(retina_sample.flatten())
        boat_hex.append(retina_sample.flatten())
        
       
    X_hex = np.vstack(X_hex)
   
    
    ## DICTIONARY LEARNING
    
    #Same initialization sampled from patches
    n_filter = 80
    n_epochs = 6
    sparsity = 4
    EV_SCORE = True #True
    sample_ids = np.random.randint(low = 0, high = n_samples, size = n_filter)
  

    data = X_hex - np.expand_dims(np.mean(X_hex, axis=1), 1)
    data /= np.expand_dims(np.linalg.norm(data, axis=1), axis=1)
    D_0 = data[sample_ids, :].T
    t0 = time()
    D2, losses2 =  sec_rot_invar(data, n_filters = n_filter, n_theta = 6, eta = 1e-3, # D_0 = D_0,
                                 n_sections = 6, sparsity = sparsity, n_epochs = n_epochs, EV_SCORE = EV_SCORE)
    dt = time() - t0
    print 'time needed for Hexagonal Sampling: ' + str(dt)


    vis_sectional_filters(D2[:,::6], hexagonal_sp_coord, 6, patch_size)
    

    activation_barbara = code_rot_invar_dic(np.vstack(barbara_hex).T, D2, 6, sparsity)
    activation_boat = code_rot_invar_dic(np.vstack(boat_hex).T, D2, 6, sparsity)
    activation_lena = code_rot_invar_dic(np.vstack(lena_hex).T, D2, 6, sparsity)
    highest_freq = max(activation_barbara.max(), activation_boat.max(), activation_lena.max())
    
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(img1, cmap=plt.cm.Greys_r)
    plt.title('Barbara')
    plt.axis('off')
    plt.subplot(2,3,2)
    plt.imshow(img2, cmap=plt.cm.Greys_r)
    plt.title('Lena')
    plt.axis('off')
    plt.subplot(2,3,3)    
    plt.imshow(img3, cmap=plt.cm.Greys_r)
    plt.title('Sower')
    plt.axis('off')
    plt.subplot(2,3,4)
    plt.imshow(activation_barbara*1./highest_freq, interpolation='nearest', cmap='Greys', aspect='auto')
    plt.ylabel('Dictionary Element Index')
    plt.xlabel('Orientation')
    
    plt.subplot(2,3,5)
    plt.imshow(activation_boat*1./highest_freq, interpolation='nearest', cmap='Greys', aspect='auto')
    plt.xlabel('Orientation')
    plt.subplot(2,3,6)

    plt.imshow(activation_lena*1./highest_freq, interpolation='nearest', cmap='Greys', aspect='auto')
    plt.xlabel('Orientation')
    plt.show()
