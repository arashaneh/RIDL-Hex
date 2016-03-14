# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 23:00:59 2015

@author: asangari
"""
from hexagonal_sampling import Retina_model, Retina_sample
from radial_sampling import Radial_model, Radan_sample


from dic_learning_rotUpdate import sq_rot_invar, sq_dict_learning, sec_rot_invar 

from time import time

from PIL import Image, ImageDraw
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
    n_samples = 2000
    win_r = 4
    image_name = './image_samples/lena.bmp'
    
    hexagonal_sp_coord, R_i = Retina_model(M_p = 0, M_f = win_r, n_t = 6)
    retina_size = hexagonal_sp_coord.shape[0:2]
    
    radial_sp_coord1, R_i = Radial_model(M_f = win_r, n_t = 12)
    radial_sp_coord2, R_i = Radial_model(M_f = win_r, n_t = 18)
    
    # Mask for square sampling
    patch_size = (2*win_r+1, 2*win_r+1)
    mask = Image.new('L', patch_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + (2*win_r+1, 2*win_r+1), fill=255)
    mask = np.array(mask.getdata()).reshape(patch_size,order='F')/255
    
    effective_dim = mask.sum()
    dummy_dim = mask.shape[0]*mask.shape[1]
    dim_ratio = float(dummy_dim)/effective_dim

    X_hex = []
    X_sq = []
    X_rad1 = []
    X_rad2 = []
    
    
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
        
        square_sample = img1[y_c-win_r:y_c+win_r+1, x_c-win_r:x_c+win_r+1]
        X_sq.append((mask*square_sample).flatten())
        
        radan_sample = Radan_sample(img1, fovea_xy, radial_sp_coord1)
        X_rad1.append(radan_sample.flatten())
        
        radan_sample = Radan_sample(img1, fovea_xy, radial_sp_coord2)
        X_rad2.append(radan_sample.flatten())
        
        
    X_hex = np.vstack(X_hex)
    X_sq = np.vstack(X_sq)
    X_rad1 = np.vstack(X_rad1)
    X_rad2 = np.vstack(X_rad2)
    
    ## DICTIONARY LEARNING
    
    #Same initialization sampled from patches
    n_filter = 80
    n_epochs = 6
    sparsity = 2
    EV_SCORE = True
    sample_ids = np.random.randint(low = 0, high = n_samples, size = n_filter)
  
    
    data = X_rad2 - np.expand_dims(np.mean(X_rad2, axis=1), 1)
    data /= np.expand_dims(np.linalg.norm(data, axis=1), axis=1)
    D_0 = data[sample_ids, :].T
    t0 = time()
    D1, losses1 =  sec_rot_invar(data, n_filters = n_filter, n_theta = 6, eta = 1e-3,# D_0 = D_0,
                                 n_sections = 6, sparsity = sparsity, n_epochs = n_epochs, EV_SCORE = EV_SCORE)
    dt = time() - t0
    print 'time needed for Radial Sampling: ' + str(dt)

       
    
    data = X_hex - np.expand_dims(np.mean(X_hex, axis=1), 1)
    data /= np.expand_dims(np.linalg.norm(data, axis=1), axis=1)
    D_0 = data[sample_ids, :].T
    t0 = time()
    D2, losses2 =  sec_rot_invar(data, n_filters = n_filter, n_theta = 6,  eta = 1e-3, # D_0 = D_0,
                                 n_sections = 6, sparsity = sparsity, n_epochs = n_epochs, EV_SCORE = EV_SCORE)
    dt = time() - t0
    print 'time needed for Hexagonal Sampling: ' + str(dt)
    
    ## Visualize the learned dictionary elements
    vis_sectional_filters(D2[:,::6], hexagonal_sp_coord, 6, patch_size)
                     
  
   
    
    D_0 = data[sample_ids, :].T
    t0 = time()
    D4, losses4 = sq_dict_learning(data, mask,  n_filters = n_filter,  eta = 1e-3, #D_0 = D_0,
                                  sparsity = sparsity, n_epochs = n_epochs, EV_SCORE = EV_SCORE)
    dt = time() - t0
    print 'time needed for Square Sampling (old): ' + str(dt)      

    
    
    D_0 = data[sample_ids, :].T
    t0 = time()
    D5, losses5 = sq_dict_learning(data, mask,  n_filters = n_filter*6,  eta = 1e-3, #D_0 = D_0,
                                  sparsity = sparsity, n_epochs = n_epochs, EV_SCORE = EV_SCORE)
    dt = time() - t0
    print 'time needed for Square Sampling (old): ' + str(dt)      

  
    losses1 = np.array(losses1)
    losses2 = np.array(losses2)
    losses4 = np.array(losses4)
    losses5 = np.array(losses5)
    


    plt.figure()
    plt.hold(True)
    iterations = np.arange(losses1.shape[0])*500
    plt.plot(iterations,losses1,'-*k',linewidth=1, markersize=10)
    plt.plot(iterations,losses2,'-k',linewidth=1)
    plt.plot(iterations,losses4*mask.sum()/mask.shape[0]/mask.shape[1],'-.k',linewidth=3)
    plt.plot(iterations,losses5*mask.sum()/mask.shape[0]/mask.shape[1],'.-k',linewidth=1,markersize=8)
    plt.legend(['Polar Grid (Rot_Invar, 80)','Hexagonal Grid (Rot_Invar, 80)', \
    'Square Grid (NRot_Invar, 80)', 'Square (NRot_Invar, 480)'],fontsize=14)
    plt.xlabel('Iteration',fontsize=18)
    plt.ylabel('MSE',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Rotation Matrix Update')
    

    plt.show()