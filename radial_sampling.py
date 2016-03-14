# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:20:10 2015

@author: Arash
"""

import numpy as np
from scipy import ndimage


def Radial_model(M_f = 10, n_t = 6):
    r_i = []
    photoreceptor_coord = np.zeros((n_t, M_f, 2))
    col = 0
    for i in range(M_f):    
        R_i = i+1
        r_i.append(R_i)

        for j in range(n_t):
            theta = 2*np.pi/n_t
            x = R_i*np.cos(theta*j) 
            y = R_i*np.sin(theta*j)
 
            photoreceptor_coord[j, col, :] = [y, x]
        col += 1
            
            
    return photoreceptor_coord, np.array(r_i)

def Radan_sample(img, fovea_xy, photoreceptor_coord):
    x_c = fovea_xy[1]
    y_c = fovea_xy[0]
    xy_coord = np.zeros_like(photoreceptor_coord)           
    xy_coord[:,:,0] =  photoreceptor_coord[:,:,0] + y_c     
    xy_coord[:,:,1] =  photoreceptor_coord[:,:,1] + x_c 
    
    c_shape = list(xy_coord.shape)
    im_size = img.shape
    h, w = im_size[0:2] 
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    
    if img.dtype == 'uint8':
        img = np.array(img,'float32')/255
    
    if img.ndim==3:
        c_shape[2] = 1
        # Convert image to log-polar coordinate representation
        interp2r = ndimage.map_coordinates(img[:,:,0], xy_coord.reshape(c_shape[0]*c_shape[1],2).T, order=1)
        interp2g = ndimage.map_coordinates(img[:,:,1], xy_coord.reshape(c_shape[0]*c_shape[1],2).T, order=1)
        interp2b = ndimage.map_coordinates(img[:,:,2], xy_coord.reshape(c_shape[0]*c_shape[1],2).T, order=1)
        retina_sample = np.concatenate((interp2r.reshape(c_shape), interp2g.reshape(c_shape), interp2b.reshape(c_shape)), axis=2)
        
    elif img.ndim==2:    
        interp2 = ndimage.map_coordinates(img, xy_coord.reshape(c_shape[0]*c_shape[1],2).T, order=1)     
        retina_sample = interp2.reshape(c_shape[0:2])
        
    return retina_sample