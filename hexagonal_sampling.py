# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from scipy import ndimage


def Retina_model(M_p = 30, M_f = 10, n_t = 6):
    r_i = []
    photoreceptor_coord = np.zeros((n_t, M_f*(M_f+1)/2+M_p*M_f, 2))
    col = 0
    for i in range(M_f):    
        R_i = i+1
        r_i.append(R_i)
        for k in range(R_i):
            for j in range(n_t):
                theta = 2*np.pi/n_t
                x_a = R_i*(np.cos(theta*j) + ( np.cos(theta*(j+1))-np.cos(theta*j) )*k/R_i )
                y_a = R_i*(np.sin(theta*j) + ( np.sin(theta*(j+1))-np.sin(theta*j) )*k/R_i )
                
                x_b = R_i*np.cos(theta*j + ( theta*(j+1) - theta*j )*k/R_i )
                y_b = R_i*np.sin(theta*j + ( theta*(j+1) - theta*j )*k/R_i )
                
                photoreceptor_coord[j, col, :] = [ y_a + (y_b-y_a)*i/M_f, x_a + (x_b-x_a)*i/M_f]
            col += 1
    for i in range(M_f, M_f+M_p):        
        N = M_f*n_t
        theta = np.pi/N 
        b = (np.sin(theta)*(np.sin(theta)+np.sqrt(2*np.cos(theta)+1)) + np.cos(theta))/(np.cos(theta))**2
        R_i = b*R_i 
        r_i.append(R_i)
        for k in range(M_f):        
            for j in range(n_t):
                jj = j + k*n_t              
                x = R_i*np.cos(theta*(i+2*jj)) 
                y = R_i*np.sin(theta*(i+2*jj)) 
                
                photoreceptor_coord[j, col, :] = [y, x]
            col += 1
    return photoreceptor_coord, np.array(r_i)


def Retina_sample(img, fovea_xy, photoreceptor_coord):
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