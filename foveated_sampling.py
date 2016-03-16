# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:18:57 2015

@author: arash
"""
import numpy as np
import pylab as plt
from scipy.interpolate import griddata
import cv2

from scipy import ndimage


def Retina_model(M_p = 30, M_f = 10, n_t = 6, sectioned = 1):
    r_i = []
    photoreceptor_coord = np.zeros((n_t, M_f*(M_f+1)/2+M_p*M_f, 2))
    col = 0
    for i in range(M_p+M_f):
        
        if i < M_f:
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
        else:
            N = M_f*n_t
            theta = np.pi/N 
            b = (np.sin(theta)*(np.sin(theta)+np.sqrt(2*np.cos(theta)+1)) + np.cos(theta))/(np.cos(theta))**2
            R_i = b*R_i 
            r_i.append(R_i)
            for k in range(M_f):        
                for j in range(n_t):
                    jj = j + k*n_t              
                    x = R_i*np.cos(2*theta*jj) * sectioned + R_i*np.cos(theta*(i+2*jj)) * (1-sectioned)
                    y = R_i*np.sin(2*theta*jj) * sectioned + R_i*np.sin(theta*(i+2*jj)) * (1-sectioned)
                    
                    photoreceptor_coord[j, col, :] = [y, x]
                col += 1
    return photoreceptor_coord, np.array(r_i)
    
    
          
def Map2LogPolar(img, fovea_xy, photoreceptor_coord, resampling = 'linear', verbos=False):
    x_c = fovea_xy[1]
    y_c = fovea_xy[0]
    xy_coord = np.zeros_like(photoreceptor_coord)    
    xy_coord[:,:,0] =  photoreceptor_coord[:,:,0] + y_c     
    xy_coord[:,:,1] =  photoreceptor_coord[:,:,1] + x_c 
    
    cortical_sizes = xy_coord.shape
    im_size = img.shape
    h, w = im_size[0:2] 
    
    if img.dtype == 'uint8':
        img = np.array(img,'float32')/255
    
    if img.ndim==3:
        # Convert image to log-polar coordinate representation
        interp2r = ndimage.map_coordinates(img[:,:,0], xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2).T, order=1)
        interp2g = ndimage.map_coordinates(img[:,:,1], xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2).T, order=1)
        interp2b = ndimage.map_coordinates(img[:,:,2], xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2).T, order=1)
        interp2r = interp2r.reshape(cortical_sizes[0:2])
        interp2g = interp2g.reshape(cortical_sizes[0:2])    
        interp2b = interp2b.reshape(cortical_sizes[0:2])              
        if verbos:
            # Log-Polar Representation
            plt.subplot(3,1,1)
            plt.imshow(interp2r,aspect='auto',interpolation='nearest')            
            plt.subplot(3,1,2)
            plt.imshow(interp2g,aspect='auto',interpolation='nearest')            
            plt.subplot(3,1,3)
            plt.imshow(interp2b,aspect='auto',interpolation='nearest')
            plt.show()
            
        return (interp2r, interp2g, interp2b)
    
    elif img.ndim==2:    
        interp2 = ndimage.map_coordinates(img, xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2).T, order=1)
        interp2 = interp2.reshape(cortical_sizes[0:2])
        if verbos:
            plt.figure(figsize=(10,2))
            plt.imshow(interp2,aspect='auto',interpolation='nearest')
            plt.colorbar()
            plt.show()
        return interp2

          
          
def RetinaLike(img, fovea_xy, photoreceptor_coord, resampling = 'linear', verbos=False):
    x_c = fovea_xy[1]
    y_c = fovea_xy[0]
    xy_coord = np.zeros_like(photoreceptor_coord)           
    xy_coord[:,:,0] =  photoreceptor_coord[:,:,0] + y_c     
    xy_coord[:,:,1] =  photoreceptor_coord[:,:,1] + x_c 
    
    cortical_sizes = xy_coord.shape
    im_size = img.shape
    h, w = im_size[0:2] 
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    
    if img.dtype == 'uint8':
        img = np.array(img,'float32')/255
    
    if img.ndim==3:
        # Convert image to log-polar coordinate representation
        interp2r = ndimage.map_coordinates(img[:,:,0], xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2).T, order=1)
        interp2g = ndimage.map_coordinates(img[:,:,1], xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2).T, order=1)
        interp2b = ndimage.map_coordinates(img[:,:,2], xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2).T, order=1)
        
        # Convert back to the image coordination 
        grid_zr = griddata(xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2), 
                           interp2r, (grid_y, grid_x), method=resampling, fill_value=0)
        grid_zg = griddata(xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2), 
                           interp2g, (grid_y, grid_x), method=resampling, fill_value=0)
        grid_zb = griddata(xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2), 
                           interp2b, (grid_y, grid_x), method=resampling, fill_value=0)
                           
        if verbos:
            # Log-Polar Representation
            plt.subplot(3,1,1)
            plt.imshow(interp2r.reshape(cortical_sizes[0:2]))
            
            plt.subplot(3,1,2)
            plt.imshow(interp2g.reshape(cortical_sizes[0:2]))
            
            plt.subplot(3,1,3)
            plt.imshow(interp2b.reshape(cortical_sizes[0:2]))
            plt.show()
            
        return cv2.merge((grid_zr, grid_zg, grid_zb))
    
    elif img.ndim==2:    
        interp2 = ndimage.map_coordinates(img, xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2).T, order=1)
        grid_z = griddata(xy_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2), 
                           interp2, (grid_y, grid_x), method=resampling, fill_value=0)
        if verbos:
            plt.figure(figsize=(10,2))
            plt.imshow(interp2.reshape(cortical_sizes[0:2]))
            plt.colorbar()
            plt.show()
        return grid_z
        

        
# Downsample pripheral windows to the base size (fovea size windows)
def block_mean(img, xy_c, base_size, scale):
    h, w = img.shape[0:2] 
    radius = base_size*scale/2
    rem = (base_size*scale)%2
    if (xy_c[0]>radius) and (xy_c[1]>radius) and (xy_c[0]+radius<h) and (xy_c[1]+radius<w):
        
        if img.ndim==3:
            n_ch = img.shape[2] 
            sample_set = []
            if scale > 1: 
                for ch in range(n_ch):
                    Lsample = img[xy_c[0]-radius:xy_c[0]+radius+rem,xy_c[1]-radius:xy_c[1]+radius+rem,ch]
                    X, Y = np.ogrid[0:base_size*scale, 0:base_size*scale]
                    regions = base_size * (X/scale) + Y/scale
                    sample_ch = ndimage.mean(Lsample, labels=regions, index=np.arange(regions.max() + 1))   
                    
                    #sample_set.append(sample_ch[...,np.newaxis])
                    sample_set.append(sample_ch)
            else:
                for ch in range(n_ch):
                    sample_ch = img[xy_c[0]-radius:xy_c[0]+radius+rem,xy_c[1]-radius:xy_c[1]+radius+rem,ch].flatten()
                    #sample_set.append(sample_ch[...,np.newaxis])
                    sample_set.append(sample_ch)    
            #sample = np.concatenate(sample_set,axis=1)   
            sample = np.concatenate(sample_set,axis=0)    
                        
        else:
            
            Lsample = img[xy_c[0]-radius:xy_c[0]+radius+rem,xy_c[1]-radius:xy_c[1]+radius+rem]
            if scale > 1:
                X, Y = np.ogrid[0:base_size*scale, 0:base_size*scale]
                regions = base_size * (X/scale) + Y/scale
                sample = ndimage.mean(Lsample, labels=regions, index=np.arange(regions.max() + 1))
                #sample.shape = (base_size, base_size)
            else:
                sample = Lsample.flatten()
        return sample
    else:
        return np.zeros((base_size*base_size*img.ndim))


def Sample_retina(img, fovea_xy, base_size, n_t, M_f, M_p):
    sectioned = 0
    
    photoreceptor_coord, R_i = Retina_model(M_p, M_f, n_t, sectioned)
    window_scale = np.ceil(R_i[1:]-R_i[0:-1])
    n_cols = photoreceptor_coord.shape[1]
    #n_t = photoreceptor_coord.shape[1]
    x_c = fovea_xy[1]
    y_c = fovea_xy[0]
    xy_coord = np.zeros_like(photoreceptor_coord)           
    xy_coord[:,:,0] =  photoreceptor_coord[:,:,0] + y_c     
    xy_coord[:,:,1] =  photoreceptor_coord[:,:,1] + x_c 
    
    if img.ndim ==3:
        n_ch = img.shape[2]
    else:
        n_ch = 1
    sample_m = np.zeros((n_t, n_cols, base_size*base_size*n_ch))
    
    for i in range(M_p+M_f-1):
        
        if i < M_f:
            for k in range(i):
                col = (i-1)*i/2 + k
                for t in range(n_t):
                    xy_c = [xy_coord[t,k,0], xy_coord[t,col,1]]
                    sample = block_mean(img, xy_c, base_size, int(window_scale[i]))
                    sample_m[t, col, :] = sample
        else:
            for k in range(M_f):
                col = (M_f-1)*M_f/2 + (i-M_f)*M_f + k
                for t in range(n_t):            
                    xy_c = [xy_coord[t,k,0], xy_coord[t,col,1]]
                    sample = block_mean(img, xy_c, base_size, int(window_scale[i]))
                    sample_m[t, col, :] = sample
    return sample_m


def find_neighbor(t, c, M_f, M_p, n_t):

    if c > M_f*(M_f+1)/2:
        r = M_f + np.ceil((c - M_f*(M_f+1)/2.)/M_f)
        print 'phrepheral'
    else:
        print 'fovea'
        r = (1+np.sqrt(1+8*c))/2
        if np.trunc(r)==r:
            r = r
        else:
            r = np.trunc(r)
        r = int(r)
    print 'r = ', r
    el_idx = np.arange((M_f*(M_f+1)/2+M_p*M_f)*n_t).reshape(((n_t,M_f*(M_f+1)/2+M_p*M_f)))
   
    
    #Prepheral Vision
    if r > M_f+1:
        
        if t > 0 and t < 5:
            neighbors = [el_idx[t-1, c+M_f] , el_idx[t, c+M_f], el_idx[t+1, c],
                 el_idx[t+1, c-M_f], el_idx[t, c-M_f], el_idx[t-1, c]]
        elif t == 0:
            neighbors = [el_idx[-1, c+M_f-1] , el_idx[0, c+M_f], el_idx[1, c],
                 el_idx[1, c-M_f], el_idx[0, c-M_f], el_idx[-1, c-1]]
        else:
            neighbors = [el_idx[4, c+M_f] , el_idx[5, c+M_f], el_idx[0, c+1],
                 el_idx[0, c-M_f+1], el_idx[5, c-M_f], el_idx[4, c]]
    #Fovea Vision
    elif r < M_f-1:

        rp0 = (1+np.sqrt(1+8*c))/2
        rp1 = (1+np.sqrt(1+8*(c+1)))/2
        if (rp1 == np.trunc(rp1)):
            print 'rp1'
            if t<5:
                p1 = el_idx[t+1, c-r+1] #blue
                p2 = el_idx[t, c+r+1]
                p3 = el_idx[t, c+r]
                p4 = el_idx[t, c-1]
                p5 = el_idx[t, c-r]
                p6 = el_idx[t+1, c-2*r+2]  #red
            else:
                p1 = el_idx[t-5, c-5] #blue
                p2 = el_idx[t, c+r+1]
                p3 = el_idx[t, c+r]
                p4 = el_idx[t, c-1]
                p5 = el_idx[t, c-r]
                p6 = el_idx[t-5, c-r-4]  #red
            neighbors = [ p1, p2, p3, p4, p5, p6]
            
        elif (rp0 == np.trunc(rp0)):
            print 'rp0'
            p1 = el_idx[t, c+1] #blue
            p2 = el_idx[t, c+r+1]
            p3 = el_idx[t, c+r]
            p4 = el_idx[t-1, c+2*r]
            p5 = el_idx[t-1, c+r-1]
            p6 = el_idx[t, c-r+1]  
            neighbors = [ p1, p2, p3, p4, p5, p6]
        else:
            
            p1 = el_idx[t, c-r+1] #blue
            p2 = el_idx[t, c+1]
            p3 = el_idx[t, c+r+1]
            p4 = el_idx[t, c+r]
            p5 = el_idx[t, c-1]
            p6 = el_idx[t, c-r] #red
            neighbors = [ p1, p2, p3, p4, p5, p6]
        
        #neighbors = None    
    #Boundary
    else:
        print 'boundary'
        rp0 = (1+np.sqrt(1+8*c))/2
        rp1 = (1+np.sqrt(1+8*(c+1)))/2
        if (rp1 == np.trunc(rp1)):
            print 'rp1'
            p1 = el_idx[t+1, c-r+1] #blue
            p2 = el_idx[t, c+r+1]
            p3 = el_idx[t, c+r]
            p4 = el_idx[t, c-1]
            p5 = el_idx[t, c-r]
            p6 = el_idx[t+1, c-2*r+2]  #red
            neighbors = [ p1, p2, p3, p4, p5, p6]
            
        elif (rp0 == np.trunc(rp0)):
            print 'rp0'
            
            if t>1 and t<5:
                p1 = el_idx[t, c-r+2] #blue
                p2 = el_idx[t, c+1]
                
                p3 = el_idx[2*t-4, c+r+t]
                p4 = el_idx[2*t-5, c+r+t]
                
                p5 = el_idx[t-1, c+r-1]
                p6 = el_idx[t, c-r+1]  
                
            elif t == 1:
                p1 = el_idx[t, c-r+2] #blue
                p2 = el_idx[t, c+1]
                
                p3 = el_idx[2*t+2, c+r]
                p4 = el_idx[2*t+1, c+r]
                
                p5 = el_idx[t-1, c+r-1]
                p6 = el_idx[t, c-r+1]  
                
            elif t == 0:
                p1 = el_idx[t, c-r+2] #blue
                p2 = el_idx[t, c+1]
                
                p3 = el_idx[2*t+2, c+r+7]
                p4 = el_idx[2*t+1, c+r+7]
                
                p5 = el_idx[t-1, c+r-1]
                p6 = el_idx[t, c-r+1] 
                
            
            else:
                p1 = el_idx[t, c-r+1] #blue
                p2 = el_idx[t, c+1]
                
                p3 = el_idx[t-4, c+r+6]
                p4 = el_idx[t-5, c+r+6]
                9
                p5 = el_idx[t-1, c+r-1]
                p6 = el_idx[t-1, c-1] #red
                
            neighbors = [ p1, p2, p3, p4, p5, p6]
        else:
            p1 = el_idx[t, c-r+2] #blue
            p2 = el_idx[t, c+1]
            
            p3 = el_idx[2*t-3, c+r+t-1]
            p4 = el_idx[2*t-4, c+r+t-1]
            
            p5 = el_idx[t, c-1]
            p6 = el_idx[t, c-r+1]  
            
            neighbors = [ p1, p2, p3, p4, p5, p6]    
    

    return neighbors  


if __name__ == '__main__':
    
    
    M_p = 80   # Number of rings in the peripheral region
    M_f = 6    # Number of rings in the fovea region
    n_t = 6    # Number of angular divisions (rows of the grid structure matrix)
    
    orig = cv2.imread('image_samples/image_0002.jpg')
    h,w,ch = orig.shape
    
    # location of the fixation point
    x_c = 270.
    y_c = 91.
        
    sectioned = 0
    cortical_sizes =  (n_t, M_f*(M_f+1)/2+M_p*M_f)
    
    # Coordination of sampling points 
    photoreceptor_coord, R_i = Retina_model(M_p, M_f, n_t, sectioned)
    
    R_fovea = R_i[M_f]
    R_prephery = R_i[-1]
    fovea_xy = [y_c, x_c]
    
    #Find the size of rectangles for each radial distance from fovea
    base_size = 7
    window_scale = np.ceil(R_i[1:]-R_i[0:-1])
    
    # Sample matrix
    sample_M = Sample_retina(orig, fovea_xy, base_size, n_t, M_f, M_p)
    
    foveated = RetinaLike(orig, fovea_xy, photoreceptor_coord, resampling = 'nearest', verbos=False)
    foveated = np.array(foveated*255,'uint8')
    
    
    
    ## Original Image
    plt.figure()
    plt.subplot(1,2,1)
    marked = orig
    marked[y_c-10:y_c+10:,x_c,0] = 255
    marked[y_c-10:y_c+10:,x_c,1] = 0
    marked[y_c-10:y_c+10:,x_c,2] = 0
    marked[y_c,x_c-10:x_c+10,0] = 255
    marked[y_c,x_c-10:x_c+10,1] = 0
    marked[y_c,x_c-10:x_c+10,2] = 0
    
    plt.imshow(marked[20:-20,20:-20,:])
    
    fig = plt.gcf()
    circle1 = plt.Circle((x_c-20, y_c-20),R_fovea,color='r',fill=False)
    circle2 = plt.Circle((x_c-20, y_c-20),R_prephery,color='g',fill=False)
    fig.gca().add_artist(circle1)
    fig.gca().add_artist(circle2)
    plt.axis('off')
    plt.title('Original Image')
    
    # Resampled Image
    plt.subplot(1,2,2)
    plt.imshow(foveated[20:-20,20:-20,:])
    fig = plt.gcf()
    circle1 = plt.Circle((x_c-20, y_c-20),R_fovea,color='r',fill=False)
    circle2 = plt.Circle((x_c-20, y_c-20),R_prephery,color='g',fill=False)
    fig.gca().add_artist(circle1)
    fig.gca().add_artist(circle2)
    plt.axis('off')
    plt.title('Foveated Image')
    plt.show()
    
    
    f=plt.figure(figsize=(8, 8))
    #color = np.concatenate((np.arange(cortical_sizes[1]).reshape((1,cortical_sizes[1])),
    #                        np.zeros((cortical_sizes[0]-1,cortical_sizes[1])) ),axis=0 )
    
    #color = np.arange((M_f*(M_f+1)/2+M_p*M_f)*n_t).reshape(((n_t,M_f*(M_f+1)/2+M_p*M_f)))/33
    color = np.arange((M_f*(M_f+1)/2+M_p*M_f)*n_t).reshape(((M_f*(M_f+1)/2+M_p*M_f,n_t))).T<36
    
    
    ## Sampling point scatter plot
    xy_photoreceptor = photoreceptor_coord.reshape(cortical_sizes[0]*cortical_sizes[1],2)
    plt.scatter(xy_photoreceptor[:,0],
                xy_photoreceptor[:,1],c=color,s=50, cmap='Greys')      
    plt.grid()
    plt.axis('equal')
    plt.show()
    
   