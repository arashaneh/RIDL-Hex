# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:24:48 2015

@author: Arash
"""
import numpy as np
import pylab as plt
#from sklearn.linear_model import LassoLars, OrthogonalMatchingPursuit
from sklearn.utils import shuffle
from scipy.ndimage.interpolation import rotate
from OMP import rot_invar_omp, omp

def vis_filters(D, patch_size):
    m, k = D.shape
    V = D.T
    plt.figure(figsize=(8.4, 8))
    vmin = V.min()
    vmax = V.max()
    for i, comp in enumerate(V[:k]):
#        plt.subplot(n_theta,k/n_theta, i + 1)
        plt.subplot(20,24, i + 1)
        #comp_rgb = cv2.cvtColor(np.array(comp*255,'uint8').reshape(patch_size), cv2.COLOR_HSV2BGR)
        
        plt.imshow(comp.reshape(patch_size), interpolation='nearest',  vmin=vmin, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23) 
    
    
def rotate_filters(filters, n_theta, theta_t, patch_size):
    m, n_filters = filters.shape
    D_r = np.zeros((m, n_filters*n_theta))   
    for j in range(n_filters):
        ker = filters[:,j].reshape(patch_size)
        for t in range(n_theta):
            D_r[:,j*n_theta+t] = rotate(ker, (t-theta_t[j])*360./n_theta, 
                                axes=(1, 0), reshape=False, order=3, 
                                mode='nearest').flatten()
    return D_r


def shift_filters(filters, n_theta, theta_t, n_sections):
    m, n_filters = filters.shape
    n_col = m/n_sections
    D_r = np.zeros((m, n_filters*n_theta))   
    for j in range(n_filters):
        ker = filters[:,j].reshape((n_sections, n_col))
        for t in range(n_theta):
            D_r[:,j*n_theta+t] = np.roll(ker, (t-theta_t[j])*n_sections/n_theta, axis=0).flatten()
    return D_r

    
def select_correlated_orientation(D_r, x_t, n_theta, prev_theta = None, B_t = None, 
    patch_size = None, n_sections = None):
    m, k = D_r.shape
    n_filters = k/n_theta  
    D_t = np.zeros((m,n_filters))
    theta_t = np.zeros(n_filters,'int')
    B_r = np.zeros_like(B_t)
    for j in range(n_filters):
        dr = D_r[:,j*n_theta:(j+1)*n_theta]
        theta_t[j] = np.argmax( np.abs(np.dot(x_t.T, dr)).flatten()/np.linalg.norm(dr, axis = 0)) #/np.linalg.norm(dr, axis = 0)
        #theta_t[j] = 0
        D_t[:,j] = dr[:, theta_t[j]]
        
        if B_t is not None:
            b_t = B_t[:, j].reshape(patch_size)
            #B_r[:,j] = np.roll(b_t, (-theta_t[j])*n_sections/n_theta, axis=0).flatten()
            B_r[:, j] = rotate(b_t, (theta_t[j]-prev_theta[j])*360./n_theta, 
                        axes=(1, 0), reshape=False, order=3, mode='nearest').flatten()
    return theta_t, D_t, B_r



def rotate_col(b_t, pre_theta, theta_t, n_theta, patch_size):
    m, sparsity = b_t.shape
    
    b_t_new = np.zeros_like(b_t)
    for j in range(sparsity):
         b_t_new[:, j] = rotate(b_t[:,j].reshape(patch_size), (theta_t[j]-pre_theta[j])*360./n_theta, 
                        axes=(1, 0), reshape=False, order=3, mode='nearest').flatten()
    return b_t_new
    
    

def shift_col(b_t, pre_theta, theta_t, n_theta, n_sections)   :
    m, sparsity = b_t.shape     
    b_t_new = np.zeros_like(b_t)
    for j in range(sparsity):
         b_t_new[:, j] = np.roll(b_t[:,j], (theta_t[j]-pre_theta[j])*n_sections/n_theta, axis=0).flatten()
    return b_t_new    




def sq_dict_learning(row_data, mask, D_0 = None, n_filters = 20,  
    eta = 1e-2, sparsity = 10, n_epochs = 4, EV_SCORE = True):
    ''' 
    k: Number of dictionary items
    n_theta: Number of orientated realization of the filter
    '''    
    #Shuffle the data
    data = shuffle(row_data).T
    m, n = data.shape
    effective_dim = mask.sum()
    dummy_dim = mask.shape[0]*mask.shape[1]
    dim_ratio = float(dummy_dim)/effective_dim
    

    if D_0 is None:
        D_base = 1-2*np.random.rand(m,n_filters)
        D_base -= np.expand_dims(np.mean(D_base, axis=0), 0)*dim_ratio
        D_base /= np.linalg.norm(D_base,axis=0)
        D_t = D_base
    else:
        D_t = D_0   

    losses = []
    for epoch in range(n_epochs):
        
        for t in range(n):
            x_t = data[:,t]
               
            # Sparse Coding 
            idx_t, alphas_t = omp(D_t, x_t, sparsity)
            
            
            # Dictionary Update     
            ##LMS
            nnzero_coeff = len(alphas_t)
            for j in range(nnzero_coeff):
                D_t[:,idx_t[j]] += eta * (x_t - D_t[:,idx_t[j]]*alphas_t[j])*alphas_t[j]
                D_t[:,idx_t[j]] /= max(np.linalg.norm(D_t[:,idx_t[j]],ord=2),1.)
                
                

                
            D_t -= np.expand_dims(np.mean(D_t, axis=0), 0)*dim_ratio
            D_t /= np.expand_dims(np.linalg.norm(D_t, axis=0), axis=0)
        
                    
            if EV_SCORE and (t%500 == 0):
                loss = score_dict(data, D_t, sparsity )
                losses.append(loss)
        data = shuffle(data.T).T
    return D_t, losses    
    
    
def sq_rot_invar(row_data, mask, D_0 = None, n_filters = 20, n_theta = 6,
    eta = 1e-2, sparsity = 10, n_epochs = 4, EV_SCORE = True):
    ''' 
    k: Number of dictionary items
    n_theta: Number of orientated realization of the filter
    '''    
    #Shuffle the data
    #data = shuffle(row_data).T
    data = row_data.T
    m, n = data.shape
    
    effective_dim = mask.sum()
    dummy_dim = mask.shape[0]*mask.shape[1]
    dim_ratio = float(dummy_dim)/effective_dim
    
    # Number of iterations
    patch_size = mask.shape
    mask_D = np.repeat(mask.reshape((m,1)),n_filters,axis=1)
    if D_0 is None:
        D_base = 1-2*np.random.rand(m,n_filters)
        D_base -= np.expand_dims(np.mean(D_base, axis=0), 0)*dim_ratio
        D_base *= mask_D
        D_base /= np.linalg.norm(D_base,axis=0)
        D_t = D_base
    else:
        D_t = mask_D*D_0
        
    Theta_t = np.zeros(n_filters,'int')
    D_r = rotate_filters(D_t, n_theta, Theta_t, patch_size)
    D_r = D_r - np.expand_dims(np.mean(D_r, axis=0), 0)*dim_ratio
    D_r /= np.expand_dims(np.linalg.norm(D_r, axis=0), axis=0)
    
    
    losses = []
    for epoch in range(n_epochs):
        
        for t in range(n):
            x_t = data[:,t]
            
            # Selecting theta s that correlate most with x_t   
            idx_t, alphas_t, theta_t = rot_invar_omp(D_r, x_t, sparsity, n_theta) 
            
            # extract corresponding columns from B_t and rotate them according to theta t
            d_t = D_r[:,idx_t]
            
            ## Dictionary Update
#            ##LMS
            D_t[:,idx_t/n_theta] = d_t
            nnzero_coeff = len(alphas_t)
            for j in range(nnzero_coeff):
                D_t[:,idx_t[j]/n_theta] += eta * (x_t - D_t[:,idx_t[j]/n_theta]*alphas_t[j])*alphas_t[j]
                D_t[:,idx_t[j]/n_theta] /= max(np.linalg.norm(D_t[:,idx_t[j]/n_theta],ord=2),1.)
                     
                      
            # Rotate D_t back to generate D_r     
            Theta_t[idx_t/n_theta] = theta_t
            D_r = rotate_filters(D_t, n_theta, Theta_t, patch_size)    
            if EV_SCORE and (t%500 == 0):
                loss = score_rot_invar_dic(data, D_r, n_theta, sparsity, mask )
                losses.append(loss)
        data = shuffle(data.T).T
    return D_r, losses    
    


def sec_rot_invar(row_data,  D_0 = None, n_filters = 20, n_theta = 6,
    n_sections = 6, eta = 1e-2, sparsity = 10, n_epochs = 4, EV_SCORE = True):
    ''' 
    k: Number of dictionary items
    n_theta: Number of orientated realization of the filter
    '''    
    #Shuffle the data
    #data = shuffle(row_data).T
    data = row_data.T
    m, n = data.shape
    
    
    if D_0 is None:
        D_base = 1-2*np.random.rand(m,n_filters)
        D_base -= np.expand_dims(np.mean(D_base, axis=0), 0)
        D_base /= np.linalg.norm(D_base,axis=0)
        D_t = D_base
    else:
        D_t = D_0
        
    Theta_t = np.zeros(n_filters,'int')
    
    D_r = shift_filters(D_t, n_theta, Theta_t, n_sections)
    D_r = D_r - np.expand_dims(np.mean(D_r, axis=0), 0)
    D_r /= np.expand_dims(np.linalg.norm(D_r, axis=0), axis=0)
    
    
    losses = []
    for epoch in range(n_epochs):
        
        for t in range(n):
            x_t = data[:,t]
            
            # Selecting theta s that correlate most with x_t   
            idx_t, alphas_t, theta_t = rot_invar_omp(D_r, x_t, sparsity, n_theta) 
            
            # extract corresponding columns from B_t and rotate them according to theta t
            d_t = D_r[:,idx_t]
            D_t[:,idx_t/n_theta] = d_t
             
            Alpha_t = np.zeros((n_filters,1))
            Alpha_t[idx_t/n_theta,0] = alphas_t

            ## Dictionary Update
#            ##LMS       
            nnzero_coeff = len(alphas_t)
            for j in range(nnzero_coeff):
                D_t[:,idx_t[j]/n_theta] += eta * (x_t - D_t[:,idx_t[j]/n_theta]*alphas_t[j])*alphas_t[j]
                D_t[:,idx_t[j]/n_theta] /= max(np.linalg.norm(D_t[:,idx_t[j]/n_theta],ord=2),1.)
        
                        
            # Rotate D_t back to generate D_r     
            Theta_t[idx_t/n_theta] = theta_t
            D_r = shift_filters(D_t, n_theta, Theta_t, n_sections)

            if EV_SCORE and (t%500 == 0):
                loss = score_rot_invar_dic(data, D_r, n_theta, sparsity)
                losses.append(loss)
        data = shuffle(data.T).T
    return D_r, losses    




    
    
def score_rot_invar_dic(data, D_r, n_theta, sparsity, mask = None): 
    m, n = data.shape
    L = 0 
    if mask is None:
        mask_e = 1
    else:
        mask_e = mask.flatten()
    for t in range(n):
        x_t = data[:,t]

        idx_t, alphas_t, theta_t = rot_invar_omp(D_r, x_t, sparsity, n_theta) 
        
        d_t = D_r[:,idx_t]    
        e_t = (x_t - np.dot(d_t, alphas_t))*mask_e

        L += 1./(2*n)*np.dot(e_t.T,e_t).sum() 
    return L
    
    
def score_dict(data, D_t, sparsity):
    m, n = data.shape
    L = 0   
    for t in range(0,n):
        x_t = data[:,t]
        
        idx_t, alphas_t = omp(D_t, x_t, sparsity)
        d_t = D_t[:,idx_t]
        e_t = np.dot(d_t,alphas_t) - x_t
        
        L += 1./(2*n)*np.dot(e_t.T,e_t).sum() 
    return L