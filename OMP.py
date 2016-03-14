# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 12:16:56 2015

@author: Arash
"""

import numpy as np

def omp(D, y, sparsity):
    nzero = np.copy(sparsity)
    delta = 1e-6
    m = D.shape[0]
    l = D.shape[1]
    alpha = np.zeros(l)
    
    dic_mask = np.ones(l)
    index_set = []

    r_k = np.copy(y)
    r_dot_D = np.dot(r_k, D)
    
    if np.max(np.abs(r_dot_D)) < delta:
        return alpha
        
    j = np.argmax(np.abs(r_dot_D))
    
    index_set.append(j)
    dic_mask[j] = 0
    X_k = D[:,j].reshape(m,1)
    
    a_k = np.array([r_dot_D[j]])

    A_k = np.array([[np.dot(X_k[:,0], X_k[:,0])]])
    A_k_inv = 1./A_k

    r_k = y - np.dot(X_k,a_k)

    for k in range(1,sparsity):
        r_dot_D = np.dot(r_k, D).flatten()

        if np.max(np.abs(r_dot_D*dic_mask)) < delta:
            nzero = k
            break
        j = np.argmax(np.abs(r_dot_D*dic_mask))
        index_set.append(j)
        dic_mask[j] = 0

        x_k = D[:,j]
        x_k_x_k = np.dot(x_k, x_k)

        v_k = np.dot(X_k.T, x_k )
        b_k = np.dot(A_k_inv, v_k)
        
        A_k = np.concatenate((A_k, v_k.reshape(k,1)), axis=1)
        A_k = np.concatenate((A_k, np.concatenate((v_k, [x_k_x_k])).reshape(1,k+1)), axis=0)

        beta = 1./(1. - np.dot(v_k, b_k))  
        A_inv11 = A_k_inv + beta*np.dot(b_k, b_k.T)  
         
        A_k_inv = np.concatenate((A_inv11, (-beta*b_k).reshape(k,1)), axis=1)
        
        A_k_inv = np.concatenate((A_k_inv, np.concatenate((-beta*b_k.T, np.array([beta]))).reshape(1,k+1)), axis=0)

        alpha_k = (r_dot_D[j]/(x_k_x_k - np.dot(v_k, b_k)))

        a_k = a_k - alpha_k*b_k.flatten()
        a_k = np.concatenate((a_k,[alpha_k]))

        X_k = np.concatenate((X_k, x_k.reshape(m,1)), axis=1)

        r_k = y - np.dot(X_k, a_k.T)

    for k in range(nzero):  
        alpha[index_set[k]] = a_k.flatten()[k]
    
    return np.array(index_set), a_k
    #return alpha
        
def rot_invar_omp(D, y, sparsity, n_theta):
    nzero = np.copy(sparsity)
    delta = 1e-9
    m = D.shape[0]
    l = D.shape[1]
   
    dic_mask = np.ones(l)
    index_set = []
    theta_set = []

    r_k = np.copy(y)
    r_dot_D = np.dot(D.T, r_k)

    if np.max(np.abs(r_dot_D)) < delta:
        return np.array(index_set), np.array([]), np.array(theta_set)
        
    j = np.argmax(np.abs(r_dot_D))
    
    index_set.append(j)
    theta_set.append(j%n_theta)
    #change masking and save rotation
    dic_mask[j] = 0
    X_k = D[:,j].reshape(m,1)
    
    a_k = np.array([r_dot_D[j]])
    
    A_k = np.array([[np.dot(X_k[:,0], X_k[:,0])]])
    A_k_inv = 1./A_k
    r_k = y - np.dot(X_k,a_k)

    for k in range(1,sparsity):
        r_dot_D = np.dot(D.T, r_k)
        if np.max(np.abs(r_dot_D*dic_mask)) < delta:
            #TODO: change the number of selected elements
#            nzero = k
            print 'stopping due to reaching com. tol.'
            break
        j = np.argmax(np.abs(r_dot_D*dic_mask))
        index_set.append(j)
        theta_set.append(j%n_theta)
        #change masking and save rotation
        dic_mask[j] = 0

        x_k = D[:,j]
        x_k_x_k = np.dot(x_k, x_k)

        v_k = np.dot(X_k.T, x_k )
        b_k = np.dot(A_k_inv, v_k)
        
        A_k = np.concatenate((A_k, v_k.reshape(k,1)), axis=1)
        A_k = np.concatenate((A_k, np.concatenate((v_k, [x_k_x_k])).reshape(1,k+1)), axis=0)

        beta = 1./(1. - np.dot(v_k, b_k))  
        A_inv11 = A_k_inv + beta*np.dot(b_k, b_k.T)  
         
        A_k_inv = np.concatenate((A_inv11, (-beta*b_k).reshape(k,1)), axis=1)
        
        
#        A_k_inv = np.concatenate((A_k_inv, np.concatenate((-beta*b_k.T, np.array([beta])),axis=1).reshape(1,k+1)), axis=0)
        A_k_inv = np.concatenate((A_k_inv, np.concatenate((-beta*b_k.T, np.array([beta]))).reshape(1,k+1)), axis=0)
        
        alpha_k = (r_dot_D[j]/(x_k_x_k - np.dot(v_k, b_k)))

        a_k = a_k - alpha_k*b_k.flatten()
        a_k = np.concatenate((a_k,[alpha_k]))

        X_k = np.concatenate((X_k, x_k.reshape(m,1)), axis=1)
        r_k = y - np.dot(X_k, a_k)

    return np.array(index_set), np.array(a_k), np.array(theta_set)
            