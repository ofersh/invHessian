'''
 A generic family of ellipsoid objective functions
 author: Ofer M. Shir
 date: Oct-2017
 ver: 1.0
 '''
import numpy as np
from scipy.linalg import hadamard

def objFunc(x,H,a) :
	return x.dot(H).dot(x) + x.dot(a)

def genHsphere(N,c=1) :
    return c*np.eye(N)

def genHellipse(N,c) :
	H = np.zeros((N,N))
	for i in range(N) :
		H[i,i] = 1+(i*(c-1)/(N-1))
	return H
 
def genHadamardHellipse (N,c) :
    H = genHellipse(N,c)
    R = hadamard(N) / np.sqrt(N)
    H = R.dot(H).dot(np.transpose(R))
    return H

def getRotation(N,theta) :
    v = np.ones((N,1))
    u = np.ones((N,1))
    for i in range(N) :
        if np.mod(i,2)==0 :
            u[i] = 0
        else :
            v[i] = 0
    v=v/np.linalg.norm(v)
    u=u/np.linalg.norm(u)
    R = np.eye(N) + np.sin(theta)*(u.dot(np.transpose(v)) - v.dot(np.transpose(u))) + (np.cos(theta)-1.0)*(v.dot(np.transpose(v)) + u.dot(np.transpose(u)))
    return R
    
def genRotatedHellipse (N,c,theta=.333*np.pi) :
    H = genHellipse(N,c)
    R = getRotation(N,theta)
    H = R.dot(H).dot(np.transpose(R))
    return H

def genHcigar(N,c) :
    H = c*np.eye(N)
    H[-1][-1] = 1.0
    return H

def genHdiscus(N,c) :
    H = np.eye(N)
    H[-1][-1] = c
    return H    
