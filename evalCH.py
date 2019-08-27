# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:21:38 2017

@author: ofers
"""
import numpy as np

def measureCE(C,H) :
    HC = H.dot(C)
    CH = C.dot(H)
    HC /= np.max(HC)
    CH /= np.max(CH)
    d = HC - CH
#    if np.max(d) > 0 :
#        d /= np.max(d)
    return np.linalg.norm(d,ord='fro')
    
def measureID (C,H) :
    HC = H.dot(C)
    m = np.max(HC)
    w, v = np.linalg.eig(HC/m) #consider replacing with eigh
    w=np.sort(w)
    return ((w[-1]/w[0])-1.0)
    
def e1 (C,H) :
#    mc = np.max(C)
#    C = (C/mc)
#    print(C)
    HC = H.dot(C)
    m = np.max(HC)
    HC = (HC/m)
    return np.max(np.abs(np.diag(HC)-1))
    
def e2 (C,H) :
    HC = H.dot(C)
    m = np.max(HC)
    HC = (HC/m)
#    for i in range(len(HC)) :
#        print(HC[i,:])
    return np.max(np.abs(np.triu(HC,1)))
    
def mdo(X,optimum) :
    res = 0
    L = np.shape(X)
    for i in range(L[0]) :
        res = res + np.linalg.norm(X[i]-optimum)
    return res / L[0]