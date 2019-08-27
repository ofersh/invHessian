'''
 Non-minima Covariance-versus-Hessian parallel calculations
 author: Ofer M. Shir
 date: 22-Nov-2017
 ver: 1.1
 '''
import numpy as np
import ellipsoidFunctions as Efunc
import os
from joblib import Parallel, delayed

def lambdaCompetition(lmbda,N,H,a,seed=None) :
    local_state = np.random.RandomState(seed)
    fmin = np.inf
    for l in range(lmbda) :
        z = local_state.normal(size=N)
        fz = Efunc.objFunc(z,H,a)
        if fz < fmin :
            fmin = fz
            zmin = z
    return zmin,fmin

if __name__ == "__main__" :
    N = np.array([16,32,64])
    #Lmax = 10**3; dL = 5*10**1
    lmbda = list(range(5,55,5))
    lmbda.extend(range(100,1050,50))
#    lmbda.extend(range(2000,11000,1000))
    #KStops=int(Lmax/dL)
    fixedCondition = 10 
    SampleSize=10**6
    numProcess=40
    had=False
    for n in range(np.size(N)) :
        Nn = N[n]
        dirname = "rot3Ellipse"+str(Nn)+"D_"+str(fixedCondition)+"SF_Lmax1e4Niter1e6"
        try :
            os.mkdir(dirname)            
        except FileExistsError :
            print("Directory already exists")
        if had :
            H = Efunc.genHadamardHellipse(Nn,fixedCondition)
        else :
            H = Efunc.genRotatedHellipse(Nn,fixedCondition,.333*np.pi)
        a = 1*np.ones((Nn,1))
        filename = os.path.join(dirname,"config")
        np.savez_compressed(filename,h=H,v=a,L=lmbda,S=SampleSize,c=fixedCondition)
        for i in range(len(lmbda)) :
            Xwinners,Fwinners = [],[]
            r = Parallel(n_jobs=numProcess, verbose=5)(delayed(lambdaCompetition)(lmbda[i],Nn,H,a,(i+1)*SampleSize+17*k) for k in range(SampleSize)) 
            Xwinners,Fwinners = zip(*r)
            filename = os.path.join(dirname,str(lmbda[i]))
            np.savez_compressed(filename,f=Fwinners,x=Xwinners)
            print(lmbda[i])
