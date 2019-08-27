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
#    N = np.array([4,8,16,32,64])
    Nn = 4 
    lmbda = np.logspace(1,4,base=10,num=4)
    conditioning = np.logspace(2,20,base=2,num=19)
    SampleSize=10**6
    numProcess=40
    a = np.ones((Nn,1))
    for l in lmbda :
        dirname = "Hadamard"+str(Nn)+"D_L"+str(int(l))+"_Niter1e6_ConditioningUp"
        try :
            os.mkdir(dirname)            
        except FileExistsError :
            print("Directory already exists")
        filename = os.path.join(dirname,"config")
        np.savez_compressed(filename,v=a,C=conditioning,S=SampleSize)
        for c in conditioning :
            H = Efunc.genHadamardHellipse(Nn,c)
            Xwinners,Fwinners = [],[]
            r = Parallel(n_jobs=numProcess, verbose=5)(delayed(lambdaCompetition)((int(l)),Nn,H,a,k+(int(c))) for k in range(SampleSize)) 
            Xwinners,Fwinners = zip(*r)
            filename = os.path.join(dirname,str(int(c)))
            np.savez_compressed(filename,f=Fwinners,x=Xwinners)
            print(c)
