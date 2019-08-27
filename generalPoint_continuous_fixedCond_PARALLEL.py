'''
 Non-minima Covariance-versus-Hessian calculations
 author: Ofer M. Shir
 date: Sep-2017
 ver: 1.0
 '''

import numpy as np
import ellipsoidFunctions as Efunc
from joblib import Parallel, delayed
from tempfile import mkdtemp
import os.path as path

def runningLambdaCompetition(lmbda,N,H,a,dL,outputF,outputX,pid) :
    local_state = np.random.RandomState(pid+17)
    fmin = np.inf
    for l in range(lmbda) :
        z = local_state.normal(size=N)
        fz = Efunc.objFunc(z,H,a)
        if fz < fmin :
            fmin=fz
            zmin=z
        if (np.mod(l+1,dL)==0) :
            Lcurrent = (int)((l+1)/dL)-1
            outputX[Lcurrent][pid][:] = zmin
            outputF[Lcurrent][pid] = fmin
    return #Xwinners,Fwinners
#
if __name__ == "__main__" :
    N = np.array([4,8,16])
    Lmbda = 10**7
    dL = 5*10**5
    KStops=int(Lmbda/dL)
    fixedCondition = 5
    SampleSize=10**5
    numProcess=5
    for n in range(1) : #np.size(N)
        Nn = N[n]
        FF = [] #winners
        ZZ = [] #winning vectors
        filename = "Ellipse"+str(Nn)+"D_"+str(fixedCondition)+"L1e7Niter1e5_"+str(Lmbda)+"continuous"
        H = Efunc.genHellipse(Nn,fixedCondition)
        a = np.ones((Nn,1))/Nn
#        Xwinners = np.empty(KStops,dtype=np.object)
#        for i,v in enumerate(Xwinners): Xwinners[i]=[v,i]
#        Fwinners = np.memmap(sums_name, dtype=samples.dtype,shape=samples.shape[0], mode='w+')
#        for i,v in enumerate(Fwinners): Fwinners[i]=[v,i]
#        XXwinners,FFwinners = [], []
#        manager = multiprocessing.Manager()
#        Fwinners = manager.list(np.zeros([KStops,SampleSize]))
        Fwinners = np.memmap(path.join(mkdtemp(),'outputF.dat'), dtype='float32', mode='w+', shape = (KStops,SampleSize))
        Xwinners = np.memmap(path.join(mkdtemp(),'outputX.dat'), dtype='float32', mode='w+', shape = (KStops,SampleSize,Nn))
        Parallel(n_jobs=numProcess, verbose=5)(delayed(runningLambdaCompetition)(Lmbda,Nn,H,a,dL,Fwinners,Xwinners,k) for k in range(SampleSize)) 
#            XXwinners.append(X)
#            FFwinners.append(F)
#            if np.mod(k,int(SampleSize/10))==0 :
        np.savez_compressed(filename,h=H,f=Fwinners,x=Xwinners,c=fixedCondition,L=Lmbda,d=dL,S=SampleSize)
#                print(k)