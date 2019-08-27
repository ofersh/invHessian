'''
 Non-minima Covariance-versus-Hessian calculations
 author: Ofer M. Shir
 date: Sep-2017
 ver: 1.0
 '''

import numpy as np
from scipy.linalg import hadamard
import ellipsoidFunctions as Efunc

def runningLambdaCompetition(lmbda,N,H,a,dL) :
    fmin = np.inf
    for l in range(lmbda) :
        z = np.random.normal(size=N)
        fz = Efunc.objFunc(z,H,a)
        if fz < fmin :
            fmin=fz
            zmin=z
        if (np.mod(l+1,dL)==0) :
            Lcurrent = (int)((l+1)/dL)-1
            Xwinners[Lcurrent].append(zmin)
            Fwinners[Lcurrent].append(fmin)
    return #Xwinners,Fwinners
#
if __name__ == "__main__" :
    N = np.array([4,8,16])
    Lmbda = 10**4
    dL = 10**3
    KStops=int(Lmbda/dL)
    fixedCondition = 5
    SampleSize=500
    for n in range(1) : #np.size(N)
        Nn = N[n]
        FF = [] #winners
        ZZ = [] #winning vectors
        filename = "hadamardEllipse"+str(Nn)+"D_"+str(fixedCondition)+"scaleL"+str(Lmbda)+"quater"
        H = Efunc.genRotatedHellipse(Nn,fixedCondition)
        a = np.ones((Nn,1))/Nn
        Xwinners = np.empty(KStops,dtype=np.object)
        for i,v in enumerate(Xwinners): Xwinners[i]=[v,i]
        Fwinners = np.empty(KStops,dtype=np.object)
        for i,v in enumerate(Fwinners): Fwinners[i]=[v,i]
#        XXwinners,FFwinners = [], []
        for k in range(SampleSize) :
            runningLambdaCompetition(Lmbda,Nn,H,a,dL)
#            XXwinners.append(X)
#            FFwinners.append(F)
            if np.mod(k,int(SampleSize/10))==0 :
                np.savez_compressed(filename,h=H,f=Fwinners,x=Xwinners,c=fixedCondition,L=Lmbda,d=dL,S=SampleSize)
                print(k)