'''
 Non-minima Covariance-versus-Hessian parallel calculations
 author: Ofer M. Shir
 date: 10-Dec-2017
 ver: 1.1
 '''
import numpy as np
import ellipsoidFunctions as Efunc
import os

def ellOFlambdaCompetition(lmbda,l,N,H,a,seed=None) :
    local_state = np.random.RandomState(seed)
    popF = np.empty([lmbda])
    popX = np.empty([N,lmbda])
    for k in range(lmbda) :
        z = local_state.normal(size=N)
        fz = Efunc.objFunc(z,H,a)
        popF[k] = fz
        popX[:,[k]] = z.reshape([N,1])
    iperm=np.argsort(popF)
    idx = (int)(iperm[l-1])
#    print("===",idx,":\n",popF[idx],"\n",popF)
    return popX[:,[idx]].ravel(),popF[idx].ravel()

if __name__ == "__main__" :
    N = np.array([3]) #[4,8,16,32,64])
    #Lmax = 10**3; dL = 5*10**1
    lmbda = list(range(100,1050,100))
#    lmbda.extend(range(100,1050,50))
#    lmbda.extend(range(2000,11000,1000))
    ell = 1
    fixedCondition = 5
    SampleSize=10**6
    for n in range(np.size(N)) :
        Nn = N[n]
        dirname = "rotEllipse"+str(Nn)+"D_"+str(fixedCondition)+"_ELL"+str(ell)+"_Lmax1e4Niter1e6"
        try :
            os.mkdir(dirname)            
        except FileExistsError :
            print("Directory already exists")
        H = Efunc.genRotatedHellipse(Nn,fixedCondition)
        a = np.ones((Nn,1))
        filename = os.path.join(dirname,"config")
        np.savez_compressed(filename,h=H,v=a,L=lmbda,S=SampleSize)
        for i in range(len(lmbda)) :
            Xwinners,Fwinners = [],[]
            for k in range(SampleSize) :
                zmin,fmin= ellOFlambdaCompetition(lmbda[i],ell,Nn,H,a,(i+1)*SampleSize+k)
                Xwinners.append(zmin)
                Fwinners.append(fmin)
            filename = os.path.join(dirname,str(lmbda[i]))
            np.savez_compressed(filename,f=Fwinners,x=Xwinners)
            print(lmbda[i])
