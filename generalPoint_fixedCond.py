'''
 Non-minima Covariance-versus-Hessian calculations
 author: Ofer M. Shir
 date: Oct-2017
 ver: 1.0
 '''
import numpy as np
import ellipsoidFunctions as Efunc

def lambdaCompetition(lmbda,N,H,a) :
    fmin = np.inf
    for l in range(lmbda) :
        z = np.random.normal(size=N)
        fz = Efunc.objFunc(z,H,a)
        if fz < fmin :
            fmin = fz
            zmin = z
    return zmin,fmin

if __name__ == "__main__" :
    N = np.array([4,8,16])
    Lmax = 10**4; dL = 500
    lmbda = np.arange(dL,Lmax+dL,dL)
    KStops=int(Lmax/dL)
    fixedCondition = 5
    SampleSize=10**3
    had=False
    for n in range(np.size(N)) :
        Nn = N[n]
        FF = [] #winners
        ZZ = [] #winning vectors
        filename = "qEllipse"+str(Nn)+"D_"+str(fixedCondition)+"SF_test"
        if had :
            H = Efunc.genHadamardHellipse(Nn,fixedCondition)
        else :
            H = Efunc.genHellipse(Nn,fixedCondition)
        a = np.ones((Nn,1))/Nn
    #preparing an array of lists
        Xwinners = np.empty(KStops,dtype=np.object)
        for i,v in enumerate(Xwinners): Xwinners[i]=[v,i]
        Fwinners = np.empty(KStops,dtype=np.object)
        for i,v in enumerate(Fwinners): Fwinners[i]=[v,i]
    #
        for i in range(KStops) :
            for k in range(SampleSize) :
                zmin,fmin=lambdaCompetition(lmbda[i],Nn,H,a)
                Xwinners[i].append(zmin)
                Fwinners[i].append(fmin)
            np.savez_compressed(filename,h=H,v=a,f=Fwinners,x=Xwinners,c=fixedCondition,L=Lmax,d=dL,S=SampleSize)
            print(lmbda[i])
