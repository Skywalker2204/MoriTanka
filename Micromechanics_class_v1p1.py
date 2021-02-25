# -*- coding: utf-8 -*-
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
====         ===  ==       ==  ==           |      M ontan
||\\        //||  ||       ||  ||           |      U niversitaet
|| \\      // ||  ||       ||  ||           |      L eoben
||  \\    //  ||  ||       ||  ||           |
||   \\  //   ||  ||       ||  ||           |      Institute for 
||    \\//    ||  ||       ||  ||           |      polymer processing
||            ||  ||       ||  ||           |
||            ||   =========    ==========  |      Author:    Sykwalker
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This class is for evaluation of a pvT curve, to detected the transition 
temperatures as a function of the pressure. 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import RK45
from scipy.optimize import minimize
import math

import os
import glob
import time

class Micromechanics():
    
    def __init__(self):
        self.__S = np.zeros((6,6))
        self.__nu = 0.3
        self.__I = np.diag(np.ones((6)))
        self.__C = np.zeros((6,6))

        
    def Eshelby(self, a, nu=0):
        """
        Calcuation of the Eshelby tensor of a ellipsoidical inclusion.
        Equations according to Tandon, G. and Weng, G. 1984.
        Discripe the shape of the inclusion as a function of the aspect ratio
        and the poisson ratio of the Matrix material
        """
        
        if nu == 0:
            nu = self.__nu
            
        S = np.zeros((6,6))
        if a==1:
            #For sphere inclusions
            S1 = (7-5*nu)/(15*(1-nu))
            S12 = (5*nu-1)/(15*(1-nu))
            S1212 = ((4-5*nu))/(15*(1-nu))
        
            S[0,0], S[1,1], S[2,2] = [S1]*3
            S[0,1], S[0,2], S[1,2], S[1,0], S[2,0], S[2,1]= [S12]*6
            S[3,3], S[4,4], S[5,5] = [S1212]*3

            
        elif a == np.inf:
            #for cylindical inclusions
            S10_20 = nu/(2*(1-nu))
            S11_22 = (5-4*nu)/(8*(1-nu))
            S12_21 = (4*nu-1)/(8*(1-nu))
            S33_44 = 0.5
            S55 = (3-4*nu)/(4*(1-nu))
            
            S[1,0], S[2,0] = [S10_20]*2
            S[1,1], S[2,2] = [S11_22]*2
            S[1,2], S[2,1] = [S12_21]*2
            S[3,3], S[4,4] = [S33_44]*2
            S[5,5] = S55
        
        else:
            #for generals ellipsoidical incusions
            if a > 1:
                g = a/((a**2-1)**(3/2))*(a*math.sqrt((a**2-1))-math.acosh(a))
            else:
                g = a/((1-a**2)**(3/2))*(math.acos(a)-a*math.sqrt((1-a**2)))
            """    
            old equations
            S[0,0] = 1/(2*(1-nu))*((4*a**2-2)/(a**2-1)-2*nu+((4*a**2-1)/(1-a**2)+2*nu)*g)
            S[1,1], S[2,2] = [1/(4*(1-nu))*((3*a**2)/(2*(a**2-1))-2*nu+((4*a**2-13)/(4*(a**2-1))+2*nu)*g)]*2
            S[0,1], S[0,2] = [1/(2*(1-nu))*((a**2)/((1-a**2))+2*nu+((2*a**2+1)/(2*(a**2-1))-2*nu)*g)]*2
            S[1,0], S[2,0] = [1/(4*(1-nu))*((2*a**2)/((1-a**2))+((2*a**2+1)/((a**2-1))+2*nu)*g)]*2
            S[1,2], S[2,1] = [1/(4*(1-nu))*((a**2)/(2*(a**2-1))+((4*a**2-1)/(4*(1-a**2))+2*nu)*g)]*2
            S[3,3], S[4,4] = [1/(2*(1-nu))*((2)/((1-a**2))-2*nu+1/2*((2*a**2+4)/((a**2-1))+2*nu)*g)]*2
            S[5,5] = 1/(2*(1-nu))*((a**2)/(2*(a**2-1))+((4*a**2-7)/(4*(a**2-1))-2*nu)*g)
            """
            S[0,0]         = 1/(2*(1-nu)) * (1-2*nu+(3*a**2-1)/(a**2-1)-(1-2*nu+(3*a**2)/(a**2-1)) * g)
            S[1,1], S[2,2] = [3/(8*(1-nu)) * a**2/(a**2-1) + 1/(4*(a**2-1)) * (1-2*nu+3/(4*(a**2-1))) * g]*2
            S[1,2], S[2,1] = [1/(4*(1*-nu)) * (a**2/(2*(a**2-1)) - (1-2*nu+3/(4*(a**2-1))) * g)]*2
            S[1,0], S[2,0] = [-1/(2*(1-nu)) * a**2/(a**2-1) + 1/(4*(1-nu)) * ((3*a**2)/(a**2-1) - (1 -2*nu)) *g]*2
            S[0,1], S[0,2] = [-1/(2*(1-nu)) * (1-2*nu+1/(a**2-1)) + 1/(2*(1-nu)) * (1-2*nu+3/(2*(a**2-1))) * g]*2
            S[5,5]         = 1/(4*(1-nu)) * (a**2/(2*(a**2-1))+(1-2*nu-3/(4*(a**2-1)))*g)
            S[3,3], S[4,4] = [1/(4*(1-nu)) * (1-2*nu-(a**2+1)/(a**2-1)-0.54*(1-2*nu-(3*(a**2+1))/(a**2-1)) * g)]*2
        self.__S = S
        return self.__S
        
    def Elastisity_Tensor(self, Em, nu=0):
        """
        Constrution of an isotropic Elastisity Tensor as a 6x6 Matrix
        """
        if nu == 0:
            nu = self.__nu
            
        E = np.zeros((6,6))
        #try:
        E11 = Em/(1+nu)*(1+(nu/(1-2*nu)))    
        E12 = (Em*nu)/((1+nu)*(1-2*nu)) 
        E1212 = Em/(2*(1+nu))
            
        E[0,0], E[1,1], E[2,2] = [E11]*3
        E[0,1], E[0,2], E[1,2], E[1,0], E[2,0], E[2,1]= [E12]*6
        E[3,3], E[4,4], E[5,5] = [E1212]*3
        
        return np.asarray(E)
    
    def calc_Adil(self, Em, Ei, S):
        """
        Calculation of the Concentration Tesnor of the inclusion within in a 
        generalized continuum
        """
        
        X = self.__I+S*np.linalg.inv(Em)*(Ei-Em)
        
        return np.linalg.inv(X)
        
    def rotation(self, M, alpha, beta):
        """
        Rotation of a 6x6 matrix by agles according to the polar coordinate system,
        using the zyx convertion also known as roll, pitch, yaw angle, of the given 
        Matrix
        """
        
        M_rot = np.zeros(M.shape)
        
        def index(j):
            if j == 3:
                i0 = [0,1]
            elif j == 4:
                i0 = [0,2]
            elif j == 5:
                i0 = [1,2]
            else:
                i0 = j
            
            return i0
    
    
        rot = self.rotation_Tensor(alpha, beta)
        A = np.zeros((3,3,3,3))
        for i in range(6):
            if i < 3:
                for j in range(3):
                    A[i, i, j, j] = M[i, j]
                
                for j in range(3, 6):
                    j0 = index(j)
                    A[i,i, j0[0], j0[1]],A[i,i, j0[1], j0[0]] = [M[i, j]/2]*2
            else:
                i0 = index(i)
                for j in range(3):
                    A[i0[0], i0[1], j, j],A[i0[1], i0[0], j, j] = [M[i, j]/2]*2
                    
                for j in range(3,6):
                    j0 = index(j)
                    #print(i0, j0)
                    (A[i0[0], i0[1], j0[0], j0[1]], A[i0[1], i0[0], j0[0], j0[1]],
                     A[i0[0], i0[1], j0[1], j0[0]], A[i0[1], i0[0], j0[1], j0[0]]) = [M[i,j]/4]*4
        
        A_rot = np.einsum('im,jn,kp,lq,mnpq', rot, rot, rot, rot, A)
        #print(A[1,1,1,2], A[1,1,2,1], M[1,5])
        for i in range(6):
            if i < 3:
                for j in range(3):
                    M_rot[i,j] = A_rot[i,i,j,j]
                for j in range(3,6):
                    j0 = index(j)
                    M_rot[i,j] = A_rot[i,i, j0[0], j0[1]]*2
            else:
                i0 = index(i)
                for j in range(3):
                    M_rot[i,j] = A_rot[i0[0], i0[1], j, j]*2
                for j in range(3,6):
                    j0 = index(j)
                    M_rot[i,j] = A_rot[i0[0],i0[1], j0[0], j0[1]]*4
                    
        
        return M_rot
        
        
    def rotation_Tensor(self, alpha, beta):
        """
        construction of the rotation matrix used in the rotate function
        """
        gamma = 0
        beta = beta *math.pi/180
        alpha = alpha *math.pi/180
        
        c, s=[[],[]]
        for theta in [gamma, beta, alpha]:
            c.append(math.cos(theta))
            s.append(math.sin(theta))

        r_g = np.asarray([[1,0,0],[0,c[0], s[0]], [0,-s[0], c[0]]])
        r_b = np.asarray([[c[1],0,-s[1]],[0,1, 0], [s[1], 0, c[1]]])
        r_a = np.asarray([[c[2], s[2], 0], [-s[2], c[2],0],[0,0,1]])
              
        R = np.linalg.multi_dot([r_g, r_b, r_a])
        return R
        
      
    def Mori_Tanaka(self, Em, Ei, Adil, xi):
       """
       Mori-Tanaka estimation for evaluation of a continuum rspone matrix of a 
       compound material
       """
            
       if type(xi) == np.float64: 
           C1 = xi*Ei*Adil+(1-xi)*Em
           C2 = xi*Adil+(1-xi)*self.__I

       else:
           c1m = np.zeros((6,6))
           c2m = np.zeros((6,6))
           xm = 1-np.sum(xi)
           for ei, adil, x in zip(Ei, Adil, xi):
               c1m += x*np.dot(ei, adil)
               c2m += ((x)*adil)
               
           C1 = xm*Em+c1m
           C2 = xm*self.__I+c2m
           #print(xm)
           
       return np.dot(C1, np.linalg.inv(C2))
    
    
    
    def CSCS(self, Em, Ei, a0, xi0, angle = [0,0],   **kwargs):
        """
        Calssical self-consistent scheme as implicite iterative scheme to calculate
        the response matrix of a given compound
        """
        options ={
                'max_Int': 1000, 
                'cov' : 1e-15, 
                'relax' : 1.}
        options.update(kwargs)
        if type(xi0) == np.float64: 
            xi0 = [xi0]
            angle = [angle]
        #First estimation:
        E10 = self.get_E1(Em)
        
        def fun(x, *args):
            E1, nu = x
            xi, a = args
            E = self.Elastisity_Tensor(E1, nu)

            C1 = Em
            C2 = np.zeros((6,6))
            for xii, an in zip(xi, angle):
                S = self.Eshelby(a, nu)
                adil = self.calc_Adil(E,Ei,S)
                adil = self.rotation(adil, an[0], an[1])
                C2 +=xii*np.dot((Ei-Em), adil)
                
            E_SCn = C1+C2
            #print(adil)
            return np.sum((E-E_SCn)**2)
       
        
        popt = minimize(fun, x0 = [E10, self.__nu], args=(xi0, a0))
        E0 = self.Elastisity_Tensor(popt.x[0], nu = popt.x[1])
        print(popt.fun)
        return E0, popt.fun
    
    
    def differential(self, Em, Ei, a, **kwargs):
        options ={
                'max_Int': 1000, 
                'cov' : 1e-15, 
                'relax' : 1.}
        options.update(kwargs)
        
        def dED(t, y):
            xi = t
            E1D, nu0 = y
            ED = self.Elastisity_Tensor(E1D, nu0)
            
            S = self.Eshelby(a, nu0)
            
            Adil = self.calc_Adil(ED, Ei, S)
            
            E = 1/(1-xi)*np.dot((Ei-ED), Adil)
            C = np.linalg.inv(E)
            nu = -C[1,0]/C[0,0]
            E1 = 1/C[0,0]
            return [E1, nu]
        
        y0 = [self.get_E1(Em), self.__nu]
        t0 = 0
        t_bound= 0.6
        
        z = RK45(dED, t0=t0, y0=y0, t_bound=t_bound,
                 vectorized = True, first_step = 0.01)
        res = []
        for i in range(options['max_Int']):
            z.step()
            res.append([z.t, z.y[0]])
            if z.t >= t_bound:
                break
        return np.asarray(res)
    """
    Misc functions of the class to get data and setting some boundarys
    """
    def get_E1(self, E):
        E1 = 1/np.linalg.inv(E)[0,0]
        return E1
    
    def set_nu(self, nu):
        self.__nu = nu
        
        
        
if __name__ == '__main__':
    aa = 3
    MT = Micromechanics()
    #MT.rotation(0,0)
    S = MT.Eshelby(aa)
    xi = np.linspace(0,0.6,9)
    fig, ax = plt.subplots()
    Em = MT.Elastisity_Tensor(2050)
    Ei = MT.Elastisity_Tensor(0)
    Adil = MT.calc_Adil(Em, Ei, S)
    #print(S)
    Ecs=[]
    E1 = []
    cov =[]
    
    def a(x):
        if x <= 75:
            return np.inf
        else:
            return 25/(x-75)

    E1m = []
    for x in xi:
        aa = np.inf#a((1-x)*100)
        S = MT.Eshelby(aa)
        Adil = MT.calc_Adil(Em, Ei, S)
        Adil1 = MT.rotation(Adil, 45, 0)
        Adil2 = MT.rotation(Adil, -45, 0)
        Emt = MT.Mori_Tanaka(Em, [Ei, Ei], [Adil1,Adil2], [x/2, x/2])
        E1m.append(MT.get_E1(Emt))
        
    E45m = np.asarray(E1m)
    #ax.plot(xi, E45m, label = 'MT multi 45')  
    
    E1m = []
    for x in xi:
        aa = np.inf#a((1-x)*100)
        xii= [x/20*7, x/20*7, x/20*6]
        angle = [[40,0], [-80,0], [-20,0]]
        #print(aa)
        S = MT.Eshelby(aa)
        Adil0 = MT.calc_Adil(Em, Ei, S)
        Adil =  []
        for an in angle:
            Adil.append(MT.rotation(Adil0, an[0],an[1]))

        
        Emt = MT.Mori_Tanaka(Em, [Ei, Ei, Ei], Adil, xii)
        E1m.append(MT.get_E1(Emt))
        
        Ecs.append(MT.get_E1(MT.CSCS(Em, Ei, aa, xii, angle = angle)[0]))
        
    E4060m = np.asarray(E1m)
    ax.plot(xi, E4060m, label = 'Mori-Tanaka', color = 'red')  
    #ax.plot(xi, Ecs, label = 'CSCS 4060')
    
    

    #print(Adil2)
 
    
    Ediff = MT.differential(Em, Ei, aa)
    #ax.plot(Ediff[:,0], Ediff[:,1], label = 'Differential')
    
    path = r'C:\Users\p1857809\Documents\Python Scripts\Mori_Tanaka'
    dat = np.loadtxt(path+'\XY_dat.txt', delimiter = '\t')
    dat4060 = np.loadtxt(path+r'\4060_dat.txt', delimiter = '\t')
    ax.scatter(dat[:,2], dat[:,0], label = 'Messdaten 45/90', color = 'black')
    ax.scatter(dat4060[:,2], dat4060[:,0], label = 'Messdaten 40/60', color = 'red')
    ax.legend(loc='best')
    
    ax.set(xlim=[0,0.6], ylim = [150,2050])
    ax.set_xlabel('Porosität')
    ax.set_ylabel("Biegemodul in MPa")
    plt.tight_layout()
    fig.savefig(path+r'\MT_estimate.png', dpi = 600)
    plt.show()
    plt.close()   
    
    fig, ax = plt.subplots()
    ax.scatter(dat[:,2], dat[:,1], label = 'Messdaten 45/90', color = 'black')
    ax.scatter(dat4060[:,2], dat4060[:,1], label = 'Messdaten 40/60', color = 'red')

    ax.legend(loc='best')
    
    #ax.set(xlim=[0,0.6], ylim = [150,2050])
    ax.set_xlabel('Porosität')
    ax.set_ylabel("Biegefestigkeit in MPa")
    plt.tight_layout()
    fig.savefig(path+r'\Biegefestigkeit.png', dpi = 600)
    plt.show()
    plt.close()     
    
    

