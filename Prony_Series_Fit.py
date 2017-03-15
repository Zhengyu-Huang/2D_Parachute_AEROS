#This file fits Prony series of visco materials
#Determining a Prony Series for a Viscoelastic Material From Time Strain Data
#https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20000052499.pdf
import math
import numpy as np
import matplotlib.pyplot as plt
def Strain_Result(stress, para, E0, k = 100):
    '''
    :param strain: shape(2,n)
                   t0, t1, t2 ... t_{n-1}
                   ep0, ep1 ..... ep_{n-1}
    :param para: shape(2,m)
                   tau0,tau1 ...tau_{m-1}
                   p0,  p1...   p_{m-1}

    :param E0: double, Young's module

    :return: stress: shape(3,k)
                          0, dt, 2dt ... (k-1)dt
              stress   sig0, sig1 ... sig_{k-1}
              strain   ep0,  ep1  ...  ep_{k-1}
    '''

    def Yt(t,para,E0):
        '''
        :param t time
        :param para: shape(2, m)
                      tau0, tau1...tau_{m - 1}
                      p0,   p1... p_{m - 1}

        :param E0: parameter

        :return Yt value
        '''
        return E0*(1- np.dot(para[1,:], 1 - np.exp(-t/para[0,:])))

    def Int_Yt(t1,t2, para, E0):
        '''
        :param t1, t2: time
        :param para: shape(2, m)
                      tau0, tau1...tau_{m - 1}
                      p0,   p1... p_{m - 1}

        :param E0: parameter
        :return integral of Yt from t1 to t2
        '''
        return E0*(t2 - t1) - E0 * np.dot(para[1, :], t2 - t1 + para[0,:]* (np.exp(-t2 / para[0, :]) - np.exp(-t1/para[0,:])))

    assert(math.isclose(stress[1,0], stress[1,-1], rel_tol = 1e-6))

    strain = np.zeros((3,k),dtype='f')

    strain[0,:] = np.linspace(stress[0,0],stress[0,-1],num=k)

    ep0 = stress[1,0]

    _,m = np.shape(stress)

    for i in range(k):
        t = strain[0,i]
        strain[1,i] = ep0*Yt(t, para, E0)
        for j in range(m):
            t1,t2 = stress[0,j], min(stress[0,j+1], t)
            depdt = (stress[1,j+1] - stress[1,j])/(stress[0,j+1] - stress[0,j])
            strain[1,i] += depdt *Int_Yt(t - t2, t - t1,para,E0)
            if(stress[0,j+1] >= t):
                strain[2,i] = stress[1,j] + depdt * (t - stress[0,j])
                break;

    return strain


if __name__ =="__main__":
    TESTCASE = 3
    if(TESTCASE == 1):
        #This is the test case in "Determining a Prony Series for a Viscoelastic Material From Time Strain Data"
        ep1,ep2  = 0.01 ,0.0
        stress = np.array([[0, 5 , 55, 60 , 110, 115],[0, ep1, ep1, ep2 ,ep2, 0]])
        E0, para = 1e+09, np.array([[10, 100], [0.2, 0.1]])
    elif(TESTCASE == 2):
        # This is the test case in "Determining a Prony Series for a Viscoelastic Material From Time Strain Data"
        # With more load cycle
        ep1, ep2 = 0.01, 0.0
        stress = np.array([[0, 5,   55,   60, 110, 115, 165, 170, 220, 225, 275, 280, 330, 335 ],
                           [0, ep1, ep1, ep2, ep2, ep1, ep1, ep2, ep2, ep1, ep1, ep2, ep2,  0]])
        E0, para = 1e+09, np.array([[10, 100], [0.2, 0.1]])

    elif(TESTCASE == 3):
        # lbf/in^2 = 6894.76 Pa
        # 0.003 inch = 7.62e-5 m
        # strain stress(Pa)
        #   0.0       0.0
        #   0.143   37lbf/in = 12333.333333333334lbf/in^2 = 85035339.91Pa
        ep1, ep2 = 0.143, 0.0
        stress = np.array([[0, 10,  100,     110, 200, 210,    300, 310, 400,    410, 500, 510,   600, 610, 700,   710, 800, 810,  900, 910, 1000 ] ,
                           [0, ep1, ep2,     ep1, ep2, ep1,    ep2, ep1, ep2,    ep1, ep2,ep1,    ep2, ep1,ep2,    ep1, ep2,ep1,   ep2, ep1,ep2]])
        E0, para = 6.8e+8, np.array([[10, 50, 100], [0.4, 0.3, 0.2]])
        #E0, para = 6.8e+8, np.array([[10, 50, 100], [0.2, 0.1, 0.04]])


    strain = Strain_Result(stress,para,E0, k = 1000)
    lbf_in_to_pa = 6894.75729/0.003
    plt.figure(1)
    plt.plot(strain[0,:], strain[2,:])
    plt.xlabel('t')
    plt.ylabel('strain')
    plt.figure(2)
    plt.plot(strain[0, :], strain[1, :]/lbf_in_to_pa)
    plt.xlabel('t')
    plt.ylabel('stress(lbf/in)')

    plt.figure(3)
    plt.plot(strain[2, :], strain[1, :]/lbf_in_to_pa)
    plt.xlabel('strain')
    plt.ylabel('stress(lbf/in)')
    plt.show()

