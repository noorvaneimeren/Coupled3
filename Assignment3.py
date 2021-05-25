#%% MAIN 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Assignment2 as fun
import scipy.integrate
from scipy import interpolate

# Then we start running our model.
# First we require the domain discretization
# Domain
nIN = 101 #1 meter depth
zIN = np.linspace(-1.0 , 0, num=nIN).reshape(nIN, 1) #Internodes
# nIN = np.shape(zIN)[0]
zN = np.zeros(nIN - 1).reshape(nIN - 1, 1) #Nodes
zN[0, 0] = zIN[0, 0] #both nodes and internodes start at -1
zN[1:nIN - 2, 0] = (zIN[1:nIN - 2, 0] + zIN[2:nIN - 1, 0]) / 2
zN[nIN - 2, 0] = zIN[nIN - 1] #both nodes and internodes end at 0
nN = np.shape(zN)[0] #number of nodes (10)

ii = np.arange(0, nN - 1) # [0 1 2 3 4 5 6 7 8]
dzN = (zN[ii + 1, 0] - zN[ii, 0]).reshape(nN - 1, 1) # [0.25 0.1 0.1 0.1 0.1 0.1 0.1 0.15]
dzIN = (zIN[1:, 0] - zIN[0:-1, 0]).reshape(nIN - 1, 1) # [0.1 0.1 0.1 ...]

# collect model dimensions in a namedtuple: modDim
mDim = {'zN' : zN,
        'zIN' : zIN,
        'dzN' : dzN,
        'dzIN' : dzIN,
        'nN' : nN,
        'nIN' : nIN
        }
mDim = pd.Series(mDim)

rhoW = 1000  # [kg/m3] density of water
rhoS = 2650  # [kg/m3] density of solid phase
rhoB = 1700  # %[kg/m3] dry bulk density of soil
n = 1 - rhoB / rhoS  # [-] porosity of soil = saturated water content.
q = 0.75  # quartz content

# [W/(mK)] thermal conductivity of water (Remember W = J/s)
lambdaWat = 0.58
lambdaQuartz = 6.5  # [W/(mK)] thermal conductivity of quartz
lambdaOther = 2.0  # [W/(mK)] thermal conductivity of other minerals

lambdaSolids = lambdaQuartz ** q * lambdaOther ** (1 - q)
lambdaBulk = lambdaWat ** n * lambdaSolids ** (1 - n)

# collect soil parameters in a namedtuple: soilPar
'zetaBN', 'lambdaIN'
sPar = {'VGa': np.ones(np.shape(zN)) * 2,  # alpha[1/m]
        'VGn': np.ones(np.shape(zN)) * 3,  # n[-]
        'VGm': np.ones(np.shape(zN)) * (1 - 1 / 3),  # m = 1-1/n[-]
        'theta_s': np.ones(np.shape(zN)) * 0.4,  # saturated water content
        'theta_r': np.ones(np.shape(zN)) * 0.01,  # residual water content
        'Ks': np.ones(np.shape(zN)) * 0.05,  # [m/day]
        'zetaBN': np.ones(np.shape(zN)) * ((1 - n) * zetaSol
                                                    + n * zetaWat),
        'lambdaIN': np.ones(np.shape(zIN)) * lambdaBulk * (24 * 3600)}
sPar = pd.Series(sPar)          

# ## Definition of the Boundary Parameters
# boundary parameters
# collect boundary parameters in a named tuple boundpar...
                  
                                  
bPar = {'BotCond' : 'Robbin',
        'res_rob': 0.005,  # Robin resistance term for bottom
        'H_rob': -1,  # pressure head at lower boundary
        'avgT': 273.15 + 10,
        'rangeT':20,
        'tMin':46,
        'topCond':'Dirichlet',
        'lambdaRobTop':1,
        'lambdaRobBot':0,
        'TBndBot':(273.15 + 10)}
bPar = pd.Series(bPar)
  
# ## Initial Conditions
# Initial Conditions

zRef = -0.75
HIni = zRef - zN
TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K
#HIni = np.zeros(zN.shape)+0.1

# HIni[0:25] =  np.linspace(0.25,0,25)  

# Time Discretization
tOut =  np.logspace(-14, np.log10(365), num=365) 
nOut = np.shape(tOut)[0] 

#### SOLVE ####
def intFun(t, y):
        nf = DivHeatFlux(t, y, sPar, mDim, bPar)
        return nf

    def jacFun(t, y):
        jac = JacHeat(t, y, sPar, mDim, bPar)
        return (jac)


T0 = TIni.copy().squeeze()
TODE = spi.solve_ivp(intFun, [tOut[0], tOut[-1]], T0, method='BDF',
                         t_eval=tOut, vectorized=True, rtol=1e-8, jac=jacFun)
H_ode = fun.IntegrateWF(tOut, HIni, sPar, mDim, bPar)

tOut = H_ode.t

plt.close('all')

fig1, ax1 = plt.subplots(figsize=(7, 4))
for ii in np.arange(0, nN, 20):
    ax1.plot(H_ode.t, H_ode.y[ii, :], '-')
ax1.set_title('Pressure head vs. Time')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Pressure head [m]')
ax1.grid(b=True)


fig2, ax2 = plt.subplots(figsize=(4, 7))
for ii in np.arange(0, nOut, 20):
    ax2.plot(H_ode.y[:, ii], zN[:, 0], '-')


ax2.set_title('Pressure head vs. depth over time')
ax2.set_xlabel('Pressure head [m]')
ax2.set_ylabel('Depth [m]')
ax2.grid(b=True)

thODE = np.zeros(np.shape(H_ode.y))
for ii in np.arange(0, H_ode.t.size, 1):
    hwTmp = H_ode.y[:, ii].reshape(zN.shape)
    thODE[:, ii] = fun.wcont(hwTmp, sPar).reshape(1, nN)

fig3, ax3 = plt.subplots(figsize=(7, 7))
for ii in np.arange(0, H_ode.t.size, 1):
    ax3.plot(thODE[:, ii], zN[:, 0], '-')

# scipy.integrate.squad(thODE[:, 0], 0, -1)
ax3.grid(b=True)
ax3.set_title('Water content vs depth over time')
ax3.set_xlabel('water content [-]')
ax3.set_ylabel('depth [m]')

plt.show()
   
#%% FUNCTIONS WATER FLUX
def Ksat (T,sPar, mDim, H):
    nIN = mDim.nIN # 11 (depth is 1 meter discretised +1)
    # how to interpolate 
    nr,nc = H.shape
    Ksat = np.zeros([nIN, nc], dtype=H.dtype) #set permeability to zero at every node
    
    ii = np.arange(1, nIN-1) # 0-10
    temp = [273.15, 278.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15, 363.15, 373.15]
    mu = [1.787, 1.519, 1.307, 1.002, 0.798, 0.653, 0.547, 0.467, 0.404, 0.355, 0.315, 0.282]

    vis = interpolate.interp1d(temp,mu, kind='linear') 
    Ksat = np.zeros([nIN, nc], dtype=H.dtype) #set permeability to zero at every node
    ii = np.arange(1, nIN-1) # 0-10
    
    Ksat[ii] = (sPar.kapsat*sPar.rho_w*sPar.g)/vis  
    Ksat[ii] =  np.minimum(Kn[ii-1], Kn[ii])
    Ksat[0] = Kn[0]
    Ksat [ii] = (sPar.kapsat*rho_w*sPar.g) / vis(T)

    return Ksat

def Seff(H,sPar, T): 
    #sPar defined later in code: soil parameters
    #in sPar the empirical parameters n, alpha and m are defined
    hc = -H
    Seff = (1+((hc*(hc>0))*sPar.VGa)**sPar.VGn)**(-sPar.VGm)
    return Seff
# hc > 0 to ensure function only returns if Seff(hc)>0


def wcont(H, sPar):
    #function to define the water content from the Seff fraction in the assignment
    #equation rewritten to define vol. water content theta
    #From sPar we use theta_s, theta_r, 
    theta = Seff(H, sPar) * (sPar.theta_s - sPar.theta_r) + sPar.theta_r
    return theta

def Cfun (H, sPar):
    hc = -H
    Se = Seff(H,sPar)
    dSedh = sPar.vGA * sPar.vGm / (1 - sPar.vGM) * Se ** (1 / sPar.vGM) * \
            (1 - Se ** (1 / sPar.vGM)) ** sPar.vGM * (hc > 0) + (hc <= 0) * 0
    return (sPar.theta_s - sPar.theta_r) * dSedh

#this function calculates the complex derivate of the water content with respective to the head
def Chw_I(H, sPar):
    dh = np.sqrt(np.finfo(float).eps)
    if np.iscomplexobj(H):
        hcmplx = H.real + 1j*dh
    else:
        hcmplx = H.real + 1j*dh

    th = wcont(hcmplx, sPar)
    C = th.imag / dh
    return C

def CeffMat(H, sPar, mDim):
    # assume water does not change its density
    theta = wcont(H, sPar)
    S_w = theta / sPar.theta_s
    C_hw = Chw_I(H, sPar)
    beta = 4.5e-10  #compressibility
    rhowat = 1000 #density of water
    g = 9.81
    cv = 1e-8
    S_sw = rhowat * g *(cv + theta * beta) #Storativity
    cPrime = C_hw +S_w * S_sw
    
    #Ponding water condition - case when the water table is higher than GL
    cPrime[mDim.nN-1] = 1/mDim.dzIN[mDim.nIN-2] * (H[mDim.nN-1]>0) \
        + cPrime[mDim.nN-1] * (H[mDim.nN-1]<=0)
        
    return cPrime

def Kvec(H, sPar, mDim, T): 
    #function to fill in effective permeability at nodes
    nIN = mDim.nIN # 11 (depth is 1 meter discretised +1)
    #zIN = mDim.zIN # nodes [-1 0.9 0.8 0.7 ... 0]^T 
    ksat = Ksat (T,sPar, mDim, H)
    nr,nc = H.shape
    
    Sef = Seff(H, sPar)
    Kn = ksat * Sef ** 3.
    
    KIN = np.zeros([nIN, nc], dtype=H.dtype) #set permeability to zero at every node
    
    ii = np.arange(1, nIN-1) # 0-10
    # NOTE: effective permeability = relative permeability * absolute permeability
    KIN[ii] =  np.minimum(Kn[ii-1], Kn[ii])
    KIN[0] = Kn[0]
    KIN[nIN-1]= Kn[nIN-2]
    # Fill KIN on nodes using relation for effective permeability
    return KIN


def BndQTop(t): 
    # function to define fluctuating boundary conditions (water flux) at the top of the soil column
    bndQ = 0
    #bndQ=-0.0001
    #bndQ = (-0.0001)*t*(t>25)*(t<225) #m/day
        #zero flux for the first 25 days followed by 200 days flux 0.001
    return bndQ
    # check of het werkt als t een vector is 


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def WaterFlux(t, H, sPar, mDim, bPar, T):
    nIN = mDim.nIN
    dzN = mDim.dzN
    nr,nc = H.shape
    
    KIN=Kvec(H, sPar, mDim, T)

    q = np.zeros([nIN,nc],dtype=H.dtype)

    # flux at top boundary
    bndQtop = BndQTop(t)
    q[nIN-1] = bndQtop  

    # Flux in all intermediate nodes
    ii = np.arange(1, nIN - 1)
    
    q[ii] = -KIN[ii]*((H[ii] - H[ii - 1])/ dzN[ii - 1]+1) 
    #flux difference between nodes
   
    if bPar.BotCond == 'Gravity':
        #case for which there is only gravity flow
        q[0] = -KIN[0] #infiltration at bottom, head is zero          

    else: 
        # mixed condition
        # bPar.BotCond == 'Robbin'
        q[0] = -bPar.res_rob*(H[0] - bPar.H_rob) 
        
    return q


def DivWaterFlux(t, H, sPar, mDim, bPar, T): # = d(theta)/dt
    nN = mDim.nN
    dzIN = mDim.dzIN
    nr,nc = H.shape

    mass = CeffMat(H, sPar, mDim)
    
    # Calculate water fluxes across all internodes
    qH = WaterFlux(t, H, sPar, mDim, bPar, T)
    divqH = np.zeros([nN, nc]).astype(H.dtype)
    
    # Calculate divergence of flux for all nodes
    ii = np.arange(0, nN)
    divqH[ii] = -(qH[ii + 1] - qH[ii]) \
                   / (dzIN[ii] * mass[ii])

    return divqH

# functions to make the implicit solution easier
# Fill_kMat_water, Fill_mMat_water and Fill_yVec_water


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def FillkMatWater(t, H, sPar, mDim, bPar, T):

    nN = mDim.nN
    dzN = mDim.dzN
    dzIN = mDim.dzIN
    
    K_rob = bPar.res_rob # to define robin conditions??
    KIN = Kvec(H, sPar, mDim, T)

    a = np.zeros(nN, dtype=H.dtype) #dimensions 11 x 1
    b = np.zeros(nN, dtype=H.dtype)
    c = np.zeros(nN, dtype=H.dtype)

    # Fill KMat
    # lower boundary
    if bPar.BotCond == 'Gravity':
        a[0] = 0
        b[0] = -KIN[1,0] / (dzIN[0,0] * dzN[0,0])
        c[0] = KIN[1,0] / (dzIN[0,0] * dzN[0,0])
    else:
        a[0] = 0
        b[0] = (-K_rob/ dzIN[0,0]) - (KIN[1,0] / (dzIN[0,0] * dzN[0,0]))
        c[0] = KIN[1, 0] / (dzIN[0, 0] * dzN[0, 0])
    
    # middle nodes
    ii = np.arange(1, nN - 1)
    a[ii] = KIN[ii, 0] / (dzIN[ii, 0] * dzN[ii - 1, 0])

    b[ii] = -(KIN[ii, 0] / (dzIN[ii, 0] * dzN[ii - 1, 0])
                 + KIN[ii + 1, 0] / (dzIN[ii, 0] * dzN[ii, 0]))

    c[ii] = KIN[ii + 1, 0] / (dzIN[ii, 0] * dzN[ii, 0])
    
    
    # Upper boundary
    ii = nN - 1
    a[ii] = KIN[ii, 0] / (dzIN[ii, 0] * dzN[ii - 1, 0])
    b[ii] = -KIN[ii, 0] / (dzIN[ii, 0] * dzN[ii - 1, 0])
    c[ii] = 0
    
    kMat = np.diag(a[1:nN, 0], -1) + np.diag(b, 0) + np.diag(c[0:nN - 1], 1)
    return kMat


# def FillmMatWater(t, H, sPar, mDim, bPar):
#     nN = mDim.nN
#     dzIN = mDim.dzIN
#     KIN = Kvec(H, sPar, mDim)
#     ii = np.arange(1, nN - 1)
#     d = np.zeros([nN, 1])
#     d[ii, 0] = -(KIN[ii, 0] / dzIN[ii, 0]
#                  + KIN[ii + 1, 0] / dzIN[ii, 0])
#     return d


def FillyVecWater(t, H, sPar, mDim, bPar):
    nN = mDim.nN
    dzIN = mDim.dzIN
    yVec = np.zeros([nN,1])

   # Top Boundary
    yVec[nN] = BndQTop(t)
    KIN = Kvec(H, sPar, mDim)
    qBnd = BndQTop(t)
    
    # Lower Boundary 
    if bPar.BotCond == 'Gravity':
        #case for which there is only gravity flow
        yVec[0,0] = -KIN[0,0]/ dzIN[0, 0] + KIN[1, 0] / dzIN[0, 0] #infiltration at bottom, head is zero          

    else: 
        # mixed condition
        # bPar.BotCond == 'Robbin'
        yVec[0,0] = bPar.res_rob / dzIN[0, 0] * bPar.H_rob  + KIN[1, 0] / dzIN[0, 0]
        
        
        # middel nodes
    ii = np.arange(1, nN - 1)
    yVec[ii, 0] = -KIN[ii, 0] / dzIN[ii, 0] + KIN[ii + 1, 0] / dzIN[ii, 0]
    # Upper boundary
    ii = nN - 1
    yVec[ii, 0] = -KIN[ii, 0] / dzIN[ii, 0] - qBnd / dzIN[ii, 0]
    return yVec


def JacWater(t, H, sPar, mDim, bPar, T):
    # Function calculates the jacobian matrix for the Richards equation
    nN = mDim.nN
    
    kMat = FillkMatWater(t, H, sPar, mDim, bPar, T)
    M_Mat = CeffMat(H, sPar, mDim)

    # massMD = np.diag(FillmMatWater(t, H, sPar, mDim, bPar)).copy()
    jac = np.zeros((3, nN)) #3x11
    a = np.diag(kMat, -1) / M_Mat[1:nN,0]
    b = np.diag(kMat, 0) / M_Mat[0:nN,0]
    c = np.diag(kMat, 1) / M_Mat[0:nN-1,0]

    jac = np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)
    return jac

def IntegrateWF(tRange, iniSt, sPar, mDim, bPar, T):

    def dYdt(t, H):

        if len(H.shape)==1:
            H = H.reshape(mDim.nN,1)
        rates = DivWaterFlux(t, H, sPar, mDim, bPar)

        return rates

    def jacFun(t,y):
        if len(y.shape)==1:
            y = y.reshape(mDim.nN,1)

        nr, nc = y.shape
        dh = np.sqrt(np.finfo(float).eps)
        jac = np.zeros((nr,nr))
        for ii in np.arange(nr):
            ycmplx = y.copy().astype(complex)
            ycmplx[ii] = ycmplx[ii] + 1j*dh
            dfdy = dYdt(t, ycmplx).imag/dh
            jac[:,ii] = dfdy.squeeze()
        return jac
    
    def jacFunMat(t,y):
        if len(y.shape)==1:
            y = y.reshape(mDim.nN,1)
        jac = JacWater(t, y, sPar, mDim, bPar)
        return jac    

    # solve rate equatio
    t_span = [tRange[0],tRange[-1]]
    int_result = spi.solve_ivp(dYdt, t_span, iniSt.squeeze(),
                               method='BDF', vectorized=True #, jac=jacFun, 
                               ,t_eval=tRange,
                               rtol=1e-6)

    return int_result

#%% FUNCTIONS HEAT
def BndTTop(t, bPar):
    bndT = bPar.avgT - bPar.rangeT * np.cos(2 * np.pi
                                            * (t - bPar.tMin) / 365.25)
    return bndT


def HeatFlux(t, T, sPar, mDim, bPar, #H):
    nIN = mDim.nIN
    nN = mDim.nN
    dzN = mDim.dzN
    lambdaIN = sPar.lambdaIN # should depend on water content

    q = np.zeros((nIN, 1))

    # Temperature at top boundary
    bndT = BndTTop(t, bPar)
    # Implement Dirichlet Boundary  Python can mutate a list because these are
    # passed by reference and not copied to local scope...
    # locT = np.ones(np.shape(T)) * T
    locT = T.copy()
    if bPar.topCond.lower() == 'Gravity':
        locT[nN - 1, 0] = bndT

    # Calculate heat flux in domain
    # Bottom layer Robin condition
    q[0, 0] = 0.0
    q[0, 0] = -bPar.lambdaRobBot * (T[0, 0] - bPar.TBndBot)

    # Flux in all intermediate nodes
    ii = np.arange(1, nIN - 1)
    q[ii, 0] = -lambdaIN[ii, 0] * ((locT[ii, 0] - locT[ii - 1, 0])
                                   / dzN[ii - 1, 0])
    # Top layer
    if bPar.topCond.lower() == 'Gravity':
        # Temperature is forced, so we ensure that divergence of flux in top
        # layeris zero...
        q[nIN-1, 0] = q[nIN - 2, 0]
    else:
        # Robin condition
        q[nIN-1, 0] = -bPar.lambdaRobTop * (bndT - T[nN-1, 0])

    return q


def DivHeatFlux(t, T, sPar, mDim, bPar):
    nN = mDim.nN
    dzIN = mDim.dzIN
    locT = T.copy()
    zetaBN = np.diag(FillmMatHeat(t, locT, sPar, mDim, bPar ))

    # Calculate heat fluxes accross all internodes
    qH = HeatFlux(t, locT, sPar, mDim, bPar)

    divqH = np.zeros([nN, 1])
    # Calculate divergence of flux for all nodes
    ii = np.arange(0, nN-1)
    divqH[ii, 0] = -(qH[ii + 1, 0] - qH[ii, 0]) \
                   / (dzIN[ii, 0] * zetaBN[ii])

    # Top condition is special
    ii = nN-1
    if bPar.topCond.lower() == 'dirichlet':
        divqH[ii,0] = 0
    else:
        divqH[ii, 0] = -(qH[ii + 1, 0] - qH[ii, 0]) \
                       / (dzIN[ii, 0] * zetaBN[ii])


    divqHRet = divqH # .reshape(T.shape)
    return divqHRet


# ## functions to make the implicit solution easier
# 
# In order to facilitate an easy implementation of the implicit (matrix) solution I implemented three functions:
# Fill_kMat_Heat, Fill_mMat_Heat and Fill_yVec_Heat.


def FillkMatHeat(t, T, sPar, mDim, bPar):
    lambdaIN = sPar.lambdaIN
    zetaBN = sPar.zetaBN

    nN = mDim.nN
    nIN = mDim.nIN
    dzN = mDim.dzN
    dzIN = mDim.dzIN

    lambdaRobTop = bPar.lambdaRobTop
    lambdaRobBot = bPar.lambdaRobBot

    a = np.zeros([nN, 1])
    b = np.zeros([nN, 1])
    c = np.zeros([nN, 1])

    # Fill KMat
    # lower boundary
    # Robin Boundary condition

    a[0, 0] = 0
    b[0, 0] = -(lambdaRobBot / dzIN[0, 0] + lambdaIN[1, 0] / (
                dzIN[0, 0] * dzN[0, 0]))
    c[0, 0] = lambdaIN[1, 0] / (dzIN[0, 0] * dzN[0, 0])

    # middel nodes
    ii = np.arange(1, nN - 1)
    a[ii, 0] = lambdaIN[ii, 0] / (dzIN[ii, 0] * dzN[ii - 1, 0])

    b[ii, 0] = -(lambdaIN[ii, 0] / (dzIN[ii, 0] * dzN[ii - 1, 0])
                 + lambdaIN[ii + 1, 0] / (dzIN[ii, 0] * dzN[ii, 0]))

    c[ii, 0] = lambdaIN[ii + 1, 0] / (dzIN[ii, 0] * dzN[ii, 0])

    # Top boundary
    if bPar.topCond.lower() == 'dirichlet':
        a[nN-1, 0] = 0
        b[nN-1, 0] = -1
        c[nN-1, 0] = 0
    else:
        # Robin condition
        a[nN-1,0] = lambdaIN[nIN-2, 0] / (dzIN[nIN-2, 0] * dzN[nN-2, 0])
        b[nN-1, 0] = -(lambdaIN[nIN-2, 0] / (dzIN[nIN-2, 0] * dzN[nN-2, 0])
                       + lambdaRobTop / dzIN[nIN-2, 0])
        c[nN-1, 0] = 0

    kMat = np.diag(a[1:nN, 0], -1) + np.diag(b[0:nN, 0], 0) + np.diag(c[0:nN - 1, 0], 1)
    return kMat


def FillmMatHeat(t, T, sPar, mDim, bPar):
    zetaBN = sPar.zetaBN
    if bPar.topCond.lower() == 'dirichlet':
        zetaBN[mDim.nN - 1, 0] = 0
    mMat = np.diag(zetaBN.squeeze(), 0)
    return mMat


def FillyVecHeat(t, T, sPar, mDim, bPar):
    nN = mDim.nN

    yVec = np.zeros([nN, 1])

    # Bottom Boundary
    yVec[0, 0] = bPar.lambdaRobBot / mDim.dzIN[0,0] * bPar.TBndBot

    # Top Boundary (Known temperature)
    if bPar.topCond.lower() == 'dirichlet':
        yVec[nN-1, 0] = BndTTop(t, bPar)
    else:
        # Robin condition
        yVec[nN-1, 0] = bPar.lambdaRobTop / mDim.dzIN[mDim.nIN-2,0] * BndTTop(t,bPar)

    return yVec


def JacHeat(t, T, sPar, mDim, bPar):
    # Function calculates the jacobian matrix for the Richards equation
    nN = mDim.nN
    locT = T.copy().reshape(nN, 1)
    kMat = FillkMatHeat(t, locT, sPar, mDim, bPar)
    massMD = np.diag(FillmMatHeat(t, locT, sPar, mDim, bPar)).copy()

    a = np.diag(kMat, -1).copy()
    b = np.diag(kMat, 0).copy()
    c = np.diag(kMat, 1).copy()

    if bPar.topCond.lower() == 'dirichlet':
        # massMD(nN-1,1) = 0 so we cannot divide by massMD but we know that the
        # Jacobian should be zero so we set b[nN-1,0] to zero instead and
        # massMD[nN-1,0] to 1.
        b[nN-1] = 0
        massMD[nN-1] = 1

    jac = np.zeros((3, nN))
    a = a / massMD[1:nN]
    b = b / massMD[0:nN]
    c = c / massMD[0:nN - 1]
    # jac[0,0:nN-1] = a[:]
    # jac[1,0:nN] = b[:]
    # jac[2,0:nN-1] = c[:]
    jac = np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)
    return jac