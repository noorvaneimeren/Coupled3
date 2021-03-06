import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as spi 
import scipy.interpolate as spt
import scipy.sparse as sp
from scipy import signal

meteo = pd.read_csv(r'WieringermeerData_Meteo.csv', skiprows=0)
datetime1 = meteo.iloc[:,0]
rainfall = meteo.iloc[:,1] # rainfall rate over the day mm/day


days = 30
t = np.linspace(0, days, 1000)
intensity = 300/365
signal  = intensity*signal.square(t*np.pi)+intensity


# %% FUNCTIONS WATER FLUX

def TempVis(T,sPar):
    
    temp = [273.15, 278.15, 283.15, 293.15, 303.15, 313.15,
            323.15, 333.15, 343.15, 353.15, 363.15, 373.15]
    mu = [1.787, 1.519, 1.307, 1.002, 0.798, 0.653,
          0.547, 0.467, 0.404, 0.355, 0.315, 0.282]
    
    p = spt.interp1d(temp, mu, fill_value='extrapolate')
    
    VisInt = p(T)*rho_w * sPar.kapsat
    
    return VisInt


def Ksat(H, T, sPar, mDim):
    Ksat = sPar.kapsat* TempVis(298.15,sPar)/TempVis(T, sPar)
    return Ksat

# def Ksat(H, T, sPar, mDim):
#     nN = mDim.nN
#     nr, nc = H.shape
#     vis = TempVis(T,sPar)

#     # set permeability to zero at every node
#     Ksat = np.zeros((nN, nc))#, dtype=H.dtype)

#     # temp = [273.15, 278.15, 283.15, 293.15, 303.15, 313.15,
#     #         323.15, 333.15, 343.15, 353.15, 363.15, 373.15]
#     # mu = [1.787, 1.519, 1.307, 1.002, 0.798, 0.653,
#     #       0.547, 0.467, 0.404, 0.355, 0.315, 0.282]

#     # vis = interpolate.UnivariateSpline(temp, mu)

#     ii = np.arange(0, nN)
    
#     Ksat[ii] = (T>=273.15) * (sPar.kapsat*sPar.rho_w*sPar.g)/vis 

#     return Ksat





def Seff(H, sPar):
    # in sPar the empirical parameters n, alpha and m are defined
    hc = -H
    Seff = (1+((hc*(hc > 0))*sPar.VGa)**sPar.VGn)**(-sPar.VGm)
    return Seff
# hc > 0 to ensure function only returns if Seff(hc)>0


def wcont(H, sPar):
    # function to define the water content from the Seff fraction in the assignment
    # equation rewritten to define vol. water content theta
    # From sPar we use theta_s, theta_r,
    theta = Seff(H, sPar) * (sPar.theta_s - sPar.theta_r) + sPar.theta_r
    return theta


def Cfun(H, sPar):
    hc = -H
    Se = Seff(H, sPar)
    dSedh = sPar.vGA * sPar.vGm / (1 - sPar.vGM) * Se ** (1 / sPar.vGM) * \
        (1 - Se ** (1 / sPar.vGM)) ** sPar.vGM * (hc > 0) + (hc <= 0) * 0
    return (sPar.theta_s - sPar.theta_r) * dSedh

# this function calculates the complex derivate of the water content with respective to the head


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
    beta = 4.5e-10  # compressibility
    rho_w = 1000  # density of water
    g = 9.81
    cv = 1e-8
    S_sw = rho_w * g * (cv + theta * beta)  # Storativity
    cPrime = C_hw + S_w * S_sw

    # Ponding water condition - case when the water table is higher than GL
    cPrime[mDim.nN-1] = 1/mDim.dzIN[mDim.nIN-2] * (H[mDim.nN-1] > 0) \
        + cPrime[mDim.nN-1] * (H[mDim.nN-1] <= 0)

    return cPrime


def Kvec(H, T, sPar, mDim):
    # function to fill in effective permeability at nodes
    nIN = mDim.nIN  # 101 (depth is 1 meter discretised +1)
    # zIN = mDim.zIN # nodes [-1 0.9 0.8 0.7 ... 0]^T
    ksat = Ksat(H, T, sPar, mDim)
    nr, nc = H.shape

    Sef = Seff(H, sPar)
    Kn = ksat * Sef ** 3.

    # set permeability to zero at every node
    KIN = np.zeros([nIN, nc], dtype=H.dtype)

    ii = np.arange(1, nIN-1)  # 0-10
    # NOTE: effective permeability = relative permeability * absolute permeability
    KIN[ii] = np.minimum(Kn[ii-1], Kn[ii])
    KIN[0] = Kn[0]
    KIN[nIN-1] = Kn[nIN-2]
    # Fill KIN on nodes using relation for effective permeability
    return KIN


def BndQTop(t):
    # function to define fluctuating boundary conditions (water flux) at the top of the soil column
    bndQ = 0
    # bndQ=-0.0001
    #bndQ = (-0.000)*(t>25)*(t<225) #m/day
    # zero flux for the first 25 days followed by 200 days flux 0.001
    return bndQ
    # check of het werkt als t een vector is


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def WaterFlux(t, H, T, sPar, mDim, bPar):
    nIN = mDim.nIN
    dzN = mDim.dzN
    nr, nc = H.shape

    KIN = Kvec(H, T, sPar, mDim)

    qw = np.zeros((nIN, nc)).astype(H.dtype)

    # flux at top boundary
    bndQtop = BndQTop(t)
    qw[nIN-1] = bndQtop

    # Flux in all intermediate nodes
    ii = np.arange(1, nIN - 1)

    qw[ii] = -KIN[ii]*((H[ii] - H[ii - 1]) / dzN[ii - 1]+1)
    # flux difference between nodes

    if bPar.BotCond == 'Gravity':
        # case for which there is only gravity flow
        qw[0] = -KIN[0]  # infiltration at bottom, head is zero

    else:
        # mixed condition
        # bPar.BotCond == 'Robbin'
        qw[0] = -bPar.res_rob*(H[0] - bPar.H_rob)

    return qw


# def DivWaterFlux(t, H, T, sPar, mDim, bPar):  # = d(theta)/dt
#     nN = mDim.nN
#     dzIN = mDim.dzIN
#     nr, nc = H.shape

#     mass = CeffMat(H, sPar, mDim)

#     # Calculate water fluxes across all internodes
#     qW = WaterFlux(t, H, T, sPar, mDim, bPar)
#     divqW = np.zeros((nN, nc)).astype(H.dtype)

#     # Calculate divergence of flux for all nodes
#     ii = np.arange(0, nN)
#     divqW[ii] = -(qW[ii + 1] - qW[ii]) \
#         / (dzIN[ii] * mass[ii])

#     return divqW



# %% FUNCTIONS HEAT

# define duntion lamda
def LambdaCalc(H, T, sPar):

    Wat = wcont(H, sPar)
    Sat = Wat/n
    lambdaWat = 0.58
    lambdaQuartz = 6.5  # [W/(mK)] thermal conductivity of quartz
    lambdaOther = 2.0  # [W/(mK)] thermal conductivity of other minerals
    lambdaA = (0.0025 + 0.0736*Sat)

    g1 = 0.15+(0.333-0.015)*Sat
    g2 = 1 - (2*g1)
    g = np.array([g1, g1, g2])

    Fw = 1
    Fs1 = 0
    Fa1 = 0

    for i in range(len(g)):
        Fa1 += ((1+(lambdaA/lambdaWat-1)*g[i]))**-1
        Fs1 += ((1+(lambdaQuartz/lambdaWat-1)*g[i]))**-1
    Fa = Fa1*(1/3)
    Fs = Fs1*(1/3)

    LamNom1 = Fs*lambdaQuartz*(1-n)
    LamNom2 = Fw*lambdaWat*n*Sat
    LamNom3 = Fa*lambdaA*n*(1-Sat)
    LamDen1 = Fs*(1-n)
    LamDen2 = Fw*n*Sat
    LamDen3 = Fa*n*(1-Sat)
    Lambda = (LamNom1+LamNom2+LamNom3)/(LamDen1+LamDen2+LamDen3)

    return Lambda


def ZetaBCalc(H, T, sPar, mDim):
    zN = mDim.zN

    Wat = wcont(H, sPar)
    Sat = Wat/n
    zetaB = np.ones(np.shape(zN)) * ((1 - n) *
                                      sPar.zetaSol + n * sPar.zetaWat * Sat)

    return zetaB


def BndTTop(t, bPar):
    bndT = bPar.avgT - bPar.rangeT * np.cos(2 * np.pi
                                            * (t - bPar.tMin) / 365.25)
    return bndT


def HeatFlux(t, H, T, sPar, mDim, bPar):
    nr,nc = T.shape
    nIN = mDim.nIN
    nN = mDim.nN
    dzN = mDim.dzN
    lambdaIN = LambdaCalc(H, T, sPar)
    qW = WaterFlux(t, H, T, sPar, mDim, bPar)
    qh = np.zeros((nIN,nc)).astype(H.dtype)

    # Temperature at top boundary
    bndT = BndTTop(t, bPar)
    
    # Implement Dirichlet Boundary  Python can mutate a list because these are
    # passed by reference and not copied to local scope...
    # locT = np.ones(np.shape(T)) * T
    locT = T.copy()
    if bPar.topCond.lower() == 'Gravity':
        locT[nN - 1] = bndT

    # Calculate heat flux in domain
    # Bottom layer Robin condition
    qh[0] = -lambdaIN[0] * (locT[0] - bPar.TBndBot) + sPar.zetaWat[0]*qW[0]*bPar.TBndBot*(qW[0]>=0)

    # Flux in all intermediate nodes
    ii = np.arange(0, nIN - 1)
    qh[ii] = -lambdaIN[ii] * ((locT[ii] - locT[ii - 1]) / dzN[ii - 1]) + sPar.zetaWat*qW[ii]*locT[ii]
    # Top layer
    if bPar.topCond.lower() == 'Gravity':
        # Temperature is forced, so we ensure that divergence of flux in top
        # layeris zero...
        qh[nIN-1] = q[nIN - 2]
    # else:
    #     # Robin condition
    #     qh[nIN-1] = -bPar.lambdaRobTop * (bndT - T[nN-1])

    return qh


# def DivHeatFlux(t, H, T, sPar, mDim, bPar):
#     nr,nc = T.shape
#     nN = mDim.nN
#     # dzIN = mDim.dzIN
#     zetaBN = sPar.zetaBN #np.diag(FillmMatHeat(t, H, T, sPar, mDim, bPar))

#     # Calculate heat fluxes accross all internodes
#     qH = HeatFlux(t, H, T, sPar, mDim, bPar)
    

#     divqH = np.zeros([nN, nc]).astype(H.dtype)
#     # Calculate divergence of flux for all nodes
#     ii = np.arange(0, nN-1)
#     divqH[ii] = -(qH[ii + 1] - qH[ii]) \
#         / (dzIN[ii] * zetaBN[ii])

#     # Top condition is special
#     ii = nN-1
#     if bPar.topCond.lower() == 'Gravity':
#         divqH[ii] = 0
#     else:
#         divqH[ii] = -(qH[ii + 1] - qH[ii]) \
#             / (dzIN[ii] * zetaBN[ii])

#     return divqH

def Divtotal (t, H, T, sPar, mDim, bPar):
    nr, nc = H.shape
    nN = mDim.nN
    zetaBN = np.diag(np.diag(FillmMatHeat(t, H, T, sPar, mDim, bPar))) # weird solution, reshape did not work
    
    zetaWat = sPar.zetaWat
    zetaAir = sPar.zetaAir
    rHSH = np.zeros([nN,nc], dtype=H.dtype)
    rHSW = np.zeros([nN,nc], dtype=H.dtype)
    Divtot = np.zeros([2*nN,nc],dtype=H.dtype)
    
    divqW =  np.zeros([nN,nc], dtype=H.dtype)
    divqH =  np.zeros([nN,nc], dtype=H.dtype)
    
    mWat = CeffMat(H, sPar,mDim)
    qW = WaterFlux(t, H, T, sPar, mDim, bPar)
    
    ii = np.arange(0, nN)
    divqW[ii] = -(qW[ii+1]-qW[ii]) / (dzIN[ii]) 
    rHSW = divqW / mWat
    
    qH = HeatFlux(t, H, T, sPar, mDim, bPar)
    divqH[ii] = -(qH[ii+1]-qH[ii]) / (dzIN[ii])
    # rHSH = (divqH - (T*(zetaWat-zetaAir)*divqW)) / (zetaBN)
    
    Divtot = np.concatenate((rHSW, rHSH), axis=0)
    return Divtot


# ## functions to make the implicit solution easier
#
# In order to facilitate an easy implementation of the implicit (matrix) solution I implemented three functions:
# Fill_kMat_Heat, Fill_mMat_Heat and Fill_yVec_Heat.




def FillmMatHeat(t, H, T, sPar, mDim, bPar):
    zetaBN = ZetaBCalc(H, T, sPar, mDim)
    if bPar.topCond.lower() == 'Gravity':
        zetaBN[mDim.nN - 1] = 0
    mMat = np.diag(zetaBN.squeeze(), 0)
    return mMat





# %% MAIN

# Then we start running our model.
# First we require the domain discretization
# Domain
nIN = 101  # 1 meter depth
zIN = np.linspace(-1.0, 0, num=nIN).reshape(nIN, 1)  # Internodes
# nIN = np.shape(zIN)[0]
zN = np.zeros(nIN - 1).reshape(nIN - 1, 1)  # Nodes
zN[0, 0] = zIN[0, 0]  # both nodes and internodes start at -1
zN[1:nIN - 2, 0] = (zIN[1:nIN - 2, 0] + zIN[2:nIN - 1, 0]) / 2
zN[nIN - 2, 0] = zIN[nIN - 1]  # both nodes and internodes end at 0
nN = np.shape(zN)[0]  # number of nodes (10)

ii = np.arange(0, nN - 1)  # [0 1 2 3 4 5 6 7 8]
# [0.25 0.1 0.1 0.1 0.1 0.1 0.1 0.15]
dzN = (zN[ii + 1, 0] - zN[ii, 0]).reshape(nN - 1, 1)
dzIN = (zIN[1:, 0] - zIN[0:-1, 0]).reshape(nIN - 1, 1)  # [0.1 0.1 0.1 ...]

# collect model dimensions in a namedtuple: modDim
mDim = {'zN': zN,
        'zIN': zIN,
        'dzN': dzN,
        'dzIN': dzIN,
        'nN': nN,
        'nIN': nIN
        }
mDim = pd.Series(mDim)

rho_w = 1000  # [kg/m3] density of water
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

#define interface height of 2 soils

sheight = 0.95



#SAND
sVGa = 2
sVGn = 3
sVGm = 1- 1/3
sTheta_s = 0.4
sTheta_r =  0.01
sKapsat = 1
sZetasol = 2.235*10**6
sZetaWat =  4.154 *10**6

#CLAY
cVGa = 2
cVGn = 1.23
cVGm = 1 - 1/1.23
cTheta_s = 0.49
cTheta_r = 0.01
cKapsat =1
cZetasol = 3.235*10**6
cZetaWat = 5.235*10**6

VGa = np.ones(np.shape(zN))
VGn = np.ones(np.shape(zN))
VGm = np.ones(np.shape(zN))
theta_s = np.ones(np.shape(zN))
theta_r = np.ones(np.shape(zN))
kapsat = np.ones(np.shape(zN))
zetaSol = np.ones(np.shape(zN))
zetaWat = np.ones(np.shape(zN))

for i in range (len(zN)):
    if zN[i]<-sheight:
        VGa[i] = sVGa
        VGn[i] = sVGn
        VGm[i] = sVGm
        theta_s[i] = sTheta_s
        theta_r[i] = sTheta_r
        kapsat[i] = sKapsat
        zetaSol[i] = sZetasol
        zetaWat[i] = sZetaWat
    else:
        VGa[i] = cVGa
        VGn[i] = cVGn
        VGm[i] = cVGm
        theta_s[i] = cTheta_s
        theta_r[i] = cTheta_r
        kapsat[i] = cKapsat
        zetaSol[i] = cZetasol
        zetaWat[i] = cZetaWat

sPar = {'VGa': VGa,
        'VGn': VGn,
        'VGm': VGm,  # m = 1-1/n[-]
        'theta_s': theta_s,  # saturated water content
        'theta_r': theta_r,  # residual water content
        'kapsat': kapsat,
        'zetaSol': zetaSol,
        'zetaWat': zetaWat,  # at 35C
        'Ksat': np.ones(np.shape(zN)) * 0.05,
        'vis' : np.ones(np.shape(zN)) * 0.8,
        'g': np.ones(np.shape(zN)) * 9.81,
        'zetaBN': np.ones(np.shape(zN)) * ((1 - n) * zetaSol + n * zetaWat),
        'lambdaIN': np.ones(np.shape(zIN)) * lambdaBulk * (24 * 3600),
        'rho_w': np.ones(np.shape(zN)) * 1000,
        'zetaAir': np.ones(np.shape(zN)) * 0.024}

sPar = pd.Series(sPar)


# ## Definition of the Boundary Parameters
# boundary parameters
# collect boundary parameters in a named tuple boundpar...


bPar = {'BotCond': 'Robbin',
        'res_rob': 0.005,  # Robin resistance term for bottom
        'H_rob': -1,  # pressure head at lower boundary
        'avgT': 273.15 + 10,
        'rangeT': 20,
        'tMin': 46,
        'topCond': 'Dirichlet',
        'lambdaRobTop': 1,
        'lambdaRobBot': 0,
        'TBndBot': (273.15 + 10)}
bPar = pd.Series(bPar)

# ## Initial Conditions
# Initial Conditions

zRef = -0.5
HIni = zRef - zN
TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K
HIni = np.zeros(zN.shape)+0.1



# Time Discretization
tOut = np.logspace(-14, np.log10(365), num=365)
nOut = np.shape(tOut)[0]

#### SOLVE ####


def intFun(t, y):
    H = y[:nN]
    T = y[nN:]

    # nf = DivWaterFlux(t, H, T, sPar, mDim, bPar)
    # nq = DivHeatFlux(t, H, T, sPar, mDim, bPar)

    # nqf = np.concatenate([nf,nq])
    return Divtotal(t, H, T, sPar, mDim, bPar)

def jacFun(t, y): 
 
    if len(y.shape) == 1:
        y = y.reshape(2*mDim.nN, 1)

    nr, nc = y.shape
    dh = np.sqrt(np.finfo(float).eps)
    
    ycmplx = np.repeat(y, nr, axis=1).astype(complex)
    c_ex = np.eye(nr)*1j*dh
    ycmplx = ycmplx + c_ex
    dfdy = intFun(t, ycmplx).imag/dh
    return sp.coo_matrix(dfdy)
    
T0 = TIni.copy().squeeze()
hw0 = HIni.copy().squeeze()
y0 = np.concatenate((hw0,T0))
# use v_stack --> stack H0 and T0 --> solve for the whole vector once
# jacobian will speed up our simulation

TODE = spi.solve_ivp(intFun, [tOut[0], tOut[-1]], y0, method='BDF',
                     t_eval=tOut, vectorized=True, rtol=1e-5, jac=jacFun)

print('solved?')

#%%


fig1, ax1 = plt.subplots(figsize=(7, 4))
for ii in np.arange(0, nN, 20):
    ax1.plot(TODE.t, TODE.y[ii, :], '-')
ax1.set_title('Pressure head vs. Time')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Pressure head [m]')
ax1.grid(b=True)


fig2, ax2 = plt.subplots(figsize=(4, 7))
for ii in np.arange(0, nOut, 20):
    ax2.plot(TODE.y[:nN, ii], zN[:nN, 0], '-')


ax2.set_title('Pressure head vs. depth over time')
ax2.set_xlabel('Pressure head [m]')
ax2.set_ylabel('Depth [m]')
ax2.grid(b=True)

thODE = np.zeros(np.shape(TODE.y))
for ii in np.arange(0, TODE.t.size, 1):
    hwTmp = TODE.y[:nN, ii].reshape(zN.shape)
    thODE[:nN, ii] = wcont(hwTmp, sPar).reshape(1, nN)

fig3, ax3 = plt.subplots(figsize=(7, 7))
for ii in np.arange(0, TODE.t.size, 1):
    ax3.plot(thODE[:nN, ii], zN[:, 0], '-')

ax3.grid(b=True)
ax3.set_title('Water content vs depth over time')
ax3.set_xlabel('water content [-]')
ax3.set_ylabel('depth [m]')
ax3.grid(b=True)


fig4, ax4 = plt.subplots(figsize=(7, 4))
for ii in np.arange(nN,2*nN, 20):
    ax4.plot(tOut, TODE.y[ii, :], '-')
ax4.set_title('Temperature (ODE)')
ax4.grid(b=True)

fig5, ax5 = plt.subplots(figsize=(4, 7))
for ii in np.arange(0, nOut, 20):
    ax5.plot(TODE.y[nN:, ii], zN, '-')

ax5.set_title('Temperature vs. depth (ODE)')
ax5.grid(b=True)

plt.show()

#%%

print(TODE.y)

plt.plot(TODE.y[nN:,:])