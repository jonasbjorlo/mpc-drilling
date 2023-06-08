import numpy as np
from scipy.optimize import least_squares

def J12(Nr, W, a, b, c):
    return -(101*3**(1/2)*a*np.sin(np.pi/18)*(3969*np.pi*np.cos(np.pi/18)*(np.pi - np.arccos((10*W - 3969*np.cos(np.pi/18))/(3969*np.cos(np.pi/18)))) - (5**(1/2)*W*(- 5*W**2 + 3969*np.pi*np.cos(np.pi/18)*W)**(1/2))/(630*np.pi*np.cos(np.pi/18)))*(b - 1))/(140000*Nr**b*W**c*np.pi*np.cos(np.pi/18))

def J11(Nr, W, a, b, c):
    return (101*15**(1/2)*Nr**(1 - b)*a*np.sin(np.pi/18)*(20*W**2*(- 5*W**2 + 3969*np.cos(np.pi/18)*W)**(1/2) + 5000940*np.pi**2*np.cos(np.pi/18)**2*(- 5*W**2 + 3969*np.pi*np.cos(np.pi/18)*W)**(1/2) - 11907*W*np.pi*np.cos(np.pi/18)*(- 5*W**2 + 3969*np.cos(np.pi/18)*W)**(1/2)))/(176400000*W**c*np.pi**2*np.cos(np.pi/18)**2*(- 5*W**2 + 3969*np.cos(np.pi/18)*W)**(1/2)*(- 5*W**2 + 3969*np.pi*np.cos(np.pi/18)*W)**(1/2)) - (101*3**(1/2)*Nr**(1 - b)*a*c*np.sin(np.pi/18)*(3969*np.pi*np.cos(np.pi/18)*(np.pi - np.arccos((10*W - 3969*np.cos(np.pi/18))/(3969*np.cos(np.pi/18)))) - (5**(1/2)*W*(- 5*W**2 + 3969*np.pi*np.cos(np.pi/18)*W)**(1/2))/(630*np.pi*np.cos(np.pi/18))))/(140000*W**(c + 1)*np.pi*np.cos(np.pi/18))

def get_linearized_model(dt, rop_est, a, b, c, W, Nr):
    J11_ev = J11(Nr, W, a, b, c)
    J12_ev = J12(Nr, W, a, b, c)

    At = np.array([[1, dt],[0, 1]])
    Bt = np.array([[dt*rop_est*J11_ev, dt*rop_est*J12_ev],[0,0]])
    Ct = np.array([[0, 1]])
    Dt = np.array([[rop_est*J11_ev, rop_est*J12_ev]])
    return At, Bt, Ct, Dt

def hareland(Nr, W, a, b, c):
    COR = a / (np.power(Nr,b) * np.power(W,c))
    first = COR * (14.14 * Nc * Nr / Db) * np.cos(alpha) * np.sin(theta)
    
    acosarg = 1 - (4*W/(np.cos(theta)*Nc*dc**2*sigma_c))
    
    second = np.power(dc/2, 2) * np.arccos(acosarg)
    
    sqrtarg = (2*W /(np.cos(theta) * np.pi * Nc * sigma_c)) - ((4*np.power(W, 2)) / (np.power(np.cos(theta) * np.pi *Nc*dc*sigma_c, 2)))
    
    third = np.sqrt(sqrtarg)*(W / (np.cos(theta)*np.pi *Nc*sigma_c))
    
    return first * (second-third)

def drilling_scipy_opt(ROP, Nr, W, a_guess, b_guess, c_guess):
    """scipy version"""
    def residual(coeff, Nr, W, ROP):
        a = coeff[0]
        b = coeff[1]
        c = coeff[2]
        return (hareland(Nr, W, a, b, c) - ROP)
    
    bnds = ([0.001, 0.5, 0.5], [10000, 1, 1.5])
    
    try:
        sol = least_squares(residual, [a_guess, b_guess, c_guess], args=(Nr, W, ROP), bounds=bnds)
    except ValueError as e:
        if len(e.args) > 0 and e.args[0] == 'Residuals are not finite in the initial point.':
            print(a_guess, b_guess, c_guess)
            print(Nr, W, ROP)
        else:
            raise e
    
    return sol.x[0], sol.x[1], sol.x[2]

def partition_drillingdata(i, L, y, Nr, W, Q):
    """Partition dataset. i=index, L=window size"""
    if i <= L:
        si = 0
    else:
        si = i-L
    
    y_win = y[si:i+1]
    Nr_win = Nr[si:i+1]
    W_win = W[si:i+1]
    Q_win = Q[si:i+1]
    
    return y_win, Nr_win, W_win, Q_win

def partition_responsedata(i, L, wave):
    """Partition wavedata, predicted response. i=index, L=window size"""
    if i <= L:
        si = 0
    else:
        si = i-L
    
    resp_win = wave[si:i+1]
    
    return resp_win