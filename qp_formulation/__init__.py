import cvxopt
import numpy as np

def qp(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """qp wrapper for cvxopt to include lower and upper bounds in the inequality constraints.
       Inspired by: https://github.com/nolfwin/cvxopt_quadprog
       Accessed: 12.03-23
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)

    return np.array(sol['x'])

# Matrix Formulation Implementation
# Inspired by Kommedal (2021).
def P_m(At,Ct,P_h):
    n_x = At.shape[0]
    n_y = Ct.shape[0]

    P = np.zeros((P_h*n_y,n_x))
    P[:n_y,:] = Ct
    
    for i in range(P_h):
        P[(i+1)*n_y+1:(i+2)*n_y,:] = np.dot(P[(i)*n_y+1:(i+1)*n_y,:], At)

    return P

def H_m(At,Bt,Ct,Dt,P_h,eta,fac):
    n_y = Ct.shape[0]
    n_u = Bt.shape[0]

    H = np.zeros((P_h*n_y,P_h*n_u))
    for i in range(P_h*n_y):
        eta_i = eta[i] if i < len(eta) else eta[-1]
        if eta_i > 0:
            wCorr = eta_i*fac*-1
        else:
            wCorr = eta_i*fac*-1
        newDt = Dt + np.array([wCorr,0])
        H[i,i*2:i*2+2] = newDt
    
    for i in range(1,P_h*n_y):
        indx = 0
        for j in range(0,P_h*n_u,2):
            if i-1 >= indx:
                eta_j = eta[j] if j < len(eta) else eta[-1]
                if eta_j > 0:
                    wCorr = eta_j*fac*-1
                else:
                    wCorr = eta_j*fac*-1
                etaCorr = np.array([[wCorr,0],[0, 0]])
                newBt = Bt + etaCorr
                newH = np.linalg.multi_dot([Ct, np.power(At, i-1-indx), newBt])
                H[i,j:j+2] = newH
            indx = indx + 1
    return H

def K_m(At,Ct,P_h):
    n_x = At.shape[0]
    n_y = Ct.shape[0]
    
    E = np.zeros((P_h*n_y,n_x))
    E[0:n_y,:] = 0
    E[1,:] = Ct
    
    for i in range(2,P_h*n_y):
        SumOfA = 0
        for j in range(i-1):
            SumOfA = SumOfA + np.power(At, i)
        E[i,:] = Ct + np.dot(Ct, SumOfA)
    return E

def I_m(P_h,n_u):
    kronD = np.kron(np.tril(np.ones(P_h*n_u)),np.eye(n_u))
    Delta = kronD[:P_h*n_u,:P_h*n_u]
    return Delta

def get_opt_drillparams(i, P_h, At, Bt, Ct, Dt, r_y, r_u, gamma, phi, Q, R, x0, u0, ymax, eta, fac, uk_lb, uk_ub, duk_ub, duk_lb, w, W):
    n_x = At.shape[0]
    n_u = Bt.shape[0]
    n_y = Ct.shape[0]

    I = I_m(P_h, n_u)
    K = K_m(At,Ct,P_h)
    H = H_m(At,Bt,Ct,Dt,P_h,eta,fac)
    P = P_m(At,Ct,P_h)

    M = np.dot(K, phi) + np.dot(np.ones((P_h*n_y,1)), gamma) - r_y[i:i+P_h*n_y]
    H_tilde = np.linalg.multi_dot([np.transpose(H), Q, H]) + R
    f = np.linalg.multi_dot([np.transpose(H), np.transpose(Q), np.dot(P, x0) + M]) - np.dot(np.transpose(R), r_u)

    A1 = np.vstack((np.linalg.inv(I),-np.linalg.inv(I)))

    I1 = np.eye(P_h*n_u)
    I2 = np.eye(P_h)
    bh = np.bmat([[duk_ub + np.linalg.solve(I,u0)], [-duk_lb - np.linalg.solve(I,u0)]])
    by = np.ones((P_h*n_y,1))*ymax - np.dot(P,x0) - np.dot(K, phi) - np.dot(np.ones((P_h*n_y,1)), gamma)
    A_in = np.bmat([
                [A1, np.zeros((A1.shape[0], 2*I1.shape[1] + I2.shape[1]))],
                [H, np.zeros((H.shape[0], 2*I1.shape[1])), -I2]])
    b_in = np.vstack((bh,by))
    lb = np.vstack((uk_lb, np.zeros((P_h*n_u,1)), np.zeros((P_h*n_u,1)), np.zeros((P_h,1))))
    ub = np.vstack((uk_ub, np.ones((P_h*n_u,1))*100000, np.ones((P_h*n_u,1))*100000, np.ones((P_h,1))*100000))

    H_QP = np.bmat([
            [H_tilde, np.zeros((H_tilde.shape[0], W.shape[1]))],
            [np.zeros((W.shape[0], H_tilde.shape[1])), W]])


    q_QP = np.vstack((f, w))

    return qp(H_QP, q_QP, L=A_in, k=b_in, lb=lb, ub=ub)