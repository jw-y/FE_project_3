import math
from scipy.stats import norm
import pandas as pd
import numpy as np
from datetime import date, datetime
import os

today = date(2021, 6, 4)
S = 4229.89 #S&P500 2021-06-04 close

#today = date(2021, 5, 14)
#expiration_date = date(2021, 6, 10)
#t_delta = expiration_date - today

def BS_call(S, K, T, sigma, r):
    N = norm.cdf
    d1 = (math.log(S/K) + (r+ sigma**2/ 2)*T) / (sigma * math.sqrt(T))
    d2 = (math.log(S/K) + (r- sigma**2/ 2)*T) / (sigma * math.sqrt(T))
    return S*N(d1) - K* math.exp(-r*T)*N(d2)

def BS_put(S, K, T, sigma, r):
    N = norm.cdf
    d1 = (math.log(S/K) + (r+ sigma**2/ 2)*T) / (sigma * math.sqrt(T))
    d2 = (math.log(S/K) + (r- sigma**2/ 2)*T) / (sigma * math.sqrt(T))
    return -S*N(-d1) + K*math.exp(-r*T)*N(-d2)

def cal_moneyness(S, K, T, sigma, r):
    return (math.log(S/K)+r*T)/(sigma*math.sqrt(T))

rt_255 = math.sqrt(255)
def annualize_std(std):
    return std * rt_255

rf_rate = pd.read_hdf('./data/TBILL_d210606.h5').iloc[-1].item()/100
def get_risk_free():
    return rf_rate

def cal_tau(today, expiration_date, PM_settle=True):
    if PM_settle:
        return (expiration_date-today).days / 365
    else:
        return ((expiration_date-today).days-1) / 365

def call_imvol(premium, S, K, T, r, opt='C'):
    if opt == 'C': BS = BS_call
    elif opt == 'P': BS = BS_put
    else: raise Exception("opt wrong")
    
    epsilon = bot = 0.000001
    top = 1.0
    diff = BS(S, K, T, top, r) - premium
    while diff <= 0:
        top += 1.0
        diff = BS(S, K, T, top, r) - premium

    MAX = 1000
    while MAX:
        h = (top+bot)/2
        diff = BS(S, K, T, h, r) - premium
        if abs(diff) <= epsilon:
            return h
        if diff < 0: bot = h
        else: top = h
        MAX-=1
    return None

def deCasteljau(i, j, u, D):
    if D[i][j]:
        return D[i][j]
    p0 = deCasteljau(i-1, j, u, D)
    p1 = deCasteljau(i-1, j+1, u, D)
    D[i][j] = ((1-u)*p0[0]+u*p1[0], (1-u)*p0[1]+u*p1[1])
    return D[i][j]

def bezier_curve(P, num_points= 100):
    points = []
    n = len(P)-1
    for u in np.linspace(0, 1, num_points):
        D = [[None]*(n+1) for _ in range(n+1)]
        for j in range(n+1):
            D[0][j] = P[j]
        points.append(deCasteljau(n, 0, u, D))
    return points

def deBoor(k, x, t, c, p=3):
    """returns S(x).
    
    Arguments
    ---------
    k: Index of knot interval that contains x. k>=p.
    x: Position
    t: Array of knot positions, needs to be padded as decribed above
    c: Array of control points.
    p: Degree of B-spline
    """
    assert(k>=p)
    d = [c[j+k-p] for j in range(0, p+1)]
    
    for r in range(1, p+1):
        for j in range(p, r-1, -1):
            alpha = (x-t[j+k-p]) / (t[j+1+k-r] - t[j+k-p])
            d[j] = ((1.0-alpha)*d[j-1][0] + alpha*d[j][0], 
                    (1.0-alpha)*d[j-1][1] + alpha*d[j][1])
    return d[p]

def B_spline(P, clamped=True, num_points=30):
    """
    cubic B-spline curve
    """
    points = []
    n_P = len(P)
    if clamped:
        knots = [0.0]*3 + np.linspace(0, 1, n_P-2).tolist() + [1.0]*3
    else:
        knots = np.linspace(0, 1, n_P+4).tolist()

    for i in range(3, len(knots)-4):
        for u in np.linspace(knots[i], knots[i+1], num_points):
            points.append(deBoor(i, u, knots, P))
    return points

def deBoor_surface(u_k, v_k, u, v, u_t, v_t, C, p=3):
    def deBoor(k, x, t, c, p=3):
        d = [c[j+k-p] for j in range(0, p+1)]
        for r in range(1, p+1):
            for j in range(p, r-1, -1):
                alpha = (x-t[j+k-p]) / (t[j+1+k-r] - t[j+k-p])
                d[j] = ((1.0-alpha)*d[j-1][0] + alpha*d[j][0],
                        (1.0-alpha)*d[j-1][1] + alpha*d[j][1],
                        (1.0-alpha)*d[j-1][2] + alpha*d[j][2])
        return d[p]
    q = []
    for i in range(u_k-3, u_k+1):
        q.append(deBoor(v_k, v, v_t, C[i]))
    for r in range(1, p+1):
        for j in range(p, r-1, -1):
            alpha = (u-u_t[j+u_k-p]) / (u_t[j+1+u_k-r]-u_t[j+u_k-p])
            q[j] = ((1.0-alpha)*q[j-1][0] + alpha*q[j][0],
                    (1.0-alpha)*q[j-1][1] + alpha*q[j][1],
                    (1.0-alpha)*q[j-1][2] + alpha*q[j][2])
    return q[p]

def B_spline_surface(P, num_points=20):
    """
    cubic B-spline surface
    """
    #u_m, v_n = len(P), len(P[0])
    u_m = len(P)
    u_knots = [0.0]*3+ np.linspace(0, 1, u_m-2).tolist() + [1.0]*3
    #v_knots = [0.0]*3+ np.linspace(0, 1, v_n-2).tolist() + [1.0]*3

    grid = []
    for i in range(3, len(u_knots)-4):
        tmp_p = None
        v_n = len(P[i])
        v_knots = [0.0]*3+ np.linspace(0, 1, v_n-2).tolist() + [1.0]*3

        for j in range(3, len(v_knots)-4):
            tmp_u = []
            for u in np.linspace(u_knots[i], u_knots[i+1], num_points):
                tmp_v = []
                for v in np.linspace(v_knots[j], v_knots[j+1], num_points):
                    tmp_v.append(deBoor_surface(i, j, u, v, u_knots, v_knots, P))
                tmp_u.append(tmp_v)

            if not tmp_p:
                tmp_p = tmp_u
            else:
                for a in range(len(tmp_u)):
                    tmp_p[a] += tmp_u[a]
        grid += tmp_p    
    return grid

def get_imvol_list(expir, df, moneyness=True, inc_tau=False, imvol=True):
    print("calculating calls expiring", expir)
    imvol_list = []
    for idx, row in df.iterrows():
        contractSym = row.contractSymbol
        lastTradeDate = row.lastTradeDate
        premium = row.lastPrice
        strike = row.strike
        if lastTradeDate.year!=2021 or lastTradeDate.month!=6 or lastTradeDate.day!=4:
            continue

        if contractSym[3]=='W':
            tau = cal_tau(today, expir)
            maturity = float((expir-today).days)
        else:
            tau = cal_tau(today, expir, PM_settle=False)
            maturity = float((expir-today).days -1)

        if imvol:
            x = call_imvol(premium, S, strike, tau, rf_rate)
        else:
            x = premium
        if x:
            if moneyness:
                m = cal_moneyness(S, strike, tau, x, rf_rate)
                if inc_tau:
                    imvol_list.append((m, tau, x))
                else:
                    imvol_list.append((m, x))
            else:
                if inc_tau:
                    imvol_list.append((strike, tau, x))
                else:
                    imvol_list.append((strike, x))

    return sorted(imvol_list)

def get3Dpoints():
    points = []
    for filename in os.listdir('./data/calls'):
        df = pd.read_hdf('./data/calls/'+filename)
        expir = datetime.strptime(filename[6:12], "%y%m%d").date()
        pp = get_imvol_list(expir=expir, df = df, inc_tau=True)
        if len(pp) >= 4:
            points.append(pp)
    return points






