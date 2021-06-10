import math
from scipy.stats import norm
import pandas as pd
import numpy as np
from datetime import date, datetime
import os
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib import cm

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

def BS_call_np(S, K, T, sigma, r):
    N = norm.cdf
    d1 = (np.log(S/K)) + (r+ sigma**2/2)*T / (sigma*np.sqrt(T))
    d2 = (np.log(S/K)) + (r- sigma**2/2)*T / (sigma*np.sqrt(T))
    return S*N(d1) - K*np.exp(-r*T)*N(d2)

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

def cal_imvol_np(premium, S, K, T, r):
    if isinstance(premium, np.ndarray):
        bot, top = np.zeros(premium.shape[0]), 3*np.ones(premium.shape[0])
    else:
        bot, top = 0, 3
    h = (bot+top)/2
    for _ in range(40):
        diff = BS_call_np(S, K, T, h, r) - premium
        bot = np.where(diff<0, h, bot)
        top = np.where(diff>=0, h, top)
        h = (bot+top)/2
    return h

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

        tau = cal_tau(today, expir)
        maturity = float((expir-today).days)
        '''
        if contractSym[3]=='W':
            tau = cal_tau(today, expir)
            maturity = float((expir-today).days)
        else:
            tau = cal_tau(today, expir, PM_settle=False)
            maturity = float((expir-today).days -1)
        '''

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
    for filename in sorted(os.listdir('./data/calls')):
        df = pd.read_hdf('./data/calls/'+filename)
        expir = datetime.strptime(filename[6:12], "%y%m%d").date()
        pp = get_imvol_list(expir=expir, df = df, moneyness=False, inc_tau=True, imvol=False)
        if len(pp) >= 4:
            points.append(pp)
    return points

def b_spline_base(P, x):
    """P: control points"""
    delta = np.mean(np.diff(P))
    n_P = len(P)
    t = [P[0]-delta*3, P[0]-delta*2, P[0]-delta*1]+\
            np.linspace(P[0],P[-1], n_P-2).tolist()+\
            [P[-1]+delta*1, P[-1]+delta*2, P[-1]+delta*3]
    B = [[0.0]*(4) for _ in range(n_P+4)]

    for i in range(len(t)):
        if t[i]<=x and x<t[i+1]:
            B[i][0]=1
            break
    for k in range(1, 4):
        for i in range(len(t)-k-1):
            B[i][k]= ((x-t[i])/(t[i+k]-t[i]))*B[i][k-1] + ((t[i+k+1]-x)/(t[i+k+1]-t[i+1]))*B[i+1][k-1]
    return [B[i][3] for i in range(len(B))]

def penalized_B_spline(P, tau,lam=0.1):
    _x, _y = list(zip(*P))
    _y = np.array(_y)
    bases = [b_spline_base(_x, i) for i in _x]
    bases = np.array(bases)

    m = gp.Model()
    n_P = len(P)
    #n_K = n_P+3+1
    #max_x = max(_x)

    x_c = np.linspace(_x[0], _x[-1], 1000)
    c_bases = [b_spline_base(_x, i) for i in x_c]

    alpha = m.addMVar(len(bases[0]), lb=-GRB.INFINITY, ub = GRB.INFINITY, name='alpha')

    BB = bases.transpose()@bases
    v = -2*bases.transpose() @_y
    D = np.diff(np.eye(len(_y)), 2).transpose() #second derivative
    dev = lam*bases.transpose()@D.transpose()@D@bases
    obj = _y@_y + alpha@v+ alpha@BB@alpha + alpha@dev@alpha
    m.setObjective(obj)

    c_bases = np.array(c_bases)
    #y_hat = c_bases@alpha
    #call_hat = [BS_functions.BS_call(S, x_c[i], tau, y_hat[i], rf_free) for i in range(len(y_hat))]

    D1 = np.diff(np.eye(len(x_c)), 1).transpose() #first derivative
    D2 = np.diff(np.eye(len(x_c)), 2).transpose() #second derivative

    #slope downward constraint
    D1_C = D1@c_bases
    m.addConstrs(D1_C[i]@alpha<=0 for i in range(D1.shape[0]))

    #slope shoud be bigger than e^(-rt)
    D1_C_d = -D1@c_bases/(x_c[1]-x_c[0])
    m.addConstrs(D1_C_d[i]@alpha-math.exp(-rf_rate*tau)<=0 for i in range(D1.shape[0]))

    #convex constraint
    D2_C = -D2@c_bases
    m.addConstrs(D2_C[i]@alpha<=0 for i in range(D2.shape[0]))

    #bigger than 0
    m.addConstr(c_bases[-1]@alpha>=0)

    m.optimize()

    return m


def gen_b_spline_curve():
    #expri_date = date(2021, 6, 11)
    expri_date = date(2021, 6, 7)
    tau = cal_tau(today, expiration_date=expri_date)
    df = pd.read_hdf('./data/calls/call_s'+expri_date.strftime('%y%m%d')+'_d210606.h5')
    points = get_imvol_list(expir=expri_date, df=df, moneyness=False, imvol=False)
    #points = points[1:]
    _x, _y = list(zip(*points))
    model = penalized_B_spline(points, tau)
    alpha_hat = [v.x for v in model.getVars()]
    bases = [b_spline_base(_x, i) for i in _x]

    xx, yy = np.linspace(_x[0], _x[-1], 1000), []
    for x in xx:
        yy.append(b_spline_base(_x, x)@np.array(alpha_hat))

    _x, _y = np.array(_x), np.array(_y)
    plt.plot(xx, yy)
    plt.scatter(_x, _y, marker='.', c='k')
    plt.xlabel("Strike")
    plt.ylabel("Premium")
    plt.title('SPX call expiring 21-06-11')
    plt.savefig("spxcall210611_curve.png")
    plt.show()
    
    imvol_y = cal_imvol_np(_y, S, _x, tau, rf_rate)
    yy = np.array(yy)
    line_y_imvol = cal_imvol_np(yy, S, xx, tau, rf_rate)
    plt.plot(xx, line_y_imvol)
    plt.scatter(_x, imvol_y, marker='.', c='k')
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title('SPX call expiring 21-06-11')
    plt.savefig("spxcall210611_curve_imvol.png")
    plt.show()

def gen_b_spline_surface():
    points3D= get3Dpoints()
    _x, _y, _z = [], [], []
    for p in points3D:
        t_x, t_y, t_z = list(map(list, zip(*p)))
        _x.append(t_x)
        _y.append(t_y)
        _z.append(t_z)
    tmp_x = []
    for i in _x:
        tmp_x.extend(i)

    tmp_y = []
    for i in _y:
        tmp_y.extend(i)

    mesh_tau = sorted(list(set(tmp_y)))[:4]
#mesh_strike = sorted(list(set(tmp_x)))[7:-9]
    mesh_strike = sorted(list(set(tmp_x)))[80:-77]

    mask = [[False]*len(mesh_strike) for _ in range(len(mesh_tau))]
    for i, _t in enumerate(mesh_tau):
        for j, _s in enumerate(mesh_strike):
            if _s in _x[i]:
                mask[i][j] = True

    bases_st = [b_spline_base(mesh_strike, i) for i in mesh_strike]
    bases_tau = [b_spline_base(mesh_tau, i) for i in mesh_tau]

    bases_3D = []
    _xx,_yy,_zz = [], [], []
    for i, _t in enumerate(mesh_tau):
        for j, _s in enumerate(mesh_strike):
            if _s in _x[i]:
                bases_3D.append(np.kron(bases_st[j], bases_tau[i]))
                idx = _x[i].index(_s)
                _zz.append(_z[i][idx])
                _xx.append(_x[i][idx])
                _yy.append(_y[i][idx])
    bases_3D = np.array(bases_3D)
    _xx, _yy, _zz = np.array(_xx), np.array(_yy), np.array(_zz)

    c_x = np.linspace(mesh_strike[0], mesh_strike[-1], 100)
    c_y = np.linspace(mesh_tau[0], mesh_tau[-1], 10)

    c_bases_st = [b_spline_base(mesh_strike, i) for i in c_x]
    c_bases_tau = [b_spline_base(mesh_tau, i) for i in c_y]

    c_bases_3D = []
    for i in range(len(c_y)):
        for j in range(len(c_x)):
            c_bases_3D.append(np.kron(c_bases_st[j], c_bases_tau[i]))
    c_bases_3D = np.array(c_bases_3D)

    m = gp.Model()
    alpha = m.addMVar(bases_3D[0].shape[0], lb=-GRB.INFINITY, ub=GRB.INFINITY, name='alpha')

    BB = bases_3D.transpose()@bases_3D
    v = -2*bases_3D.transpose()@_zz
    #lam = 0.001
    #dev = lam*bases_3D.transpose@
    obj = _zz@_zz + alpha@v + alpha@BB@alpha
    m.setObjective(obj)

    c_bases_3D = np.array(c_bases_3D)

    D1 = np.diff(np.eye(len(c_x)), 1).transpose()
    D2 = np.diff(np.eye(len(c_x)), 2).transpose()

    for i in range(len(c_y)):
        cur_bases = c_bases_3D[(i)*len(c_x):(i+1)*len(c_x)]
        
        D1_C = D1@cur_bases
        m.addConstrs(D1_C[j]@alpha<=0 for j in range(D1.shape[0]))
        
        D1_C_d = -D1@cur_bases/(c_x[1]-c_x[0])
        m.addConstrs(D1_C_d[j]@alpha-math.exp(-rf_rate*c_y[i])<=0 for j in range(D1.shape[0]))
        
        D2_c = -D2@cur_bases
        m.addConstrs(D2_c[j]@alpha<=0 for j in range(D2.shape[0]))
        
        m.addConstr(cur_bases[-1]@alpha>=0)

    D1_y = np.diff(np.eye(len(c_y)), 1).transpose()
    for i in range(len(c_x)):
        cur_bases = c_bases_3D[i::len(c_x)]
        D1_CY = D1_y@cur_bases
        m.addConstrs(D1_CY@alpha>=0 for j in range(D1_y.shape[0]))

    m.optimize()

    alpha_hat = np.array([v.x for v in m.getVars()])
    Z = (c_bases_3D@alpha_hat).reshape((len(c_y), len(c_x)))
    X, Y = np.meshgrid(c_x, c_y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8,
                           linewidth=0, antialiased=False)
    ax.scatter(_xx, _yy, _zz, marker='.')
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("Premium")
    plt.title('SPX call expiring 21-06-11 - 21-06-14')
    plt.savefig("spxcall210611_210611_surface.png")
    plt.show()

    imvol_Z = []
    for i in range(len(c_y)):
        imvol_Z.append(cal_imvol_np(Z[i], S, c_x,c_y[i], rf_rate))
    imvol_Z = np.array(imvol_Z)
        
    imvol_zz = []
    for i in range(len(_yy)):
        imvol_zz.append(cal_imvol_np(_zz[i], S, _xx[i],_yy[i], rf_rate))      

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, imvol_Z, cmap=cm.coolwarm, alpha=0.8,
                           linewidth=0, antialiased=False)
    ax.scatter(_xx, _yy, imvol_zz, marker='.')
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("Implied Volatility")
    plt.title('SPX call expiring 21-06-11 - 21-06-14')
    plt.savefig("spxcall210611_210611_surface_imvol.png")
    plt.show()

if __name__=="__main__":
    gen_b_spline_curve()
    gen_b_spline_surface()







