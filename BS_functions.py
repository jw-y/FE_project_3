import math
from scipy.stats import norm
import pandas as pd

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

rt_255 = math.sqrt(255)
def annualize_std(std):
    return std * rt_255

rf_rate = pd.read_hdf('./data/TBILL.h5').iloc[-1].item()/100
def get_risk_free():
    return rf_rate

def cal_tau(today, expiration_date):
    return (expiration_date-today).days / 365

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
    while True:
        h = (top+bot)/2
        diff = BS(S, K, T, h, r) - premium
        if abs(diff) <= epsilon:
            return h
        if diff < 0: bot = h
        else: top = h
