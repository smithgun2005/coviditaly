# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 19:59:56 2024

@author: 12114
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
file_path = r'C:\Users\12114\Desktop\italy_total.csv'
df = pd.read_csv(file_path, header=None, skiprows=1)
I_actual = df[6].tolist()
R_actual = df[7].tolist()
column_12 = df[11].tolist()
column_9 = df[8].tolist()
activecases = [column_12[i] - R_actual[i] - column_9[i] for i in range(len(column_12))]
I_actual = [0, 0, 0, 0, 1, 1, 1, 1] + activecases

# 初始变量
N = 60461826
days = 1139
t = np.linspace(0, days - 1, days)
true_I = np.array(I_actual)[:days]

# SEIR微分方程
def deriv(y, t, N, beta1, beta2, gamma, delta, alpha, mu, tau, k, eta, mu_inf):
    S, E, I, R = y
    dSdt = tau(t) - (beta1(t) * S * E / N + beta2(t) * S * I / N + mu(t) * S + alpha(t) * S)
    dEdt = beta1(t) * S * E / N + beta2(t) * S * I / N - (k(t) + gamma) * E
    dIdt = k(t) * E - (delta + mu(t) + mu_inf(t) + eta(t)) * I
    dRdt = delta * I - mu(t) * R
    return dSdt, dEdt, dIdt, dRdt

# 定义参数函数
def beta1(t, beta10, beta11):
    return beta10 + (beta11 - beta10) * t / days

def beta2(t, beta20, beta21):
    return beta20 + (beta21 - beta20) * t / days

def alpha(t, alpha0, alpha1):
    return alpha0 + (alpha1 - alpha0) * t / days

def mu(t, mu0, mu1):
    return mu0 + (mu1 - mu0) * t / days

def tau(t, tau0, tau1):
    return tau0 + (tau1 - tau0) * t / days

def k(t, k0, k1):
    return k0 + (k1 - k0) * t / days

def eta(t, eta0, eta1):
    return eta0 + (eta1 - eta0) * t / days

def mu_inf(t, mu10, mu11):
    return mu10 + (mu11 - mu10) * t / days

# 初始参数值
params_init = [0.3, 0.2, 0.3, 0.2, 0.02, 0.005, 0.01, 0.015, 0.25, 0.1, 0.2, 0.1, 0.05, 0.02, 0.01, 0.02]
gamma = 1 / 7.0
delta = 1 / 10.0

# 误差函数
def objective(params, true_segment, t_range, y0):
    beta10, beta11, beta20, beta21, alpha0, alpha1, mu0, mu1, tau0, tau1, k0, k1, eta0, eta1, mu10, mu11 = params
    
    # 解微分方程
    ret = odeint(deriv, y0, t_range, args=(N, 
                                           lambda t: beta1(t, beta10, beta11),
                                           lambda t: beta2(t, beta20, beta21),
                                           gamma, delta,
                                           lambda t: alpha(t, alpha0, alpha1),
                                           lambda t: mu(t, mu0, mu1),
                                           lambda t: tau(t, tau0, tau1),
                                           lambda t: k(t, k0, k1),
                                           lambda t: eta(t, eta0, eta1),
                                           lambda t: mu_inf(t, mu10, mu11)))
    S, E, I, R = ret.T
    
    model_values = I
    
    # 计算R²
    ss_res = np.sum((true_segment - model_values) ** 2)
    ss_tot = np.sum((true_segment - np.mean(true_segment)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 负的R²值作为误差进行最小化
    return -r_squared

# 分段拟合
def fit_segment(start_day, end_day, y0, min_days=10, target_r2=0.9):
    t_range = np.linspace(start_day, end_day - 1, end_day - start_day)
    true_segment = true_I[start_day:end_day]
    
    result = minimize(objective, params_init, args=(true_segment, t_range, y0), method='Nelder-Mead')
    best_params = result.x
    
    ret = odeint(deriv, y0, t_range, args=(N, 
                                           lambda t: beta1(t, best_params[0], best_params[1]),
                                           lambda t: beta2(t, best_params[2], best_params[3]),
                                           gamma, delta,
                                           lambda t: alpha(t, best_params[4], best_params[5]),
                                           lambda t: mu(t, best_params[6], best_params[7]),
                                           lambda t: tau(t, best_params[8], best_params[9]),
                                           lambda t: k(t, best_params[10], best_params[11]),
                                           lambda t: eta(t, best_params[12], best_params[13]),
                                           lambda t: mu_inf(t, best_params[14], best_params[15])))
    S, E, I, R = ret.T
    
    # 计算分段的R²
    ss_res = np.sum((true_segment - I) ** 2)
    ss_tot = np.sum((true_segment - np.mean(true_segment)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"Subsegment ({start_day}, {end_day}) R² for I: {r_squared}")

    if r_squared >= target_r2 or (end_day - start_day) <= min_days:
        return best_params, I, (S[-1], E[-1], I[-1], R[-1]), r_squared
    else:
        mid_day = (start_day + end_day) // 2
        params1, segment1, y0_1, r2_1 = fit_segment(start_day, mid_day, y0, min_days, target_r2)
        params2, segment2, y0_2, r2_2 = fit_segment(mid_day, end_day, y0_1, min_days, target_r2)
        return params1 + params2, np.concatenate((segment1, segment2)), y0_2, (r2_1 + r2_2) / 2

# 自适应拟合
def adaptive_fitting(t, y0, true_data, segment_days=100, min_days=10, target_r2=0.9):
    segments = [(i, min(i + segment_days, len(t))) for i in range(0, len(t), segment_days)]
    all_params = []
    all_segments = []
    
    for start, end in segments:
        params, segment, y0, r_squared = fit_segment(start, end, y0, min_days, target_r2)
        all_params.extend(params)
        all_segments.extend(segment)
        print(f"Segment ({start}, {end}) R² for I: {r_squared}")
    
    return np.array(all_params), np.array(all_segments)

# 执行自适应拟合
y0 = (N - 3 - 0, 150, 3, 0)  # 初始条件
params_I, all_I = adaptive_fitting(t, y0, true_I)

# 计算整体R²
def calculate_r_squared(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

r_squared_I = calculate_r_squared(true_I, all_I)

print("Overall R² for I:", r_squared_I)

# 绘图
plt.figure(figsize=(24, 8))

# I的拟合图
plt.plot(t, true_I, 'bo', label='True Infected')
plt.plot(t, all_I, 'r-', label='Model Infected', linewidth=4)
plt.xlabel('Days')
plt.ylabel('Infected Population')
plt.title('SEIR Model Fit to Real Data (Infected)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
