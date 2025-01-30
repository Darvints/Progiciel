# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:25:34 2024

@author: Darvin
"""

"""
Liste des fonctions utiles dans le cadre du projet Prima
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from arch import arch_model
import math
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import geom, chisquare
from scipy import optimize
from scipy.stats import chi2


# Fonction qui récupère les prix d'aujourd'hui jusqu'à une période T
def PriceToRdt(actif, T):
    data = yf.download(actif,start=T, interval='1d')['Close']
    Rdt = data.pct_change().dropna()
    return Rdt

# Fonction qui fait des simulations Bootstrap à partir des rendements d'actifs N fois
def Bootstraap_Rdt(Rdt, N):
    Bootstrapped = {}
    T = len(Rdt)
    for i in range(N):
        Random_tirage = Rdt.sample(T, replace=True)
        Bootstrapped[i] = Random_tirage.values
    return Bootstrapped

# VaR Historical Simulation
def VaR_HS(Bootstrapped, w, alpha):
    N = len(Bootstrapped)
    HS = np.zeros(N)
    for i in range(N):
        Rdt = Bootstrapped[i]
        Portfolio_Rdt = np.dot(Rdt, w)
        HS[i] = np.quantile(Portfolio_Rdt, alpha)
    VaR_HS = -np.mean(HS)
    return VaR_HS

# VaR Variance-Covariance
def VaR_VARCOV(Rdt, w, alpha):
    mu = Rdt.mean(axis=0)
    Sigma = Rdt.cov()
    z_alpha = norm.ppf(1 - alpha)  # Quantile de la distribution normale standard
    portfolio_mean = np.dot(w, mu)
    portfolio_std = np.sqrt(np.dot(w, np.dot(Sigma, w)))
    VaR = -(-portfolio_mean - z_alpha * portfolio_std)
    return VaR

# Riskmetrics
def RiskMetrics(returns, alpha, lambda_decay=0.94):
    # Calcul des variances EWMA
    var_ewma = returns.ewm(alpha=(1-lambda_decay)).var()
    current_var = var_ewma.iloc[-1]
    
    # Calcul des covariances EWMA
    cov_ewma = returns.ewm(alpha=(1-lambda_decay)).cov()
    current_cov = cov_ewma.iloc[-len(returns.columns):]
    
    # Calcul de la matrice de corrélation
    std_dev = np.sqrt(current_var)
    corr = current_cov / np.outer(std_dev, std_dev)
    
    # Calcul de la VaR
    z_alpha = norm.ppf(1-alpha)
    position_risk = -z_alpha * std_dev
    var_portfolio = np.sqrt(np.dot(position_risk, np.dot(corr, position_risk)))
    
    return var_portfolio

#VaR GARCH Univarié
def VaR_GARCH(Rdt, w, alpha=0.05):
    Portfolio_rdt = Rdt @ w
    
    fit_mod = arch_model(Portfolio_rdt, p=1, q=1, mean='AR', vol='GARCH', dist='normal')
    fm_result = fit_mod.fit(disp='off')
    
    params = fm_result.params
    
    rho = params['Const']
    alpha0 = params['omega']
    alpha = params['alpha[1]']
    beta = params['beta[1]']
    
    Rt = np.array(Portfolio_rdt)[-1]
    epsilon_t = np.array(fm_result.resid)[-1]
    ht = np.array(fm_result.conditional_volatility)[-1]**2
    
    vol_forecast = np.sqrt(alpha0 + alpha * epsilon_t**2 + beta * ht)
    
    z_alpha = norm.ppf(1 - alpha)  # Quantile de la distribution normale
    expected_return = rho * Rt
    VaR = abs(-(expected_return + vol_forecast * z_alpha))
    
    return VaR

def VaR_CF(Rdt,w,alpha):
    z= norm.ppf(1 - alpha)
    PRdt = Rdt @ w
    S = PRdt.skew(axis = 0)
    K = PRdt.kurt()
    Std = PRdt.std(axis=0)
    VaR = -Std * (z + (z**2 - 1)*(S/6)+(z**3 - 3*z)*(K/24) + (2*z**3 - 5 * z)*((S**2)/36))
    VaR = abs(VaR)
    return VaR

def Violation_VaR(Rdt, VaR):
    violations = np.where(Rdt < VaR, 1, 0)
    return violations

def Kupiec(Rdt,VaR,alpha):
    V = Violation_VaR(Rdt, VaR)
    N = sum(V)
    T = len(Rdt)
    
    alpha_e = N/T
    LRUC = 2 * math.log((((1-alpha_e)/(1-alpha))**(T-N) )* ((alpha_e/alpha)**N))
    #SI LRUC > 3.841 alors test rejeté
    return LRUC
    

def n_ij(V,i,j):
    S = 0
    for k in range(len(V)-1):
        if V[k] == i and V[k+1] == j:
            S = S+1
    return S

def LR_ind(Rdt, VaR, alpha):
    """
    Calcule la statistique du test d'indépendance
    """
    # Calcul des violations
    V = Violation_VaR(Rdt, VaR)
    
    # Calcul des transitions
    n_00 = n_ij(V,0,0)
    n_01 = n_ij(V,0,1)
    n_10 = n_ij(V,1,0)
    n_11 = n_ij(V,1,1)
    
    # Debug info
    print(f"Transitions: n_00={n_00}, n_01={n_01}, n_10={n_10}, n_11={n_11}")
    
    # Éviter division par zéro
    epsilon = 1e-10
    
    # Calcul des probabilités de transition
    pi_01 = (n_01 + epsilon) / (n_00 + n_01 + epsilon)
    pi_11 = (n_11 + epsilon) / (n_10 + n_11 + epsilon)
    
    # Nombre total de violations
    N = n_01 + n_11
    T = len(V)
    pi = N/T  # Probabilité inconditionnelle de violation
    
    try:
        # Premier terme : hypothèse nulle d'indépendance
        term1 = -2 * math.log(max(epsilon, (1 - pi)**(T-N) * pi**N))
        
        # Second terme : vraisemblance alternative
        term2 = 2 * math.log(max(epsilon, 
            (1-pi_01)**(n_00) * 
            max(epsilon, pi_01)**(n_01) * 
            (1-pi_11)**n_10 * 
            max(epsilon, pi_11)**n_11
        ))
        
        LRind = term1 + term2
        
        print(f"Debug: pi={pi:.4f}, term1={term1:.4f}, term2={term2:.4f}")
        
        return LRind
        
    except ValueError as e:
        print(f"Error in LR_ind calculation:")
        print(f"pi={pi}, pi_01={pi_01}, pi_11={pi_11}")
        print(f"N={N}, T={T}")
        raise e

def LR_christoffersen(Rdt,VaR,alpha):
    LRcc = LR_ind(Rdt,VaR,alpha) + Kupiec(Rdt, VaR, alpha)
    #SI LRcc > 5.991 alors test rejeté
    return LRcc

def Auto_cor_LB(Rdt,VaR,alpha):
    V = Violation_VaR(Rdt, VaR)
    It = V-alpha
    # Paramètres
    m = 5  # Nombre de lags à tester

    # Calcul du test de Ljung-Box
    ljung_box_result = acorr_ljungbox(It, lags=m, return_df=True)
    #Si p-value > 0.05 on ne rejette pas le Test
    return np.mean(ljung_box_result['lb_pvalue'])



def duree_test_cp(violations, alpha):
    """
    Implémente le test de durée de Christoffersen & Pelletier (2004)
    
    Parameters:
    -----------
    violations : array-like
        Série de violations (0 ou 1)
    alpha : float, optional
        Niveau de significativité du test (défaut: 0.05)
    
    Returns:
    --------
    dict
        Dictionnaire contenant la statistique LR, la p-value et la décision du test
    """
    
    def duree(violations):
        """Calcule les durées entre violations"""
        durees = []
        compteur = 0
        
        for v in violations:
            if v == 0:
                compteur += 1
            else:
                if compteur > 0:
                    durees.append(compteur)
                compteur = 0
                
        if compteur > 0:
            durees.append(compteur)
            
        return np.array(durees)
    
    def log_likelihood_contraint(params, durees):
        """Log-vraisemblance sous H0 (distribution exponentielle)"""
        a = params[0]
        ll = 0
        for d in durees:
            ll += np.log(a) - a * d
        return -ll
    
    def log_likelihood_non_contraint(params, durees):
        """Log-vraisemblance sous H1 (distribution Weibull)"""
        a, b = params
        ll = 0
        for d in durees:
            ll += np.log(a) + np.log(b) + (b-1)*np.log(d) - a*(d**b)
        return -ll
    
    # Calcul des durées
    durees = duree(violations)
    
    # Optimisation sous H0 (distribution exponentielle)
    res_contraint = optimize.minimize(
        log_likelihood_contraint,
        x0=[0.5],
        args=(durees,),
        method='Nelder-Mead'
    )
    ll_contraint = -res_contraint.fun
    
    # Optimisation sous H1 (distribution Weibull)
    res_non_contraint = optimize.minimize(
        log_likelihood_non_contraint,
        x0=[0.5, 0.5],
        args=(durees,),
        method='Nelder-Mead'
    )
    ll_non_contraint = -res_non_contraint.fun
    
    # Calcul de la statistique LR
    lr_stat = 2 * (ll_non_contraint - ll_contraint)
    
    # Calcul de la p-value (distribution chi2 à 2 degrés de liberté)
    p_value = 1 - chi2.cdf(lr_stat, df=2)
    
    # Décision du test
    decision = "Rejet de H0" if p_value < alpha else "Non rejet de H0"
    
    return p_value

def CP(Rdt,VaR,alpha):
    V = Violation_VaR(Rdt, VaR)
    return duree_test_cp(V, alpha)
    
    



