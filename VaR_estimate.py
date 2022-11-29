import numpy as np
import pandas as pd
from data import (
    stocks_returns,
    commodities_returns,
    cryptocurrencies_returns,
    Dataloader
)
from metrics import pof_test, if_test, quantile_loss
from models import HistoricalSimulation, RiskMetrics, GARCH11
from scipy.stats import norm, genhyperbolic

VAR = 0.9
VAR_WINDOW = 10
WINDOWS_SIZE = 125
N = 200
INF = 10 ** 9


def mc_window_var(model_type):
    assets = ['AAPL', 'GOOGL']
    weights = [0.3, 0.7]
    returns = stocks_returns(assets, weights, from_date='09/01/2020', to_date='09/01/2022')
    logreturns = np.log(returns + 1)
    loader = Dataloader(
        series=logreturns,
        window_size=WINDOWS_SIZE,  # a half of trading year
        step_size=1,
        horizon=1,
        first_pred=WINDOWS_SIZE + 1
    )
    if model_type == 'rm':
        model = RiskMetrics(VAR)
    elif model_type == 'garch':
        model = GARCH11(VAR, WINDOWS_SIZE)
    breaks = 0
    window_count = 0
    for mod in range(VAR_WINDOW):
        window_count -= 1
        count = 0
        actual_window_return = 0
        window_var = -INF
        for feat, _ in loader:
            if count % VAR_WINDOW == mod:
                window_count += 1
                if actual_window_return < window_var:
                    breaks += 1
                actual_window_return = 0
                if model_type == 'rm':
                    hyperbolic_params, sigma_sq = model.get_hyperbolic_dist(feat)
                elif model_type == 'garch':
                    params, hyperbolic_params, sigma_sq = model.get_hyperbolic_dist(feat)
                    omega, alpha1, beta1 = params[0], params[1], params[2]

                window_returns = list()
                log_return = feat.iloc[-1]
                for mc_simulation in range(N):
                    gen_epsilons = genhyperbolic.rvs(*hyperbolic_params, size=VAR_WINDOW)
                    summ = 0
                    for i in range(VAR_WINDOW):
                        summ += log_return
                        if model_type == 'rm':
                            sigma_sq = model.lambd * sigma_sq + (1 - model.lambd) * (log_return ** 2)
                        elif model_type == 'garch':
                            sigma_sq = omega + alpha1 * (log_return ** 2) + beta1 * sigma_sq
                        log_return = (sigma_sq ** 0.5) * gen_epsilons[i]
                    window_returns.append(summ)
                window_var = np.quantile(window_returns, 1 - VAR)
            actual_window_return += feat.iloc[-1]
            count += 1

    output = open('results.txt', 'w')

    print('USING MODEL', model_type.capitalize(), file=output)
    print('ESTIMATING VAR =', 1 - VAR, file=output)
    print('NUMBER OF WINDOWS', window_count, file=output)
    print('NUMBER OF BREACHES', breaks, file=output)
    print('BREACHES RATIO:', breaks / window_count, file=output)


def is_window_var(model_type):
    assets = ['AAPL', 'GOOGL']
    weights = [0.3, 0.7]
    returns = stocks_returns(assets, weights, from_date='09/01/2020', to_date='09/01/2022')
    logreturns = np.log(returns + 1)
    loader = Dataloader(
        series=logreturns,
        window_size=WINDOWS_SIZE,  # a half of trading year
        step_size=1,
        horizon=1,
        first_pred=WINDOWS_SIZE + 1
    )
    if model_type == 'rm':
        model = RiskMetrics(VAR)
    elif model_type == 'garch':
        model = GARCH11(VAR, WINDOWS_SIZE)
    breaks = 0
    window_count = 0
    for mod in range(VAR_WINDOW):
        window_count -= 1
        count = 0
        actual_window_return = 0
        window_var = -INF
        for feat, _ in loader:
            if count % VAR_WINDOW == mod:
                window_count += 1
                if actual_window_return < window_var:
                    breaks += 1
                actual_window_return = 0
                if model_type == 'rm':
                    hyperbolic_params, sigma_sq = model.get_hyperbolic_dist(feat)
                elif model_type == 'garch':
                    params, hyperbolic_params, sigma_sq = model.get_hyperbolic_dist(feat)
                    omega, alpha1, beta1 = params[0], params[1], params[2]

                window_returns = list()
                log_return = feat.iloc[-1]
                for mc_simulation in range(N):
                    gen_epsilons = genhyperbolic.rvs(*hyperbolic_params, size=VAR_WINDOW)
                    summ = 0
                    for i in range(VAR_WINDOW):
                        summ += log_return
                        if model_type == 'rm':
                            sigma_sq = model.lambd * sigma_sq + (1 - model.lambd) * (log_return ** 2)
                        elif model_type == 'garch':
                            sigma_sq = omega + alpha1 * (log_return ** 2) + beta1 * sigma_sq
                        log_return = (sigma_sq ** 0.5) * gen_epsilons[i]
                    window_returns.append(summ)
                window_var = np.quantile(window_returns, 1 - VAR)
            actual_window_return += feat.iloc[-1]
            count += 1

    output = open('results.txt', 'w')

    print('USING MODEL', model_type.capitalize(), file=output)
    print('ESTIMATING VAR =', 1 - VAR, file=output)
    print('NUMBER OF WINDOWS', window_count, file=output)
    print('NUMBER OF BREACHES', breaks, file=output)
    print('BREACHES RATIO:', breaks / window_count, file=output)


mc_window_var('garch')
