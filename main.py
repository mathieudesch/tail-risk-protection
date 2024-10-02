import numpy as np
import scipy.stats as stats
from scipy.special import erfc
import pandas as pd
from yahoo_fin import options
import yfinance as yf
from datetime import datetime, timedelta
from fredapi import Fred
import pytz
import json

eastern = pytz.timezone('US/Eastern')
options_data = {"calls": [], "puts": []}


def hill_estimator(data, threshold):
    """
    Estimate the tail index using the Hill estimator.
    """
    data = np.array(data)
    exceedances = data[data > threshold]
    n_exceedances = len(exceedances)

    if n_exceedances == 0:
        raise ValueError("No data points exceed the threshold.")

    log_exceedances = np.log(exceedances)
    log_threshold = np.log(threshold)

    hill_estimate = 1 / (np.mean(log_exceedances) - log_threshold)

    return hill_estimate

def call_option_price(S0, K1, K2, C1, alpha):
    if K2 <= S0 or K1 <= S0:
        return C1 * ((K2 - S0) / (K1 - S0)) ** (1 - alpha)
    return ((K2 - S0) / (K1 - S0)) ** (1 - alpha) * C1

def put_option_price(S0, K1, K2, P1, alpha):
    if S0 <= K2 or S0 <= K1:
        return P1 * ((S0 - K2) / (S0 - K1)) ** (1 - alpha)
    numerator = (S0 - K2) ** (1 - alpha) - S0 ** (1 - alpha) * ((alpha - 1) * K2 + S0)
    denominator = (S0 - K1) ** (1 - alpha) - S0 ** (1 - alpha) * ((alpha - 1) * K1 + S0)
    return numerator / denominator * P1


def sigma_prime(K, S0, t, r, option_type='call', h=0.01):
    chain = options.get_options_chain(ticker, closest_exp_date)
    if option_type == 'call':
        option_data = chain['calls']
    else:
        option_data = chain['puts']

    option_data['Strike'] = option_data['Strike'].astype(float)
    option_data['Implied Volatility'] = option_data['Implied Volatility'].str.rstrip('%').astype(float) / 100

    strike_diff = np.abs(option_data['Strike'] - K)

    index_plus = strike_diff[option_data['Strike'] >= K].idxmin()
    index_minus = strike_diff[option_data['Strike'] <= K].idxmin()

    if index_plus == index_minus:
        return option_data.loc[index_plus, 'Implied Volatility']

    sigma_plus = option_data.loc[index_plus, 'Implied Volatility']
    sigma_minus = option_data.loc[index_minus, 'Implied Volatility']

    strike_plus = option_data.loc[index_plus, 'Strike']
    strike_minus = option_data.loc[index_minus, 'Strike']

    sigma = sigma_minus + (sigma_plus - sigma_minus) * (K - strike_minus) / (strike_plus - strike_minus)
    return sigma

def arbitrage_boundary(S0, K1, delta_K, C1, alpha, t, option_type='call'):
    K_plus = K1 + delta_K
    K_minus = K1 - delta_K
    C_plus = call_option_price(S0, K1, K_plus, C1, alpha)
    C_minus = call_option_price(S0, K1, K_minus, C1, alpha)
    return C_plus - C_minus >= 0

def lower_bound_tail_index(S0, K, l, t, option_type='call'):
    sigma = sigma_prime(K, S0, t, r, option_type)
    if sigma <= 0 or K <= S0 or l <= 0 or t <= 0:
        return np.nan
    term1 = -np.log(K - S0) + np.log(l) + np.log(S0)
    term2 = 0.5 * erfc((t * sigma ** 2 + 2 * np.log(K) - 2 * np.log(S0)) / (2 * np.sqrt(2) * np.sqrt(t) * sigma))
    term3 = -np.sqrt(S0) * np.sqrt(t) * sigma_prime(K, S0, t, r, option_type) * K * np.log(S0) / (t * sigma ** 2)
    term4 = 0.5 * np.exp(-(np.log(K) ** 2 + np.log(S0) ** 2) / (2 * t * sigma ** 2) - 0.125 * t * sigma ** 2) / np.sqrt(2 * np.pi)
    return 1 / (term1 * np.log(term2 + term3 + term4))

def black_scholes_call(S0, K, t, sigma, r=0):
    if sigma <= 0 or t <= 0:
        return np.nan
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S0 * stats.norm.cdf(d1) - K * np.exp(-r * t) * stats.norm.cdf(d2)

def main():
    global ticker, closest_exp_date, r

    ticker = input("Enter the ticker symbol: ")
    input_rate = input("Enter the time frame for the bond rate (3m, 6m, 1y, or 2y): ")

    # Fetch historical stock prices
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    stock_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    sigma = stock_data['Daily Return'].std() * np.sqrt(252)  # Annualize sigma

    # Hill estimator
    returns = stock_data['Close'].pct_change().dropna()
    threshold = np.percentile(returns, 95)
    alpha = hill_estimator(returns, threshold)
    print(f"Tail Index (alpha): {alpha:.2f}")

    # Get the current stock price
    current_price = stock_data['Close'].iloc[-1]

    # Fetch US Treasury rates for 3, 6, 12, and 24 months
    fred = Fred(api_key='api key goes here')
    if input_rate == '3m':
        BOND_DATA = fred.get_series('DGS3MO')
        days = 30 * 3
    elif input_rate == '6m':
        BOND_DATA = fred.get_series('DGS6MO')
        days = 30 * 6
    elif input_rate == '1y':
        BOND_DATA = fred.get_series('DGS1')
        days = 30 * 12
    elif input_rate == '2y':
        BOND_DATA = fred.get_series('DGS2')
        days = 30 * 24
    else:
        raise ValueError("Invalid input rate.")

    df_sorted = BOND_DATA.sort_index(ascending=False)
    r = df_sorted.iloc[0]

    # Fetch options data
    exp_dates = options.get_expiration_dates(ticker)
    target_date = datetime.now() + timedelta(days)  # Target date is X months from now

    closest_exp_date = min(exp_dates, key=lambda d: abs(datetime.strptime(d, "%B %d, %Y") - target_date))
    chain = options.get_options_chain(ticker, closest_exp_date)
    puts = chain['puts']
    calls = chain['calls']

    print(f"Ticker: {ticker}")
    print(f"Current Price: {current_price:.2f}")
    print(f"US Treasury Rate: {r:.4f}")
    print(f"Tail Index (alpha): {alpha:.2f}")
    print(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"Closest Expiration Date: {closest_exp_date}")

    print("\nCall Options:")
    print("{:<10} {:<10} {:<15}".format("Strike", "Market", "Calculated"))

    anchor_call = calls.loc[(calls['Strike'] < current_price) & (calls['Last Price'] > 0)].iloc[0]
    anchor_call_strike = anchor_call['Strike']
    anchor_call_price = anchor_call['Last Price']

    print(f"Anchor Call: Strike={anchor_call_strike}, Price={anchor_call_price}")

    for index, row in calls.iterrows():
        K = row['Strike']
        market_price = row['Last Price']
        calculated_price = call_option_price(current_price, anchor_call_strike, K, anchor_call_price, alpha)

        options_data["calls"].append({
            "Strike": row['Strike'],
            "Market Price": market_price,
            "Calculated Price": calculated_price
        })

        print("{:<10} {:<10} {:<15}".format(
            K,
            f"{market_price:.2f}" if not np.isnan(market_price) else "N/A",
            "N/A" if calculated_price is None else f"{calculated_price:.2f}"
        ))

    print("\nPut Options:")
    print("{:<10} {:<10} {:<15}".format("Strike", "Market", "Calculated"))

    anchor_put = puts.loc[(puts['Strike'] > current_price) & (puts['Last Price'] > 0)].iloc[0]
    anchor_put_strike = anchor_put['Strike']
    anchor_put_price = anchor_put['Last Price']

    print(f"Anchor Put: Strike={anchor_put_strike}, Price={anchor_put_price}")

    for index, row in puts.iterrows():
        K = row['Strike']
        market_price = row['Last Price']
        calculated_price = put_option_price(current_price, anchor_put_strike, K, anchor_put_price, alpha)

        options_data["puts"].append({
            "Strike": row['Strike'],
            "Market Price": market_price,
            "Calculated Price": calculated_price
        })

        print("{:<10} {:<10} {:<15}".format(
            K,
            f"{market_price:.2f}" if not np.isnan(market_price) else "N/A",
            "N/A" if calculated_price is None else f"{calculated_price:.2f}"
        ))

    # Arbitrage boundary check
    delta_K = 1.0  # Adjust as needed
    is_arbitrage = arbitrage_boundary(current_price, anchor_call_strike, delta_K, anchor_call_price, alpha, days / 365)
    print(f"Arbitrage Boundary Satisfied: {is_arbitrage}")

    # Lower bound on tail index
    l = 0.1  # Adjust as needed
    t = days / 365  # Adjust as needed
    alpha_lower_bound = lower_bound_tail_index(current_price, anchor_call_strike, l, t)
    print(f"Lower Bound on Tail Index: {alpha_lower_bound:.2f}")


if __name__ == "__main__":
    main()
