import numpy as np
import scipy.stats as stats
import pandas as pd
from yahoo_fin import options
import yfinance as yf
from datetime import datetime, timedelta
from fredapi import Fred
import pytz
import json

eastern = pytz.timezone('US/Eastern')
options_data = {"calls": [], "puts": []}


def option_price(S, K, r, sigma, T, option_type='call'):
    """
    Calculate the option price.
    """
    #S = current price
    #K = strike price
    #r = 1 year bond
    #sigma = degree of variation over the last year
    #T = time to expiration in years
    #

    if T <= 0:
        return 9999
    xd = sigma * np.sqrt(T)
    xd2 = K

    if xd != 0 and xd2 != 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    else:
        return 6666
    if option_type == 'call':
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:  # 'put'
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return price


def main():
    ticker = input("Enter the ticker symbol: ")

    # Fetch historical stock prices
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    stock_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    sigma = stock_data['Daily Return'].std() * np.sqrt(252)  # Annualize sigma

    # Get the current stock price
    current_price = stock_data['Close'].iloc[-1]

    # Fetch the 1-year US Treasury rate
    fred = Fred(api_key='81b42521f02d005e5d11afe53a81b757')
    ONE_YEAR_BONDS = fred.get_series('DGS1')
    df_sorted = ONE_YEAR_BONDS.sort_index(ascending=False)
    CURRENT_ONE_YEAR_RATE = df_sorted.iloc[0]
    r = CURRENT_ONE_YEAR_RATE

    # Fetch options data
    exp_dates = options.get_expiration_dates(ticker)
    target_date = datetime.now() + timedelta(days=365)  # Target date is 12 months from now

    closest_exp_date = min(exp_dates, key=lambda d: abs(datetime.strptime(d, "%B %d, %Y") - target_date))
    chain = options.get_options_chain(ticker, closest_exp_date)
    puts = chain['puts']
    calls = chain['calls']

    print(f"Ticker: {ticker}")
    print(f"Current Price: {current_price:.2f}")
    print(f"1-Year US Treasury Rate: {r:.4f}")
    print(f"Annualized Volatility (sigma): {sigma:.4f}")
    print(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"Closest Expiration Date: {closest_exp_date}")
    print("\nCall Options:")
    print("{:<10} {:<10} {:<15} {:<20}".format("Strike", "Market", "Calculated", "Difference"))

    for index, row in calls.iterrows():
        K = row['Strike']
        exp_date = pd.to_datetime(closest_exp_date, format='%B %d, %Y')
        now_date = datetime.now().date()
        T = abs((exp_date.date() - now_date).days / 365.0)

        calculated_price = option_price(current_price, K, r, sigma, T, 'call')
        market_price = row['Last Price']
        difference = calculated_price - market_price if not np.isnan(calculated_price) and not np.isnan(market_price) else None

        options_data["calls"].append({
            "Strike": row['Strike'],
            "Market Price": market_price,
            "Calculated Price": calculated_price if not np.isnan(calculated_price) else None,
            "Difference": difference
        })

        print("{:<10} {:<10} {:<15} {:<20}".format(
            K,
            f"{market_price:.2f}" if not np.isnan(market_price) else "N/A",
            f"{calculated_price:.2f}" if not np.isnan(calculated_price) else "N/A",
            f"{difference:.2f}" if difference is not None else "N/A"
        ))

    print("\nPut Options:")
    print("{:<10} {:<10} {:<15} {:<20}".format("Strike", "Market", "Calculated", "Difference"))

    for index, row in puts.iterrows():
        K = row['Strike']
        exp_date = pd.to_datetime(closest_exp_date, format='%B %d, %Y')
        now_date = datetime.now().date()
        T = abs((exp_date.date() - now_date).days / 365.0)

        calculated_price = option_price(current_price, K, r, sigma, T, 'put')
        market_price = row['Last Price']
        difference = calculated_price - market_price if not np.isnan(calculated_price) and not np.isnan(market_price) else None

        options_data["puts"].append({
            "Strike": row['Strike'],
            "Market Price": market_price,
            "Calculated Price": calculated_price if not np.isnan(calculated_price) else None,
            "Difference": difference
        })

        print("{:<10} {:<10} {:<15} {:<20}".format(
            K,
            f"{market_price:.2f}" if not np.isnan(market_price) else "N/A",
            f"{calculated_price:.2f}" if not np.isnan(calculated_price) else "N/A",
            f"{difference:.2f}" if difference is not None else "N/A"
        ))



if __name__ == "__main__":
    main()