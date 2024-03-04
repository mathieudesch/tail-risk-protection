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
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:  # 'put'
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return price


def main():
    ticker = input("Enter the ticker symbol: ")

    # Automatic date range for exactly one year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Fetch historical stock prices
    stock_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    sigma = stock_data['Daily Return'].std() * np.sqrt(252)  # Annualize sigma

    # Get the current stock price
    current_price = stock_data['Close'].iloc[-1]

    # Fetch the 1-year US Treasury rate
    # Create a Fred object
    fred = Fred(api_key='81b42521f02d005e5d11afe53a81b757')

    # Retrieve the data
    ONE_YEAR_BONDS = fred.get_series('DGS1')

    # Print the data
    df_sorted = ONE_YEAR_BONDS.sort_index(ascending=False)
    CURRENT_ONE_YEAR_RATE = df_sorted.iloc[0]
    r = CURRENT_ONE_YEAR_RATE

    # Fetch options data
    exp_dates = options.get_expiration_dates(ticker)
    if exp_dates:
        chain = options.get_options_chain(ticker, exp_dates[0])  # For simplicity, use the first expiration date
        puts = chain['puts']
        calls = chain['calls']
        # Example: Calculate and display for calls (similar for puts)
        print("Call Options:")

        for index, row in calls.iterrows():
            K = row['Strike']
            # Extract just the date part from 'Last Trade Date'
            # Assuming the date format is like '2024-02-29 3:19PM EST'
            last_trade_date_str = row['Last Trade Date'].split(' ')[0]  # Gets '2024-02-29'
            last_trade_date = pd.to_datetime(last_trade_date_str, format='%Y-%m-%d')

            # Calculate 'T' using just the date part, ignoring time
            now_date = datetime.now().date()  # Get the current date without time
            T = (last_trade_date.date() - now_date).days / 365.0

            calculated_price = option_price(current_price, K, r, sigma, T, 'call')
            options_data["calls"].append({
                "Date": last_trade_date_str,
                "Strike": row['Strike'],
                "Market Price": row['Last Price'],
                "Calculated Price": calculated_price if not np.isnan(calculated_price) else None  # Handle NaN values
            })

    # Similar block for puts...
        print("Put Options:")
        for index, row in calls.iterrows():
            K = row['Strike']
            # Extract just the date part from 'Last Trade Date'
            # Assuming the date format is like '2024-02-29 3:19PM EST'
            last_trade_date_str = row['Last Trade Date'].split(' ')[0]  # Gets '2024-02-29'
            last_trade_date = pd.to_datetime(last_trade_date_str, format='%Y-%m-%d')

            # Calculate 'T' using just the date part, ignoring time
            now_date = datetime.now().date()  # Get the current date without time
            T = (last_trade_date.date() - now_date).days / 365.0

            calculated_price = option_price(current_price, K, r, sigma, T, 'put')
            options_data["puts"].append({
                "Date": last_trade_date_str,
                "Strike": row['Strike'],
                "Market Price": row['Last Price'],
                "Calculated Price": calculated_price if not np.isnan(calculated_price) else None  # Handle NaN values
            })
            options_json = json.dumps(options_data, indent=4)  # 'indent' for pretty printing

            # Print or save the JSON string
            print(options_json)
if __name__ == "__main__":
    main()
