import yfinance as yf
import pandas as pd
from pathlib import Path

def download_price_data(ticker: str, start_date: str, end_date: str, save_to: str = None) -> pd.DataFrame:
    """
    Downloads historical price data from Yahoo Finance for a given ticker.

    Args:
        ticker (str): Stock or crypto symbol (e.g. 'AAPL', 'BTC-USD')
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        save_to (str): Optional path to save CSV file

    Returns:
        pd.DataFrame: DataFrame with historical prices
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)

    if save_to:
        output_path = Path(save_to)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df

# Example usage:
if __name__ == "__main__":
    df = download_price_data("AAPL", "2022-01-01", "2023-01-01", save_to="data/AAPL_prices.csv")
    print(df.head())