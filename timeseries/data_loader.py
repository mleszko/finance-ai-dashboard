import yfinance as yf
import pandas as pd
from pathlib import Path


def download_price_data(ticker: str, period: str, save_to: str = None) -> pd.DataFrame:
    """
    Downloads historical price data from Yahoo Finance for a given ticker.

    Args:
        ticker (str): Stock or crypto symbol (e.g. 'AAPL', 'BTC-USD')
        period (str): Time period to download data for (e.g. '1y', '1mo')
        save_to (str, optional): Optional path to save CSV file

    Returns:
        pd.DataFrame: DataFrame with historical prices
    """
    try:
        # Create Ticker object
        stock = yf.Ticker(ticker)
        print(f"Info on Ticker {stock.info}")
        # Get historical data
        df = stock.history(period=period)

        if df.empty:
            print(f"No data available for {ticker}")
            return pd.DataFrame()

        df.reset_index(inplace=True)

        if save_to:
            output_path = Path(save_to)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

        return df

    except Exception as e:
        print(f"Error downloading data for {ticker}: {str(e)}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Test with just the ticker
    df = download_price_data(
        ticker="MSFT",
        period="1y",
        save_to="data/MSFT_prices.csv"
    )

    if not df.empty:
        print("\nFirst few rows of data:")
        print(df.head())
    else:
        print("Failed to retrieve data")