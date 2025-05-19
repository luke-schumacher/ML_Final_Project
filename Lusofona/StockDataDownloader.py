import yfinance as yf
import pandas as pd
import time

def download_stock_data(ticker, max_retries=5, retry_delay=5):
    """
    Downloads historical stock data from Yahoo Finance, handling potential rate limit errors.

    Args:
        ticker (str): The stock ticker symbol (e.g., "NVDA").
        max_retries (int, optional): Maximum number of retries. Defaults to 5.
        retry_delay (int, optional): Delay in seconds between retries. Defaults to 5.

    Returns:
        pandas.DataFrame: The historical stock data, or None if download fails after retries.
    """
    for attempt in range(max_retries):
        try:
            # Attempt to download the data
            stock_data = yf.download(ticker, period="max")

            # If the download is successful, break out of the retry loop
            if stock_data is not None and not isinstance(stock_data, str) and not stock_data.empty:
                print(f"Successfully downloaded data for {ticker} after {attempt + 1} attempts.")
                return stock_data
            elif isinstance(stock_data, str):
                print(f"Error downloading data for {ticker}: {stock_data}")
                return None

        except yf.YFError as e:
            if "Too Many Requests" in str(e):
                print(f"Rate limit hit for {ticker}. Retrying in {retry_delay} seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                print(f"Error downloading data for {ticker}: {e}")
                return None  # Exit the function for other errors
        except Exception as e:
            print(f"An unexpected error occurred for {ticker}: {e}")
            return None

    print(f"Failed to download data for {ticker} after {max_retries} attempts.")
    return None  # Return None if all retries fail


def process_stock_data(stock_data, ticker):
    """
    Processes the downloaded stock data, handling MultiIndex columns and saving to CSV.

    Args:
        stock_data (pandas.DataFrame): The downloaded stock data.
        ticker (str): The stock ticker symbol.
    """
    # Display the type of columns and the columns themselves before processing
    print("Type of DataFrame columns before processing:", type(stock_data.columns))
    print("DataFrame columns before processing:\n", stock_data.columns)
    print("DataFrame index name:", stock_data.index.name)

    # Check if the columns are a MultiIndex and flatten them if necessary
    if isinstance(stock_data.columns, pd.MultiIndex):
        print("\nDetected MultiIndex columns. Flattening columns...")
        flat_column_names = [col[0] for col in stock_data.columns]
        stock_data.columns = flat_column_names
        print("DataFrame columns after flattening:", stock_data.columns)
    else:
        print("\nColumns are not a MultiIndex. Proceeding...")

    # Reset the index to make 'Date' a regular column
    stock_data_reset = stock_data.reset_index()

    # Display the columns of the new DataFrame after resetting the index
    print("\nDataFrame columns after reset_index():", stock_data_reset.columns)
    print("First few rows of the DataFrame after reset_index():\n", stock_data_reset.head())

    # Save the processed DataFrame to CSV
    output_filename = f"{ticker}_historical_data.csv"
    stock_data_reset.to_csv(output_filename, index=False)
    print(f"\nDownloaded historical data for {ticker} and saved to {output_filename} with corrected format.")

    # Optional: Print the first few lines of the saved CSV to confirm
    print(f"\nFirst few lines of the generated CSV ({output_filename}):")
    try:
        with open(output_filename, 'r') as f:
            max_lines_to_print = 7
            lines_printed = 0
            while lines_printed < max_lines_to_print:
                line = f.readline().strip()
                if not line:
                    if lines_printed == 0:
                        print("CSV file is empty or could not be read.")
                    break
                print(f"CSV Line {lines_printed + 1}: {line}")
                lines_printed += 1
    except FileNotFoundError:
        print(f"Corrected CSV file '{output_filename}' not found.")



if __name__ == "__main__":
    # Define the ticker
    ticker = "AMT"
    # Download and process
    stock_data = download_stock_data(ticker)
    if stock_data is not None:
        process_stock_data(stock_data, ticker)
    else:
        print(f"Failed to retrieve stock data for {ticker} after multiple retries.")
