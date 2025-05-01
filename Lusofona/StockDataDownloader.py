import yfinance as yf
import pandas as pd

# Define the ticker
ticker = "VST"

# Download historical data (max available)
# yf.download can sometimes return a DataFrame with MultiIndex columns
# when downloading a single ticker, which seems to be causing the extra header row.
stock_data = yf.download(ticker, period="max")

# Display the type of columns and the columns themselves before processing
print("Type of DataFrame columns before processing:", type(stock_data.columns))
print("DataFrame columns before processing:\n", stock_data.columns)
print("DataFrame index name:", stock_data.index.name)

# Check if the columns are a MultiIndex and flatten them if necessary
# The extra row with the ticker symbol suggests the columns are a MultiIndex
# with the structure often being (Metric, Ticker). We want to keep only the Metric.
if isinstance(stock_data.columns, pd.MultiIndex):
    print("\nDetected MultiIndex columns. Flattening columns...")
    # Get the column names from the first level of the MultiIndex.
    # This assumes the structure is (Metric, Ticker) and we want the Metric names
    # like 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
    # We use a list comprehension to iterate through the MultiIndex tuples
    # and take the first element of each tuple.
    flat_column_names = [col[0] for col in stock_data.columns]

    # Assign the new flat column names to the DataFrame
    stock_data.columns = flat_column_names
    print("DataFrame columns after flattening:", stock_data.columns)
else:
    print("\nColumns are not a MultiIndex. Proceeding...")
    # If columns are already flat, ensure standard names are present
    # This might involve renaming if yfinance output format is unexpected,
    # but based on typical yfinance output for single tickers, they should be standard.
    # We can just proceed as the columns are likely already in the desired format.
    pass # No action needed if columns are already flat


# Reset the index to make 'Date' a regular column
# This converts the Date index into a column named 'Date'.
# This is done after potentially flattening the columns so the 'Date' column
# is added correctly as the first column.
stock_data_reset = stock_data.reset_index()

# Display the columns of the new DataFrame after resetting the index
print("\nDataFrame columns after reset_index():", stock_data_reset.columns)
print("First few rows of the DataFrame after reset_index():\n", stock_data_reset.head())


# Save the processed DataFrame to CSV
# We set index=False because the Date is now a regular column and not the index.
# header=True is the default and will write the column names
# ('Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')
# as the single header row.
output_filename = f"{ticker}_historical_data.csv"
stock_data_reset.to_csv(output_filename, index=False)

print(f"\nDownloaded historical data for {ticker} and saved to {output_filename} with corrected format.")

# Optional: Print the first few lines of the saved CSV to confirm
print(f"\nFirst few lines of the generated CSV ({output_filename}):")
try:
    with open(output_filename, 'r') as f:
        # Read and print lines until a line starting with a digit (indicating data) is found,
        # or print a maximum number of lines.
        max_lines_to_print = 7 # Print header + a few data rows
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