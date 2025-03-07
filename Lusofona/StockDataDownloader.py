import yfinance as yf

# Define the ticker
ticker = "AAPL"

# Download historical data (max available)
aapl_data = yf.download(ticker, period="max")

# Save to CSV
aapl_data.to_csv(f"{ticker}_historical_data.csv")

print(f"Downloaded historical data for {ticker} and saved to {ticker}_historical_data.csv")
