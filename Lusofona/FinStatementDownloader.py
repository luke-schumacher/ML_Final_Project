import requests
import pandas as pd

# Your API key
API_KEY = "INSERT HERE"

# Base URL for the Financial Modeling Prep API
BASE_URL = "https://financialmodelingprep.com/api/v3/"

# Define the stock symbol and the types of financial statements
symbol = "NVDA"  # Example: Apple
statement_types = [
    "income-statement",  # Income Statement
    "balance-sheet-statement",  # Balance Sheet
    "cash-flow-statement",  # Cash Flow Statement
    "financial-ratios"  # Financial Ratios (can add more types)
]

# Function to fetch data from the API
def fetch_financial_data(url):
    response = requests.get(url)
    
    # Check if the response is successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

# Initialize an empty DataFrame to hold all data
all_data = pd.DataFrame()

# Loop through each statement type and fetch data
for statement_type in statement_types:
    # Construct the URL for the current financial statement type
    url = f"{BASE_URL}{statement_type}/{symbol}?apikey={API_KEY}&period=10"
    
    # Fetch the financial data for the current statement type
    financial_data = fetch_financial_data(url)
    
    # If data is fetched successfully, process and append it to the all_data DataFrame
    if financial_data:
        df = pd.DataFrame(financial_data)
        df['Statement_Type'] = statement_type  # Add a column to identify the statement type
        all_data = pd.concat([all_data, df], ignore_index=True)  # Append the data

# Save all data to a single CSV file
if not all_data.empty:
    output_filename = f"{symbol}_all_financial_data.csv"
    all_data.to_csv(output_filename, index=False)
    print(f"All financial data saved to {output_filename}")
else:
    print("No data was fetched.")
