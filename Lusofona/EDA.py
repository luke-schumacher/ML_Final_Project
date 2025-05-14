import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- GLOBAL VARIABLES START ---
# Define the stock ticker and a base title as global variables
# You can change these values here to analyze a different stock
GLOBAL_STOCK_TICKER = "AAPL" # Default ticker
GLOBAL_STOCK_TITLE = "Stock Analysis" # Base title for charts and outputs
# --- GLOBAL VARIABLES END ---


def load_data(financial_csv_path, stock_price_csv_path):
    """
    Load and perform initial processing of financial and stock price data

    Parameters:
    financial_csv_path (str): Path to financial statements CSV
    stock_price_csv_path (str): Path to stock price CSV

    Returns:
    tuple: (financial_df, stock_price_df)
    """
    # Load financial data
    print("Loading financial data...")
    # Assuming financial data CSV format is standard with a 'date' column
    financial_df = pd.read_csv(financial_csv_path)

    # Load stock price data
    print("Loading stock price data...")
    # Adjusting read_csv to handle the specific header structure observed
    # We assume the column names are in the first row (index 0)
    # and there might be extra rows (like the ticker row) that should not be headers.
    # parse_dates=['Date'] ensures the 'Date' column is treated as datetime objects.
    try:
        stock_price_df = pd.read_csv(stock_price_csv_path, header=[0], parse_dates=['Date'])
        # After reading with header=[0], the columns should be correctly named
        # and the data starts from the third row (index 2 in 0-based).
        # The 'Date' column will be a regular column, not the index.

        # Rename the 'Date' column to lowercase 'date' for consistency with financial data
        if 'Date' in stock_price_df.columns:
             stock_price_df.rename(columns={'Date': 'date'}, inplace=True)
        elif 'date' not in stock_price_df.columns:
             print("Warning: 'Date' column not found in stock price data after loading.")


    except Exception as e:
        print(f"Error loading stock price data with adjusted header: {e}")
        print("Attempting to load with default settings as a fallback...")
        # Fallback to default read if the specific header handling fails
        stock_price_df = pd.read_csv(stock_price_csv_path, parse_dates=['Date'])
         # Rename the 'Date' column to lowercase 'date' for consistency with financial data
        if 'Date' in stock_price_df.columns:
             stock_price_df.rename(columns={'Date': 'date'}, inplace=True)
        elif 'date' not in stock_price_df.columns:
             print("Warning: 'Date' column not found in stock price data after fallback loading.")

    # Convert date columns to datetime format (redundant for stock_price_df if parse_dates worked, but safe)
    financial_df['date'] = pd.to_datetime(financial_df['date'])
    # Ensure 'date' column is datetime in stock_price_df
    if 'date' in stock_price_df.columns:
        stock_price_df['date'] = pd.to_datetime(stock_price_df['date'])
    else:
        print("Critical Error: 'date' column is missing from stock price data after loading.")
        # Depending on severity, you might want to raise an error or return None here.
        # For now, we'll continue but subsequent steps might fail.


    # Display information about the loaded data
    print(f"Financial data shape: {financial_df.shape}")
    print(f"Financial data date range: {financial_df['date'].min()} to {financial_df['date'].max()}")
    print(f"Financial data columns: {financial_df.columns.tolist()}")
    if 'period' in financial_df.columns:
        print(f"Financial data period types: {financial_df['period'].unique().tolist()}")
    else:
        print("Warning: 'period' column not found in financial data.")


    print(f"\nStock price data shape: {stock_price_df.shape}")
    if 'date' in stock_price_df.columns:
        print(f"Stock price data date range: {stock_price_df['date'].min()} to {stock_price_df['date'].max()}")
        print(f"Stock price data columns: {stock_price_df.columns.tolist()}")
    else:
         print("Stock price data date range: N/A ('date' column missing)")
         print(f"Stock price data columns: {stock_price_df.columns.tolist()}")


    return financial_df, stock_price_df

def prepare_financial_data(financial_df):
    """
    Clean and prepare financial data for merging

    Parameters:
    financial_df (DataFrame): Raw financial data

    Returns:
    DataFrame: Processed financial data
    """
    print("\nPreparing financial data...")

    # Create a copy to avoid modifying the original
    df = financial_df.copy()

    # Convert numeric columns that might be stored as strings
    # Added 'weightedAverageShsOutDil' here as it's used later for market cap
    numeric_columns = [
        'revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio',
        'researchAndDevelopmentExpenses', 'operatingExpenses', 'costAndExpenses',
        'interestIncome', 'interestExpense', 'ebitda', 'ebitdaratio',
        'operatingIncome', 'operatingIncomeRatio', 'incomeBeforeTax',
        'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio',
        'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil'
    ]

    # Make sure 'period' is not in the numeric columns list
    if 'period' in numeric_columns:
        numeric_columns.remove('period')

    # Try converting each column if it exists
    for col in numeric_columns:
        if col in df.columns:
            try:
                # Use errors='coerce' to turn unparseable values into NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # print(f"Converted {col} to numeric") # Keep this for debugging if needed
            except Exception as e:
                print(f"Could not convert {col}: {e}")

    # Select relevant columns for financial analysis
    # Keep period as a string - it contains values like "FY" (Fiscal Year)
    key_metrics = [
        'date', 'calendarYear', 'period', 'revenue', 'grossProfit', 'netIncome',
        'operatingIncome', 'eps', 'epsdiluted', 'researchAndDevelopmentExpenses',
        'operatingExpenses', 'grossProfitRatio', 'operatingIncomeRatio', 'netIncomeRatio',
        'weightedAverageShsOutDil' # Include shares outstanding for market cap calculation
    ]

    # Filter columns that exist in the dataframe
    existing_columns = [col for col in key_metrics if col in df.columns]
    df = df[existing_columns]

    # Sort by date
    df = df.sort_values('date')

    # Print sample of prepared data
    print(f"\nPrepared financial data with {len(existing_columns)} metrics")
    print("\nSample of prepared financial data:")
    print(df.head(2))

    return df

def merge_datasets(financial_df, stock_price_df):
    """
    Merge financial and stock price data with forward fill for financial metrics

    Parameters:
    financial_df (DataFrame): Prepared financial data
    stock_price_df (DataFrame): Stock price data

    Returns:
    DataFrame: Merged dataset
    """
    print("\nMerging datasets...")

    # Create copies of the dataframes
    fin_df = financial_df.copy()
    stock_df = stock_price_df.copy()

    # Ensure both dataframes have a 'date' column and are sorted
    if 'date' not in fin_df.columns or 'date' not in stock_df.columns:
        print("Error: 'date' column missing in one or both dataframes during merge preparation.")
        return None # Or raise an error

    fin_df = fin_df.sort_values('date')
    stock_df = stock_df.sort_values('date')

    # Filter stock data to only include dates on or after the first financial report
    min_financial_date = fin_df['date'].min()
    print(f"First financial report date: {min_financial_date}")

    # Filter stock data to only include dates on or after the first financial report
    stock_df_filtered = stock_df[stock_df['date'] >= min_financial_date].copy() # Use .copy() to avoid SettingWithCopyWarning
    print(f"Stock data filtered from {stock_df.shape[0]} to {stock_df_filtered.shape[0]} rows based on financial data start date")

    # Create a merged dataframe by first determining all unique dates
    # Use the date range from the first financial report to the last stock price date
    if not stock_df_filtered.empty:
        all_dates = pd.DataFrame({'date': pd.date_range(
            start=min_financial_date,
            end=stock_df_filtered['date'].max(),
            freq='D'
        )})
    else:
         print("Error: Filtered stock data is empty. Cannot create date range for merge.")
         return None


    # Merge financial data (lower frequency) with all dates
    print("Merging financial data with complete date range...")
    # Use a left merge on all_dates to keep all dates in the range
    merged_financial = pd.merge(all_dates, fin_df, on='date', how='left')

    # Forward fill financial data (each day gets most recent quarterly/annual data)
    print("Forward filling financial data...")
    # First sort to ensure correct fill order (already sorted, but good practice)
    merged_financial = merged_financial.sort_values('date')
    # Specify columns to ffill - exclude 'date', 'calendarYear', 'period'
    cols_to_ffill = [col for col in merged_financial.columns if col not in ['date', 'calendarYear', 'period']]
    merged_financial[cols_to_ffill] = merged_financial[cols_to_ffill].ffill()

    # Merge with stock price data (higher frequency)
    print("Merging with stock price data...")
    # Use a left merge on merged_financial to keep all dates in the range
    merged_df = pd.merge(merged_financial, stock_df_filtered, on='date', how='left')

    # Filter out days with no stock trading (weekends and holidays)
    # These will have NaN values in stock price columns after the left merge
    rows_before_dropna = merged_df.shape[0]
    # Drop rows where 'Close' is NaN - assuming 'Close' is always present on trading days
    if 'Close' in merged_df.columns:
        merged_df = merged_df.dropna(subset=['Close']).copy() # Use .copy() after dropna
        rows_after_dropna = merged_df.shape[0]
        print(f"Filtered out {rows_before_dropna - rows_after_dropna} rows with no stock trading (weekends/holidays)")
    else:
        print("Warning: 'Close' column not found after merge. Cannot filter non-trading days.")


    # Check for any duplicate dates
    duplicate_dates = merged_df[merged_df.duplicated('date', keep=False)] # Check for all duplicates, not just first
    if not duplicate_dates.empty:
        print(f"Warning: Found {len(duplicate_dates)} duplicate dates. Keeping first occurrence for each date.")
        merged_df = merged_df.drop_duplicates('date', keep='first').copy() # Use .copy() after drop_duplicates

    print(f"Final merged dataset has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")

    # Print a sample of the merged data
    print("\nSample of merged data:")
    cols_to_show = ['date', 'period', 'revenue', 'netIncome', 'eps', 'Open', 'Close', 'Volume']
    # Ensure columns exist before trying to select them
    cols_to_show = [col for col in cols_to_show if col in merged_df.columns]
    print(merged_df[cols_to_show].head())

    return merged_df

def add_derived_metrics(merged_df):
    """
    Add derived financial and technical metrics to the dataset

    Parameters:
    merged_df (DataFrame): Merged financial and stock data

    Returns:
    DataFrame: Enhanced dataset with derived metrics
    """
    print("\nAdding derived metrics...")
    df = merged_df.copy()

    # Ensure necessary columns exist before calculations
    required_cols = ['date', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns for derived metrics: {', '.join([col for col in required_cols if col not in df.columns])}")
        return df # Return original df if required columns are missing


    # Create a column to track the number of days since the last financial report
    # This logic assumes 'eps' changes only on financial report dates.
    # Find dates where 'eps' changes (indicating a new financial report)
    if 'eps' in df.columns:
        eps_changes = df['eps'].diff() != 0
        # The first row will be True for diff() != 0, but it's the start, not an update
        # We need the dates where the value is different from the previous day *and* not the very first row
        financial_update_dates = df.loc[eps_changes.shift(-1).fillna(False), 'date'].tolist()
        # Add the date of the first financial report if it's not already included
        if not df.empty and 'date' in df.columns and 'period' in df.columns:
             first_financial_date = df.loc[df['period'].first_valid_index(), 'date'] if df['period'].first_valid_index() is not None else None
             if first_financial_date and first_financial_date not in financial_update_dates:
                 financial_update_dates.insert(0, first_financial_date)

        df['days_since_financial_update'] = 0
        if financial_update_dates:
            # Iterate through update dates and calculate days since for subsequent rows
            last_update_date = financial_update_dates[0]
            for index, row in df.iterrows():
                if row['date'] in financial_update_dates:
                    last_update_date = row['date']
                df.loc[index, 'days_since_financial_update'] = (row['date'] - last_update_date).days
        else:
             print("Warning: Could not determine financial update dates. 'days_since_financial_update' will be 0.")


    # Add stock price changes
    df['price_change_pct'] = df['Close'].pct_change() * 100
    df['price_change'] = df['Close'].diff()

    # Technical indicators
    # 1. Moving Averages
    for window in [5, 10, 20, 50, 200]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()

    # 2. Relative Strength Index (RSI)
    # RSI calculation requires careful handling of initial NaNs
    window_rsi = 14
    if len(df) >= window_rsi:
        delta = df['Close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)

        # Calculate initial average gain and loss over the first window
        avg_gain_init = gain.iloc[:window_rsi].mean()
        avg_loss_init = loss.iloc[:window_rsi].mean()

        # Initialize lists for subsequent calculations
        avg_gains = [avg_gain_init]
        avg_losses = [avg_loss_init]

        # Calculate subsequent averages using smoothing formula
        for i in range(window_rsi, len(df)):
            avg_gain_next = (avg_gains[-1] * (window_rsi - 1) + gain.iloc[i]) / window_rsi
            avg_loss_next = (avg_losses[-1] * (window_rsi - 1) + loss.iloc[i]) / window_rsi
            avg_gains.append(avg_gain_next)
            avg_losses.append(avg_loss_next)

        # Create pandas Series from calculated averages
        avg_gain_series = pd.Series(avg_gains, index=df.index[window_rsi-1:])
        avg_loss_series = pd.Series(avg_losses, index=df.index[window_rsi-1:])

        # Calculate RS and RSI
        rs = avg_gain_series / avg_loss_series
        df['RSI'] = pd.Series(100 - (100 / (1 + rs)), index=df.index[window_rsi-1:])
    else:
        df['RSI'] = np.nan # Not enough data for RSI calculation
        print(f"Warning: Not enough data ({len(df)} rows) to calculate RSI (requires {window_rsi} rows).")


    # 3. Volatility measures
    df['returns'] = df['Close'].pct_change()
    for window in [5, 10, 20]:
        df[f'volatility_{window}d'] = df['returns'].rolling(window=window).std() * np.sqrt(window)

    # 4. Price-to-earnings ratio
    if 'eps' in df.columns and 'Close' in df.columns:
        # Avoid division by zero or NaN in eps
        df['PE_ratio'] = df['Close'] / df['eps'].replace(0, np.nan) # Replace 0 eps with NaN
    else:
        print("Warning: 'eps' or 'Close' column not found. Cannot calculate PE_ratio.")
        if 'PE_ratio' not in df.columns:
             df['PE_ratio'] = np.nan # Ensure column exists even if not calculated


    # 5. Volume metrics
    if 'Volume' in df.columns:
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
    else:
        print("Warning: 'Volume' column not found. Cannot calculate volume metrics.")
        if 'volume_change' not in df.columns: df['volume_change'] = np.nan
        if 'volume_ma_5' not in df.columns: df['volume_ma_5'] = np.nan


    # 6. Price momentum
    if 'Close' in df.columns:
        for window in [5, 10, 20]:
            df[f'momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1
    else:
        print("Warning: 'Close' column not found. Cannot calculate momentum.")
        for window in [5, 10, 20]:
             if f'momentum_{window}d' not in df.columns: df[f'momentum_{window}d'] = np.nan


    # 7. Stock-to-Revenue ratio (Price-to-Sales)
    if 'revenue' in df.columns and 'weightedAverageShsOutDil' in df.columns and 'Close' in df.columns:
        # Calculate market cap (price * shares outstanding)
        # Avoid division by zero or NaN in shares outstanding or revenue
        df['market_cap'] = df['Close'] * df['weightedAverageShsOutDil'].replace(0, np.nan)
        # Calculate price-to-sales ratio
        df['price_to_sales'] = df['market_cap'] / df['revenue'].replace(0, np.nan)
    else:
        print("Warning: Missing 'revenue', 'weightedAverageShsOutDil', or 'Close' column. Cannot calculate price-to-sales.")
        if 'market_cap' not in df.columns: df['market_cap'] = np.nan
        if 'price_to_sales' not in df.columns: df['price_to_sales'] = np.nan


    # Handle any potential NaN values from calculations
    # (first few rows often have NaNs due to rolling calculations)
    # Use bfill (backward fill) to fill initial NaNs with the next valid observation
    # Only apply to numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns # Use np.number for broader numeric type check
    if not numeric_cols.empty:
        # Apply bfill only to a copy of the numeric columns slice to avoid SettingWithCopyWarning
        df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
        # After bfill, there might still be NaNs at the end if the last values are NaN
        # Consider adding a ffill here if appropriate, or leaving as NaN
        # df[numeric_cols] = df[numeric_cols].fillna(method='ffill') # Optional: forward fill remaining NaNs


    print(f"Added derived metrics. Dataset now has {df.shape[1]} columns")

    # Print a sample of the derived metrics
    print("\nSample of derived metrics:")
    derived_cols = ['date', 'Close', 'MA_20', 'RSI', 'volatility_20d', 'PE_ratio', 'momentum_20d', 'days_since_financial_update']
    # Ensure columns exist before trying to select them
    derived_cols = [col for col in derived_cols if col in df.columns]
    print(df[derived_cols].head())

    return df

def perform_basic_eda(df):
    """
    Perform basic exploratory data analysis

    Parameters:
    df (DataFrame): The processed dataset

    Returns:
    DataFrame: The dataset (unchanged)
    """
    print("\n==== Basic Exploratory Data Analysis ====")

    # Check for missing values
    missing_values = df.isnull().sum()
    # Filter to show only columns with missing values
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print("\nMissing values summary:")
        print(missing_values)
    else:
        print("\nNo missing values found in the dataset.")

    # Descriptive statistics for key columns
    key_cols = ['Close', 'Volume', 'eps', 'netIncome', 'revenue', 'RSI', 'PE_ratio', 'days_since_financial_update']
    # Ensure columns exist before trying to select them
    key_cols = [col for col in key_cols if col in df.columns]

    if key_cols:
        print("\nDescriptive statistics for key metrics:")
        print(df[key_cols].describe().T)
    else:
        print("\nNo key metrics found for descriptive statistics.")

    # Correlations with stock price - only use numeric columns
    print("\nTop correlations with stock price (Close):")
    # Get only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=[np.number]) # Use np.number
    if 'Close' in numeric_df.columns and len(numeric_df.columns) > 1: # Need at least two numeric columns for correlation
        correlations = numeric_df.corr()['Close'].sort_values(ascending=False)
        # Exclude self-correlation (Close with Close)
        correlations = correlations[correlations.index != 'Close']
        print(correlations.head(10))
        print("\nBottom correlations with stock price (Close):")
        print(correlations.tail(5))
    elif 'Close' not in numeric_df.columns:
        print("Warning: 'Close' column not found in numeric columns for correlation analysis.")
    else:
        print("Warning: Only one numeric column found ('Close'). Cannot perform correlation analysis.")


    # Analyze price movements over time
    if 'price_change_pct' in df.columns:
        price_changes = df['price_change_pct'].describe()
        print("\nStock price daily percentage change statistics:")
        print(price_changes)
    else:
        print("\n'price_change_pct' column not found. Cannot provide daily percentage change statistics.")

    # Financial periods analysis
    if 'period' in df.columns and 'calendarYear' in df.columns and 'date' in df.columns:
        # Get unique financial reporting dates and corresponding financial data
        # Use drop_duplicates on a subset of columns that define a unique report
        financial_report_dates = df.drop_duplicates(subset=['calendarYear', 'period'], keep='first')['date'].tolist()

        if financial_report_dates:
            # Filter the main DataFrame to get the rows corresponding to these dates
            financial_summary_df = df[df['date'].isin(financial_report_dates)].sort_values('date')

            print("\nFinancial reporting periods in the dataset:")
            summary_cols = ['date', 'calendarYear', 'period', 'revenue', 'netIncome', 'eps']
            # Ensure columns exist before trying to select them
            summary_cols = [col for col in summary_cols if col in financial_summary_df.columns]
            if summary_cols:
                 print(financial_summary_df[summary_cols])
            else:
                 print("Warning: Key financial summary columns not found.")
        else:
             print("\nNo unique financial reporting periods found in the dataset.")

    else:
        print("\nMissing 'period', 'calendarYear', or 'date' column for financial periods analysis.")


    return df

def visualize_key_metrics(df, output_path=None):
    """
    Create visualizations for key metrics

    Parameters:
    df (DataFrame): The processed dataset
    output_path (str, optional): Path to save visualizations
    """
    print("\n==== Creating Visualizations ====")

    # Set style
    plt.style.use('ggplot')

    # Ensure 'date' column is datetime and sort the DataFrame by date
    if 'date' not in df.columns:
        print("Error: 'date' column missing for visualizations.")
        return
    df = df.sort_values('date')

    # Use the global stock ticker and title for chart titles
    stock_ticker = GLOBAL_STOCK_TICKER
    base_title = GLOBAL_STOCK_TITLE


    # 1. Stock Price Over Time with Moving Averages
    if 'Close' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['Close'], label='Close Price', linewidth=1)

        # Add moving averages if they exist
        for ma in [50, 200]:
            if f'MA_{ma}' in df.columns:
                plt.plot(df['date'], df[f'MA_{ma}'], label=f'{ma}-day MA', linewidth=1.5)

        plt.title(f'{stock_ticker} {base_title} - Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if output_path:
            try:
                # Use the ticker in the output filename
                plt.savefig(f"{output_path.split('.')[0]}_{stock_ticker}_price_chart.png")
                print(f"Created {stock_ticker} stock price chart with moving averages")
            except Exception as e:
                print(f"Error saving price chart for {stock_ticker}: {e}")

        plt.close()
    else:
        print(f"Warning: 'Close' column not found for {stock_ticker}. Skipping price chart.")


    # 2. RSI indicator
    if 'RSI' in df.columns and 'date' in df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(df['date'], df['RSI'], color='purple', linewidth=1)
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.3, label='Overbought (70)')
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.3, label='Oversold (30)')
        plt.title(f'{stock_ticker} {base_title} - Relative Strength Index (RSI)')
        plt.ylabel('RSI')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if output_path:
            try:
                # Use the ticker in the output filename
                plt.savefig(f"{output_path.split('.')[0]}_{stock_ticker}_rsi_chart.png")
                print(f"Created {stock_ticker} RSI chart")
            except Exception as e:
                print(f"Error saving RSI chart for {stock_ticker}: {e}")
        plt.close()
    else:
        print(f"Warning: 'RSI' or 'date' column not found for {stock_ticker}. Skipping RSI chart.")


    # 3. Financial metrics over time
    if 'revenue' in df.columns and 'netIncome' in df.columns and 'date' in df.columns and 'calendarYear' in df.columns and 'period' in df.columns:
        # Get unique financial reporting dates
        financial_dates = df.drop_duplicates(['calendarYear', 'period'])['date'].tolist()

        if financial_dates:
            # Filter data for those dates and sort
            financial_data = df[df['date'].isin(financial_dates)].sort_values('date')

            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Plot revenue on primary y-axis
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Revenue ($)', color='tab:blue')
            ax1.plot(financial_data['date'], financial_data['revenue'], color='tab:blue', marker='o', linestyle='-') # Added linestyle
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.grid(True, axis='y', alpha=0.5) # Add grid for primary axis

            # Create secondary y-axis for net income
            ax2 = ax1.twinx()
            ax2.set_ylabel('Net Income ($)', color='tab:red')
            ax2.plot(financial_data['date'], financial_data['netIncome'], color='tab:red', marker='s', linestyle='-') # Added linestyle
            ax2.tick_params(axis='y', labelcolor='tab:red')
            ax2.grid(True, axis='y', alpha=0.5) # Add grid for secondary axis

            plt.title(f'{stock_ticker} {base_title} - Revenue and Net Income')
            # Add a legend that combines elements from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, ['Revenue', 'Net Income'], loc='upper left') # Manually set labels for clarity

            plt.tight_layout()

            if output_path:
                try:
                    # Use the ticker in the output filename
                    plt.savefig(f"{output_path.split('.')[0]}_{stock_ticker}_financial_metrics.png")
                    print(f"Created {stock_ticker} financial metrics chart")
                except Exception as e:
                    print(f"Error saving financial metrics chart for {stock_ticker}: {e}")
            plt.close()
        else:
            print(f"Warning: No unique financial reporting dates found for {stock_ticker}. Skipping financial metrics chart.")
    else:
        print(f"Warning: Missing financial columns for {stock_ticker} ('revenue', 'netIncome', 'date', 'calendarYear', 'period'). Skipping financial metrics chart.")


    # 4. Volume over time
    if 'Volume' in df.columns and 'date' in df.columns:
        plt.figure(figsize=(12, 4))
        plt.bar(df['date'], df['Volume'], color='blue', alpha=0.6, width=1) # Use width=1 for daily data
        plt.title(f'{stock_ticker} {base_title} - Trading Volume')
        plt.ylabel('Volume')
        plt.xlabel('Date')
        plt.grid(True, axis='y', alpha=0.5)
        plt.tight_layout()

        if output_path:
            try:
                # Use the ticker in the output filename
                plt.savefig(f"{output_path.split('.')[0]}_{stock_ticker}_volume_chart.png")
                print(f"Created {stock_ticker} volume chart")
            except Exception as e:
                print(f"Error saving volume chart for {stock_ticker}: {e}")
        plt.close()
    else:
        print(f"Warning: 'Volume' or 'date' column not found for {stock_ticker}. Skipping volume chart.")


    # 5. Correlation heatmap for key metrics
    key_metrics_for_heatmap = ['Close', 'Volume', 'revenue', 'netIncome', 'eps', 'RSI',
                     'volatility_20d', 'MA_50', 'momentum_20d', 'PE_ratio', 'price_to_sales', 'days_since_financial_update']
    # Ensure columns exist and are numeric before including in heatmap
    key_metrics_for_heatmap = [col for col in key_metrics_for_heatmap if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if len(key_metrics_for_heatmap) > 1: # Need at least two columns for correlation
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[key_metrics_for_heatmap].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f") # Added fmt for better annotation
        plt.title(f'{stock_ticker} {base_title} - Correlation Matrix of Key Metrics')
        plt.tight_layout()

        if output_path:
            try:
                # Use the ticker in the output filename
                plt.savefig(f"{output_path.split('.')[0]}_{stock_ticker}_correlation_heatmap.png")
                print(f"Created {stock_ticker} correlation heatmap")
            except Exception as e:
                print(f"Error saving correlation heatmap for {stock_ticker}: {e}")
        plt.close()
    else:
        print(f"Warning: Not enough numeric key metrics found for correlation heatmap for {stock_ticker}.")


    print("Visualization complete")

def main(financial_path, stock_path, output_path=None, create_visualizations=True):
    """
    Main function to process and merge financial and stock price data

    Parameters:
    financial_path (str): Path to financial statements CSV
    stock_path (str): Path to stock price CSV
    output_path (str, optional): Path to save the merged dataset
    create_visualizations (bool): Whether to create visualization charts

    Returns:
    DataFrame: The final merged and enhanced dataset or None if an error occurred
    """
    try:
        # Load data
        financial_df, stock_price_df = load_data(financial_path, stock_path)

        # Check if data loading was successful
        if financial_df is None or stock_price_df is None or 'date' not in financial_df.columns or 'date' not in stock_price_df.columns:
             print("Error loading data. Exiting.")
             return None

        # Prepare financial data
        financial_clean = prepare_financial_data(financial_df)
        # Check if financial data preparation was successful and has required columns
        if financial_clean is None or 'date' not in financial_clean.columns:
             print("Error preparing financial data. Exiting.")
             return None


        # Merge datasets
        merged_df = merge_datasets(financial_clean, stock_price_df)
        # Check if merging was successful
        if merged_df is None or merged_df.empty or 'date' not in merged_df.columns:
             print("Error merging datasets or merged dataset is empty. Exiting.")
             return None


        # Add derived metrics
        enhanced_df = add_derived_metrics(merged_df)
        # Check if adding derived metrics was successful
        if enhanced_df is None or enhanced_df.empty:
             print("Error adding derived metrics. Exiting.")
             return None


        # Perform basic EDA
        # EDA function doesn't modify the DataFrame, so no need to reassign
        perform_basic_eda(enhanced_df)

        # Create visualizations if requested
        if create_visualizations:
            visualize_key_metrics(enhanced_df, output_path)

        # Save the processed dataset if output path is provided
        if output_path:
            try:
                # Use the global ticker in the output filename
                output_filename_with_ticker = f"{output_path.split('.')[0]}_{GLOBAL_STOCK_TICKER}.csv"
                enhanced_df.to_csv(output_filename_with_ticker, index=False)
                print(f"\nMerged dataset saved to {output_filename_with_ticker}")
            except Exception as e:
                print(f"Error saving merged dataset to {output_path}: {e}")


        print("\nData processing complete!")
        return enhanced_df

    except Exception as e:
        print(f"\nAn unexpected error occurred during data processing: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    # --- Update file paths and global variables here ---
    # Set the global ticker and title
    GLOBAL_STOCK_TICKER = "VST" # Example: Change to "AAPL" for Apple
    GLOBAL_STOCK_TITLE = "Stock Analysis" # Can keep generic or make specific

    # Construct file paths using the global ticker
    # Assuming your financial data file naming convention might also include the ticker
    # Adjust these paths based on your actual file structure and naming
    financial_csv = f"ML_Final_Project/Lusofona/allFinData/{GLOBAL_STOCK_TICKER}_all_financial_data.csv"
    stock_price_csv = f"ML_Final_Project/Lusofona/allFinData/{GLOBAL_STOCK_TICKER}_historical_data.csv"
    output_csv = f"ML_Final_Project/Lusofona/allFinData/{GLOBAL_STOCK_TICKER}_merged_dataset.csv" # Output file name will now include the ticker

    print(f"Starting EDA for {GLOBAL_STOCK_TICKER} with financial data from: {financial_csv}")
    print(f"and stock price data from: {stock_price_csv}")

    result_df = main(financial_csv, stock_price_csv, output_csv)

    if result_df is not None:
        print(f"\nFinal processed DataFrame for {GLOBAL_STOCK_TICKER} is available.")
        # You can add further analysis or display here if needed
        # print(result_df.info())
        # print(result_df.head())
    else:
        print(f"\nData processing failed for {GLOBAL_STOCK_TICKER}.")

