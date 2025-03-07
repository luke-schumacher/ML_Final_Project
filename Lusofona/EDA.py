import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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
    financial_df = pd.read_csv(financial_csv_path)
    
    # Load stock price data
    print("Loading stock price data...")
    stock_price_df = pd.read_csv(stock_price_csv_path)
    
    # Convert date columns to datetime format
    financial_df['Date'] = pd.to_datetime(financial_df['Date'])
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    
    # Display information about the loaded data
    print(f"Financial data shape: {financial_df.shape}")
    print(f"Financial data date range: {financial_df['Date'].min()} to {financial_df['Date'].max()}")
    print(f"Financial data columns: {financial_df.columns.tolist()}")
    print(f"Financial data period types: {financial_df['period'].unique().tolist()}")
    
    print(f"\nStock price data shape: {stock_price_df.shape}")
    print(f"Stock price data date range: {stock_price_df['Date'].min()} to {stock_price_df['Date'].max()}")
    
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
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted {col} to numeric")
            except Exception as e:
                print(f"Could not convert {col}: {e}")
    
    # Select relevant columns for financial analysis
    # Keep period as a string - it contains values like "FY" (Fiscal Year)
    key_metrics = [
        'Date', 'calendarYear', 'period', 'revenue', 'grossProfit', 'netIncome', 
        'operatingIncome', 'eps', 'epsdiluted', 'researchAndDevelopmentExpenses',
        'operatingExpenses', 'grossProfitRatio', 'operatingIncomeRatio', 'netIncomeRatio'
    ]
    
    # Filter columns that exist in the dataframe
    existing_columns = [col for col in key_metrics if col in df.columns]
    df = df[existing_columns]
    
    # Sort by date
    df = df.sort_values('Date')
    
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
    
    # Filter stock data to only include dates after the first financial report
    min_financial_date = fin_df['Date'].min()
    print(f"First financial report date: {min_financial_date}")
    
    # Filter stock data to only include dates on or after the first financial report
    stock_df_filtered = stock_df[stock_df['Date'] >= min_financial_date]
    print(f"Stock data filtered from {stock_df.shape[0]} to {stock_df_filtered.shape[0]} rows")
    
    # Create a merged dataframe by first determining all unique dates
    all_dates = pd.DataFrame({'Date': pd.date_range(
        start=min_financial_date,
        end=stock_df['Date'].max(),
        freq='D'
    )})
    
    # Merge financial data (lower frequency) with all dates
    print("Merging financial data with date range...")
    merged_financial = pd.merge(all_dates, fin_df, on='Date', how='left')
    
    # Forward fill financial data (each day gets most recent quarterly/annual data)
    print("Forward filling financial data...")
    # First sort to ensure correct fill order
    merged_financial = merged_financial.sort_values('Date')
    merged_financial = merged_financial.ffill()
    
    # Merge with stock price data (higher frequency)
    print("Merging with stock price data...")
    merged_df = pd.merge(merged_financial, stock_df, on='Date', how='left')
    
    # Filter out weekends and holidays (when stock market is closed)
    rows_before = merged_df.shape[0]
    merged_df = merged_df.dropna(subset=['Close'])
    rows_after = merged_df.shape[0]
    print(f"Filtered out {rows_before - rows_after} rows with no stock trading (weekends/holidays)")
    
    # Check for any duplicate dates
    duplicate_dates = merged_df[merged_df.duplicated('Date')]
    if not duplicate_dates.empty:
        print(f"Warning: Found {len(duplicate_dates)} duplicate dates. Keeping first occurrence.")
        merged_df = merged_df.drop_duplicates('Date', keep='first')
    
    print(f"Final merged dataset has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
    
    # Print a sample of the merged data
    print("\nSample of merged data:")
    cols_to_show = ['Date', 'period', 'revenue', 'netIncome', 'eps', 'Open', 'Close', 'Volume']
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
    
    # Create a column to track the number of days since the last financial report
    df['days_since_financial_update'] = 0
    # Group by financial reporting periods (where financial data changes)
    for period_start in df[df['eps'].diff() != 0]['Date'].tolist():
        mask = (df['Date'] >= period_start)
        # Calculate days since this financial update
        df.loc[mask, 'days_since_financial_update'] = (df.loc[mask, 'Date'] - period_start).dt.days
    
    # Add stock price changes
    df['price_change_pct'] = df['Close'].pct_change() * 100
    df['price_change'] = df['Close'].diff()
    
    # Technical indicators
    # 1. Moving Averages
    for window in [5, 10, 20, 50, 200]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        
    # 2. Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. Volatility measures
    df['returns'] = df['Close'].pct_change()
    for window in [5, 10, 20]:
        df[f'volatility_{window}d'] = df['returns'].rolling(window=window).std() * np.sqrt(window)
    
    # 4. Price-to-earnings ratio
    if 'eps' in df.columns:
        df['PE_ratio'] = df['Close'] / df['eps']
    
    # 5. Volume metrics
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
    
    # 6. Price momentum
    for window in [5, 10, 20]:
        df[f'momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1
    
    # 7. Stock-to-Revenue ratio (Price-to-Sales)
    if 'revenue' in df.columns and 'weightedAverageShsOutDil' in df.columns:
        # Calculate market cap (price * shares outstanding)
        df['market_cap'] = df['Close'] * df['weightedAverageShsOutDil']
        # Calculate price-to-sales ratio
        df['price_to_sales'] = df['market_cap'] / df['revenue']
    
    # Handle any potential NaN values from calculations
    # (first few rows often have NaNs due to rolling calculations)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
    
    print(f"Added derived metrics. Dataset now has {df.shape[1]} columns")
    
    # Print a sample of the derived metrics
    print("\nSample of derived metrics:")
    derived_cols = ['Date', 'Close', 'MA_20', 'RSI', 'volatility_20d', 'PE_ratio', 'momentum_20d']
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
    if missing_values.any():
        print("\nMissing values summary:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found in the dataset.")
    
    # Descriptive statistics for key columns
    key_cols = ['Close', 'Volume', 'eps', 'netIncome', 'revenue', 'RSI', 'PE_ratio']
    key_cols = [col for col in key_cols if col in df.columns]
    
    print("\nDescriptive statistics for key metrics:")
    print(df[key_cols].describe().T)
    
    # Correlations with stock price - only use numeric columns
    print("\nTop correlations with stock price (Close):")
    # Get only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if 'Close' in numeric_df.columns:
        correlations = numeric_df.corr()['Close'].sort_values(ascending=False)
        print(correlations.head(10))
        print("\nBottom correlations with stock price (Close):")
        print(correlations.tail(5))
    else:
        print("Warning: 'Close' column not found in numeric columns")
    
    # Analyze price movements over time
    if 'price_change_pct' in df.columns:
        price_changes = df['price_change_pct'].describe()
        print("\nStock price daily percentage change statistics:")
        print(price_changes)
    
    # Financial periods analysis
    if 'period' in df.columns:
        period_data = df.drop_duplicates(['calendarYear', 'period'])
        print("\nFinancial reporting periods in the dataset:")
        if len(period_data) > 0:
            financial_summary = period_data[['Date', 'calendarYear', 'period', 'revenue', 'netIncome', 'eps']]
            financial_summary = financial_summary[[col for col in financial_summary.columns if col in df.columns]]
            print(financial_summary)
    
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
    
    # 1. Stock Price Over Time with Moving Averages
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', linewidth=1)
    
    # Add moving averages if they exist
    for ma in [50, 200]:
        if f'MA_{ma}' in df.columns:
            plt.plot(df['Date'], df[f'MA_{ma}'], label=f'{ma}-day MA', linewidth=1.5)
    
    plt.title('NVIDIA Stock Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(f"{output_path.split('.')[0]}_price_chart.png")
    plt.close()
    print("Created stock price chart with moving averages")
    
    # 2. RSI indicator
    if 'RSI' in df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(df['Date'], df['RSI'], color='purple', linewidth=1)
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
        plt.title('Relative Strength Index (RSI)')
        plt.ylabel('RSI')
        plt.xlabel('Date')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(f"{output_path.split('.')[0]}_rsi_chart.png")
        plt.close()
        print("Created RSI chart")
    
    # 3. Financial metrics over time
    if 'revenue' in df.columns and 'netIncome' in df.columns:
        # Get unique financial reporting dates
        financial_dates = df.drop_duplicates(['calendarYear', 'period'])['Date']
        
        # Filter data for those dates
        financial_data = df[df['Date'].isin(financial_dates)].sort_values('Date')
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot revenue on primary y-axis
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Revenue ($)', color='tab:blue')
        ax1.plot(financial_data['Date'], financial_data['revenue'], color='tab:blue', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Create secondary y-axis for net income
        ax2 = ax1.twinx()
        ax2.set_ylabel('Net Income ($)', color='tab:red')
        ax2.plot(financial_data['Date'], financial_data['netIncome'], color='tab:red', marker='s')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title('NVIDIA Revenue and Net Income')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(f"{output_path.split('.')[0]}_financial_metrics.png")
        plt.close()
        print("Created financial metrics chart")
    
    # 4. Volume over time
    plt.figure(figsize=(12, 4))
    plt.bar(df['Date'], df['Volume'], color='blue', alpha=0.6)
    plt.title('Trading Volume')
    plt.ylabel('Volume')
    plt.xlabel('Date')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(f"{output_path.split('.')[0]}_volume_chart.png")
    plt.close()
    print("Created volume chart")
    
    # 5. Correlation heatmap for key metrics
    key_metrics = ['Close', 'Volume', 'revenue', 'netIncome', 'eps', 'RSI', 
                  'volatility_20d', 'MA_50', 'momentum_20d']
    key_metrics = [col for col in key_metrics if col in df.columns]
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[key_metrics].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Key Metrics')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(f"{output_path.split('.')[0]}_correlation_heatmap.png")
    plt.close()
    print("Created correlation heatmap")
    
    print("Visualization complete")

def main(financial_path, stock_path, output_path=None, create_visualizations=True):
    """
    Main function to process and merge NVIDIA financial and stock price data
    
    Parameters:
    financial_path (str): Path to financial statements CSV
    stock_path (str): Path to stock price CSV
    output_path (str, optional): Path to save the merged dataset
    create_visualizations (bool): Whether to create visualization charts
    
    Returns:
    DataFrame: The final merged and enhanced dataset
    """
    try:
        # Load data
        financial_df, stock_price_df = load_data(financial_path, stock_path)
        
        # Prepare financial data
        financial_clean = prepare_financial_data(financial_df)
        
        # Merge datasets
        merged_df = merge_datasets(financial_clean, stock_price_df)
        
        # Add derived metrics
        enhanced_df = add_derived_metrics(merged_df)
        
        # Perform basic EDA
        enhanced_df = perform_basic_eda(enhanced_df)
        
        # Create visualizations if requested
        if create_visualizations:
            visualize_key_metrics(enhanced_df, output_path)
        
        # Save the processed dataset if output path is provided
        if output_path:
            enhanced_df.to_csv(output_path, index=False)
            print(f"\nMerged dataset saved to {output_path}")
        
        print("\nData processing complete!")
        return enhanced_df
        
    except Exception as e:
        print(f"\nError in data processing: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    financial_csv = "ML_Final_Project/Lusofona/NVDA_all_financial_data.csv"
    stock_price_csv = "ML_Final_Project/Lusofona/NVDA_historical_data.csv"
    output_csv = "ML_Final_Project/Lusofona/nvda_merged_dataset.csv"
    
    result_df = main(financial_csv, stock_price_csv, output_csv)