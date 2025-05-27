import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
import requests
import time
import math
from datetime import timedelta, datetime
import os
import traceback
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Sequential

# --- Configuration and Global Variables ---
# IMPORTANT: Your actual Financial Modeling Prep API Key
FMP_API_KEY = "INSERT API KEY HERE"  # Replace with your actual API key
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3/"

# Define the types of financial statements to fetch
FMP_STATEMENT_TYPES = [
    "income-statement",
    "balance-sheet-statement",
    "cash-flow-statement",
    "financial-ratios"
]

# --- Helper Functions for Transformer Model (Ensuring consistency) ---

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Implements a single Transformer encoder block.
    Args:
        inputs: Input tensor to the encoder.
        head_size: Dimensionality of the attention heads.
        num_heads: Number of attention heads.
        ff_dim: Hidden layer size of the feed-forward network.
        dropout: Dropout rate.
    Returns:
        Output tensor of the encoder block.
    """
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, mlp_dropout, dropout):
    """
    Builds the Transformer model for time series prediction.
    Args:
        input_shape: Shape of the input sequences (sequence_length, num_features).
        head_size: Dimensionality of the attention heads.
        num_heads: Number of attention heads.
        ff_dim: Hidden layer size of the feed-forward network.
        num_transformer_blocks: Number of Transformer encoder blocks.
        mlp_units: List of hidden units for the MLP head.
        mlp_dropout: Dropout rate for the MLP head.
        dropout: Dropout rate for Transformer blocks.
    Returns:
        A Keras Model instance.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Use GlobalAveragePooling1D with channels_last if your data is (batch, sequence, features)
    # which is the default for Keras Conv1D and expected for this model setup.
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x) # Output is a single value (predicted close price)
    return tf.keras.Model(inputs, outputs)

def create_sequences(data, sequence_length):
    """
    Creates sequences of features (X) and corresponding target (y) for the Transformer model.
    The 'close' price (assumed to be the first column) is the target.
    All other columns are features.
    Args:
        data (np.array): Scaled data where the first column is the 'close' price.
        sequence_length (int): The length of input sequences.
    Returns:
        tuple: (X, y) where X are feature sequences and y are target values.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Features (X) will be all columns EXCEPT the first one ('close') for the current sequence
        X.append(data[i:(i + sequence_length), 1:])
        # Target (y) will be the 'close' price of the next step (sequence_length steps ahead)
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

# --- 1. Data Download Functions (from StockDataDownloader.py and FinStatementDownloader.py) ---

def flatten_multiindex_columns(df):
    """
    Flattens MultiIndex columns to a single level.
    Handles cases where the second level might be an empty string.
    """
    if isinstance(df.columns, pd.MultiIndex):
        new_columns = []
        for col in df.columns:
            if col[1] == '': # If the second level is empty (e.g., ('Date', ''))
                new_columns.append(col[0])
            else:
                # Join non-empty parts of the tuple
                new_columns.append('_'.join(str(c) for c in col if str(c) != ''))
        df.columns = new_columns
    return df

@st.cache_data(ttl=3600) # Cache data for 1 hour
def download_stock_data(ticker, use_local_data=False, local_data_path="", max_retries=5, retry_delay=5):
    """
    Downloads historical stock data from Yahoo Finance or loads from local CSV.
    Ensures 'date' column is standardized and datetime, flattens MultiIndex columns,
    and renames stock price columns to generic lowercase names.
    """
    stock_data_df = None
    if use_local_data:
        local_file_path = os.path.join(local_data_path, f"{ticker}_historical_data.csv")
        st.info(f"Attempting to load historical stock data for {ticker} from local file: {local_file_path}...")
        try:
            if os.path.exists(local_file_path):
                stock_data_df = pd.read_csv(local_file_path, header=0)
                st.success(f"Successfully loaded historical stock data for {ticker} from local file.")
                
                # Flatten MultiIndex columns if present
                stock_data_df = flatten_multiindex_columns(stock_data_df)

                # Standardize column names (lowercase, strip whitespace)
                stock_data_df.columns = stock_data_df.columns.str.lower().str.strip()

                # Rename specific columns that might have ticker suffixes (e.g., 'close_aapl' to 'close')
                standard_stock_cols = {
                    'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
                    'adj close': 'adj close', 'volume': 'volume'
                }
                rename_map = {}
                for col in stock_data_df.columns:
                    for standard_name, expected_name in standard_stock_cols.items():
                        if col.startswith(standard_name) and col.endswith(ticker.lower()):
                            rename_map[col] = expected_name
                        elif col == standard_name: # Also handle if it's already the standard name
                            rename_map[col] = expected_name
                
                if rename_map:
                    stock_data_df.rename(columns=rename_map, inplace=True)
                    st.info(f"Standardized stock price columns: {rename_map}")

                # Ensure 'date' column is present and converted to datetime
                if 'date' in stock_data_df.columns:
                    stock_data_df['date'] = pd.to_datetime(stock_data_df['date'])
                else:
                    st.error(f"Local file {local_file_path} is missing a 'date' column after standardization.")
                    stock_data_df = None # Invalidate if date column is missing
            else:
                st.warning(f"Local historical stock data file not found: {local_file_path}. Attempting to download from Yahoo Finance.")
        except Exception as e:
            st.error(f"Error loading local historical stock data file {local_file_path}: {e}. Attempting to download from Yahoo Finance.")
            stock_data_df = None

    if stock_data_df is None or stock_data_df.empty: # Fallback to Yahoo Finance if local loading failed or not requested
        st.info(f"Attempting to download historical stock data for {ticker} from Yahoo Finance...")
        for attempt in range(max_retries):
            try:
                stock_data = yf.download(ticker, period="max")
                if stock_data is not None and not isinstance(stock_data, str) and not stock_data.empty:
                    st.success(f"Successfully downloaded data for {ticker} after {attempt + 1} attempts.")
                    stock_data_df = stock_data.reset_index()
                    
                    # Flatten MultiIndex columns (yfinance often creates them)
                    stock_data_df = flatten_multiindex_columns(stock_data_df)

                    # Standardize column names (lowercase, strip whitespace)
                    stock_data_df.columns = stock_data_df.columns.str.lower().str.strip()

                    # Rename specific columns that might have ticker suffixes
                    standard_stock_cols = {
                        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
                        'adj close': 'adj close', 'volume': 'volume'
                    }
                    rename_map = {}
                    for col in stock_data_df.columns:
                        for standard_name, expected_name in standard_stock_cols.items():
                            if col.startswith(standard_name) and col.endswith(ticker.lower()):
                                rename_map[col] = expected_name
                            elif col == standard_name:
                                rename_map[col] = expected_name
                    if rename_map:
                        stock_data_df.rename(columns=rename_map, inplace=True)
                        st.info(f"Standardized stock price columns: {rename_map}")

                    stock_data_df['date'] = pd.to_datetime(stock_data_df['date'])
                    return stock_data_df
                elif isinstance(stock_data, str):
                    st.error(f"Error downloading data for {ticker}: {stock_data}")
                    return None
            except yf.YFError as e:
                if "Too Many Requests" in str(e):
                    st.warning(f"Rate limit hit for {ticker}. Retrying in {retry_delay} seconds (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    st.error(f"An unexpected error occurred during Yahoo Finance download for {ticker}: {e}")
                    return None
            except Exception as e:
                st.error(f"An unexpected error occurred during Yahoo Finance download for {ticker}: {e}")
                return None
        st.error(f"Failed to download data for {ticker} after {max_retries} attempts.")
        return None
    return stock_data_df

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_financial_data(symbol, api_key, statement_types, use_local_data=False, local_data_path=""):
    """
    Fetches financial statements and ratios from Financial Modeling Prep API or loads from local CSV.
    Ensures 'date' and 'fillingDate' columns are standardized and datetime, and flattens MultiIndex columns.
    """
    all_data = pd.DataFrame()
    if use_local_data:
        local_file_path = os.path.join(local_data_path, f"{symbol}_all_financial_data.csv")
        st.info(f"Attempting to load financial data for {symbol} from local file: {local_file_path}...")
        try:
            if os.path.exists(local_file_path):
                all_data = pd.read_csv(local_file_path, header=0)
                st.success(f"Successfully loaded financial data for {symbol} from local file.")
                
                # Flatten MultiIndex columns if present
                all_data = flatten_multiindex_columns(all_data)

                # Standardize column names (lowercase, strip whitespace)
                all_data.columns = all_data.columns.str.lower().str.strip()

                # Convert date columns to datetime
                if 'date' in all_data.columns:
                    all_data['date'] = pd.to_datetime(all_data['date'])
                else:
                    st.error(f"Local financial file {local_file_path} is missing a 'date' column after standardization.")
                    all_data = pd.DataFrame() # Invalidate to trigger API fetch
                
                if 'fillingdate' in all_data.columns: # Note: 'fillingDate' becomes 'fillingdate'
                    all_data['fillingdate'] = pd.to_datetime(all_data['fillingdate'])

                return all_data
            else:
                st.warning(f"Local financial data file not found: {local_file_path}. Attempting to fetch from FMP API.")
        except Exception as e:
            st.error(f"Error loading local financial data file {local_file_path}: {e}. Attempting to fetch from FMP API.")
            all_data = pd.DataFrame() # Reset all_data to empty to trigger API fetch

    if all_data.empty: # Fallback to FMP API if local loading failed or not requested
        st.info(f"Attempting to fetch financial data for {symbol} from FMP API...")
        if not api_key or api_key == "YOUR_FINANCIAL_MODELING_PREP_API_KEY":
            st.error("Financial Modeling Prep API Key is missing or invalid. Cannot fetch financial data.")
            return None

        for statement_type in statement_types:
            url = f"{FMP_BASE_URL}{statement_type}/{symbol}?apikey={api_key}&period=10"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    financial_data = response.json()
                    if financial_data:
                        df = pd.DataFrame(financial_data)
                        df['Statement_Type'] = statement_type # Keep original Statement_Type for reference
                        
                        # Flatten MultiIndex columns if present (unlikely for FMP, but safe)
                        df = flatten_multiindex_columns(df)

                        # Standardize column names (lowercase, strip whitespace)
                        df.columns = df.columns.str.lower().str.strip()

                        all_data = pd.concat([all_data, df], ignore_index=True)
                    else:
                        st.warning(f"No data found for {statement_type} for {symbol}.")
                else:
                    st.error(f"Error fetching {statement_type} data: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Network error fetching {statement_type} for {symbol}: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred while processing {statement_type} for {symbol}: {e}")
        
        if not all_data.empty:
            st.success(f"Successfully fetched financial data for {symbol}.")
            # Convert date columns to datetime after all concatenations
            if 'date' in all_data.columns:
                all_data['date'] = pd.to_datetime(all_data['date'])
            if 'fillingdate' in all_data.columns: # Note: 'fillingDate' becomes 'fillingdate'
                all_data['fillingdate'] = pd.to_datetime(all_data['fillingdate'])
            return all_data
        else:
            st.warning(f"No financial data could be fetched for {symbol}.")
            return None
    return all_data

# --- 2. EDA Functions (from EDA.py) ---

def load_and_prepare_data_eda(financial_df, stock_price_df):
    """
    Load and perform initial processing of financial and stock price data.
    Ensures consistent column naming and numeric types.
    """
    st.subheader("Loading and Preparing Data for EDA")
    
    if financial_df is None or financial_df.empty:
        st.error("Financial data DataFrame is empty or None.")
        return None, None
    if stock_price_df is None or stock_price_df.empty:
        st.error("Stock price data DataFrame is empty or None.")
        return None, None

    # Ensure 'date' is datetime for both (already handled by download/fetch functions)
    if 'date' not in financial_df.columns or 'date' not in stock_price_df.columns:
        st.error("One or both DataFrames are missing the 'date' column after initial loading.")
        return None, None
    
    st.write(f"Financial data shape before preparation: {financial_df.shape}")
    st.write(f"Stock price data shape before preparation: {stock_price_df.shape}")

    # Prepare financial data (from prepare_financial_data logic)
    df_prepared = financial_df.copy()
    
    # Convert relevant columns to numeric, coercing errors
    # Use lowercase names as standardized by fetch_financial_data
    numeric_columns = [
        'revenue', 'costofrevenue', 'grossprofit', 'grossprofitratio',
        'researchanddevelopmentexpenses', 'operatingexpenses', 'costandexpenses',
        'interestincome', 'interesstexpense', 'ebitda', 'ebitdaratio',
        'operatingincome', 'operatingincomeratio', 'incomebeforetax',
        'incomebeforetaxratio', 'incometaxexpense', 'netincome', 'netincomeratio',
        'eps', 'epsdiluted', 'weightedaverageshsout', 'weightedaverageshsoutdil'
    ]
    for col in numeric_columns:
        if col in df_prepared.columns:
            df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce')

    # Select relevant columns for financial analysis (using lowercase names)
    key_metrics = [
        'date', 'calendaryear', 'period', 'revenue', 'grossprofit', 'netincome',
        'operatingincome', 'eps', 'epsdiluted', 'researchanddevelopmentexpenses',
        'operatingexpenses', 'grossprofitratio', 'operatingincomeratio',
        'netincomeratio', 'weightedaverageshsoutdil'
    ]
    
    # Filter df_prepared to include only existing key_metrics columns
    existing_columns = [col for col in key_metrics if col in df_prepared.columns]
    df_prepared = df_prepared[existing_columns]
    df_prepared = df_prepared.sort_values('date')
    
    st.success("Data loaded and prepared for EDA.")
    return df_prepared, stock_price_df

def merge_datasets_eda(financial_df_prepared, stock_price_df, ticker):
    """
    Merge financial and stock price dataframes using a robust date-range and ffill strategy.
    This is inspired by the more effective merging logic from the user's previous EDA script.
    """
    st.subheader("Merging Datasets")
    if financial_df_prepared is None or financial_df_prepared.empty:
        st.error("Financial data for merging is empty or None.")
        return None
    if stock_price_df is None or stock_price_df.empty:
        st.error("Stock price data for merging is empty or None.")
        return None

    # Ensure 'date' is datetime and sort both dataframes
    # (Already handled by download/fetch functions, but good to ensure sort)
    financial_df_prepared = financial_df_prepared.sort_values('date').copy()
    stock_price_df = stock_price_df.sort_values('date').copy()

    # Determine the overall date range for the merged data
    min_financial_date = financial_df_prepared['date'].min()
    max_stock_date = stock_price_df['date'].max()

    if pd.isna(min_financial_date) or pd.isna(max_stock_date):
        st.error("Could not determine valid date range for merging. Check financial and stock data dates.")
        return None

    st.info(f"First financial report date: {min_financial_date.strftime('%Y-%m-%d')}")
    st.info(f"Last stock price date: {max_stock_date.strftime('%Y-%m-%d')}")

    # Create a complete daily date range for the entire period
    all_dates_df = pd.DataFrame({'date': pd.date_range(start=min_financial_date, end=max_stock_date, freq='D')})

    # Merge financial data onto the full date range
    # This will create NaNs for days without a financial report
    merged_financial_on_dates = pd.merge(all_dates_df, financial_df_prepared, on='date', how='left')

    # Forward fill financial data: each day gets the most recent quarterly/annual data
    st.info("Forward filling financial data across the date range...")
    # Identify columns to ffill (all except 'date', 'calendaryear', 'period', 'statement_type')
    cols_to_ffill = [col for col in merged_financial_on_dates.columns if col not in ['date', 'calendaryear', 'period', 'statement_type']]
    merged_financial_on_dates[cols_to_ffill] = merged_financial_on_dates[cols_to_ffill].ffill()
    # Backward fill any leading NaNs that ffill couldn't handle (e.g., before the very first financial report)
    merged_financial_on_dates[cols_to_ffill] = merged_financial_on_dates[cols_to_ffill].bfill()
    
    # Merge the stock price data onto this combined financial + date range DataFrame
    # Use a left merge to keep all dates from the full date range, filling stock data with NaNs on non-trading days
    merged_df = pd.merge(merged_financial_on_dates, stock_price_df, on='date', how='left')

    # Set 'date' as index for consistency with later operations
    merged_df.set_index('date', inplace=True)
    merged_df.sort_index(inplace=True)

    # Filter out non-trading days (where 'close' price is NaN after the merge)
    rows_before_dropna_close = merged_df.shape[0]
    if 'close' in merged_df.columns:
        merged_df.dropna(subset=['close'], inplace=True)
        rows_after_dropna_close = merged_df.shape[0]
        st.info(f"Filtered out {rows_before_dropna_close - rows_after_dropna_close} rows with no stock trading (weekends/holidays/missing close price).")
    else:
        st.warning("Warning: 'close' column not found after merge. Cannot filter non-trading days effectively.")

    # Check for any duplicate dates after all merges (should be rare with date_range and dropna)
    if merged_df.index.has_duplicates:
        st.warning(f"Found {merged_df.index.duplicated().sum()} duplicate dates after merge. Removing duplicates.")
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

    st.write(f"Merged DataFrame shape: {merged_df.shape}")
    st.success("Datasets merged successfully.")
    return merged_df

def calculate_technical_indicators_eda(df):
    """
    Calculate various technical indicators.
    Ensures consistent lowercase column names.
    """
    st.subheader("Calculating Technical Indicators")
    if df is None or df.empty:
        st.error("DataFrame for technical indicator calculation is empty or None.")
        return None

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # Ensure 'close' and 'volume' columns exist and are numeric
    for col in ['close', 'volume']:
        if col not in df_copy.columns:
            st.warning(f"Column '{col}' not found for technical indicator calculation. Skipping indicator calculation.")
            return df_copy # Return as is if essential columns are missing
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    # Simple Moving Averages (MA_X -> ma_X)
    for window in [5, 10, 20, 50, 200]:
        df_copy[f'ma_{window}'] = df_copy['close'].rolling(window=window).mean()

    # Price Change Percentage
    df_copy['price_change_pct'] = df_copy['close'].pct_change() * 100

    # Relative Strength Index (RSI -> rsi)
    window_rsi = 14
    if len(df_copy) >= window_rsi:
        delta = df_copy['close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)

        avg_gain_init = gain.iloc[:window_rsi].mean()
        avg_loss_init = loss.iloc[:window_rsi].mean()

        avg_gains = [avg_gain_init]
        avg_losses = [avg_loss_init]

        for i in range(window_rsi, len(df_copy)):
            avg_gain_next = (avg_gains[-1] * (window_rsi - 1) + gain.iloc[i]) / window_rsi
            avg_loss_next = (avg_losses[-1] * (window_rsi - 1) + loss.iloc[i]) / window_rsi
            avg_gains.append(avg_gain_next)
            avg_losses.append(avg_loss_next)

        avg_gain_series = pd.Series(avg_gains, index=df_copy.index[window_rsi-1:])
        avg_loss_series = pd.Series(avg_losses, index=df_copy.index[window_rsi-1:])

        rs = avg_gain_series / avg_loss_series
        df_copy['rsi'] = pd.Series(100 - (100 / (1 + rs)), index=df_copy.index[window_rsi-1:])
    else:
        df_copy['rsi'] = np.nan
        st.warning(f"Not enough data ({len(df_copy)} rows) to calculate RSI (requires {window_rsi} rows). RSI column will be NaN.")

    # Daily Returns
    df_copy['returns'] = df_copy['close'].pct_change()

    # Volatility (Standard Deviation of Returns)
    for window in [5, 10, 20]:
        df_copy[f'volatility_{window}d'] = df_copy['returns'].rolling(window=window).std() * np.sqrt(252) # Annualized

    # Volume Change and Volume Moving Average
    if 'volume' in df_copy.columns:
        df_copy['volume_change'] = df_copy['volume'].pct_change()
        df_copy['volume_ma_5'] = df_copy['volume'].rolling(window=5).mean()
    else:
        st.warning("Volume column not found. Skipping volume change and MA calculation.")
        df_copy['volume_change'] = np.nan
        df_copy['volume_ma_5'] = np.nan

    # Momentum
    if 'close' in df_copy.columns:
        for window in [5, 10, 20]:
            df_copy[f'momentum_{window}d'] = df_copy['close'].diff(periods=window) # Using diff for momentum
    else:
        st.warning("Close column not found. Skipping momentum calculation.")
        for window in [5, 10, 20]:
            df_copy[f'momentum_{window}d'] = np.nan

    # Price-to-Earnings Ratio (pe_ratio)
    if 'eps' in df_copy.columns and 'close' in df_copy.columns:
        df_copy['eps'] = pd.to_numeric(df_copy['eps'], errors='coerce')
        df_copy['pe_ratio'] = df_copy['close'] / df_copy['eps'].replace(0, np.nan)
    else:
        st.warning("EPS or Close column not found. Cannot calculate PE_ratio.")
        df_copy['pe_ratio'] = np.nan

    # Market Cap
    if 'weightedaverageshsoutdil' in df_copy.columns and 'close' in df_copy.columns:
        df_copy['weightedaverageshsoutdil'] = pd.to_numeric(df_copy['weightedaverageshsoutdil'], errors='coerce')
        df_copy['market_cap'] = df_copy['close'] * df_copy['weightedaverageshsoutdil'].replace(0, np.nan)
    else:
        st.warning("Weighted Average Shares Outstanding Diluted or Close column not found. Cannot calculate Market Cap.")
        df_copy['market_cap'] = np.nan

    # Price to Sales
    if 'revenue' in df_copy.columns and 'market_cap' in df_copy.columns:
        df_copy['revenue'] = pd.to_numeric(df_copy['revenue'], errors='coerce')
        df_copy['price_to_sales'] = df_copy['market_cap'] / df_copy['revenue'].replace(0, np.nan)
    else:
        st.warning("Revenue or Market Cap column not found. Cannot calculate Price to Sales.")
        df_copy['price_to_sales'] = np.nan

    # Create 'days_since_financial_update' (simplified for Streamlit context)
    if 'eps' in df_copy.columns:
        # Identify dates where EPS changes (proxy for financial report updates)
        # This assumes EPS is updated on financial report dates and is constant between them
        eps_change_dates = df_copy[df_copy['eps'].diff().fillna(True) != 0].index
        
        # Calculate days since last update
        df_copy['days_since_financial_update'] = np.nan
        last_update_date = None
        for i, date in enumerate(df_copy.index):
            if date in eps_change_dates:
                last_update_date = date
            if last_update_date is not None:
                df_copy.loc[date, 'days_since_financial_update'] = (date - last_update_date).days
        
        # Fill any remaining NaNs at the beginning (before first report) with 0 or a large number
        df_copy['days_since_financial_update'].fillna(0, inplace=True) # Or a value indicating 'very old'
    else:
        st.warning("EPS column not found. Cannot calculate 'days_since_financial_update'.")
        df_copy['days_since_financial_update'] = np.nan


    st.success("Technical indicators calculated.")
    return df_copy

def preprocess_data_for_model(df):
    """
    Performs final preprocessing steps for the model:
    - Fills NaN values using forward-fill and backward-fill.
    - Drops any remaining rows with NaN values (should be minimal after filling).
    - Selects final features for the model and ensures 'close' is the first column.
    """
    st.subheader("Preprocessing Data for Model")
    if df is None or df.empty:
        st.error("DataFrame for model preprocessing is empty or None.")
        return None

    initial_rows = df.shape[0]
    df_processed = df.copy()

    # Drop any remaining non-numeric columns that might have slipped through or are not needed
    numeric_cols_only = df_processed.select_dtypes(include=np.number).columns.tolist()
    df_processed = df_processed[numeric_cols_only]

    # --- CRITICAL CHANGE: Fill NaN values before dropping ---
    # Forward-fill first to propagate last known values (e.g., financial data, indicators)
    st.info("Applying forward-fill to fill NaNs...")
    df_processed.fillna(method='ffill', inplace=True)
    # Backward-fill to catch any leading NaNs that ffill missed (e.g., at the very start of series for indicators)
    st.info("Applying backward-fill to fill remaining leading NaNs...")
    df_processed.fillna(method='bfill', inplace=True)
    # --- END CRITICAL CHANGE ---

    # After filling, check for columns that are still entirely NaN (e.g., if a feature was never available)
    cols_to_drop_all_nan = df_processed.columns[df_processed.isnull().all()].tolist()
    if cols_to_drop_all_nan:
        st.warning(f"Dropping columns with all NaN values after filling: {', '.join(cols_to_drop_all_nan)}")
        df_processed.drop(columns=cols_to_drop_all_nan, inplace=True)

    # Now, drop any remaining rows with NaN values. This should be much fewer.
    rows_before_final_dropna = df_processed.shape[0]
    df_processed.dropna(inplace=True)
    rows_after_final_dropna = df_processed.shape[0]
    st.info(f"Dropped {rows_before_final_dropna - rows_after_final_dropna} rows due to remaining NaN values after filling and column cleanup.")
    st.write(f"DataFrame shape after filling NaNs and final dropna: {df_processed.shape}")

    if df_processed.empty:
        st.error("DataFrame is empty after preprocessing (filling NaNs and final dropna). Cannot proceed with model training. This usually means there's insufficient overlapping data or critical features are entirely missing.")
        return None

    # Ensure 'close' is the first column for the create_sequences function
    # It must exist and be numeric by this point.
    if 'close' not in df_processed.columns:
        st.error("Critical Error: 'close' column is missing after preprocessing. Cannot prepare data for model.")
        return None
    
    model_features_ordered = ['close'] + [col for col in df_processed.columns if col != 'close']
    
    # Filter to only include columns that exist in the DataFrame
    final_model_df = df_processed[[col for col in model_features_ordered if col in df_processed.columns]]

    st.write(f"Final processed data shape for model: {final_model_df.shape}")
    st.write(f"Final features for model (order matters for create_sequences): {final_model_df.columns.tolist()}")
    st.success("Data preprocessed for model successfully.")
    return final_model_df

# --- 3. Transformer Model Functions (from TF_Stock_PredictionV2.ipynb) ---

# The create_sequences and build_transformer_model functions are now defined at the top
# of this script for clarity and to ensure they are available.

def run_transformer_prediction_pipeline(processed_df, ticker, sequence_length=60, prediction_days=30, num_folds=5):
    """
    Runs the full Transformer prediction pipeline including scaling,
    walk-forward validation, final model training, and future prediction.
    """
    st.subheader(f"Transformer Model Prediction for {ticker}")

    if processed_df is None or processed_df.empty:
        st.error("Processed data is empty or None. Cannot run Transformer prediction.")
        return

    st.write(f"Processed data shape entering pipeline: {processed_df.shape}")
    st.write(f"Processed data columns: {processed_df.columns.tolist()}")

    # Ensure 'close' is the first column for scaling and prediction target
    # This was already handled in preprocess_data_for_model, but double-check for robustness.
    if processed_df.columns[0] != 'close':
        st.error("Error: 'close' column is not the first column in processed_df. Re-order before passing to pipeline.")
        return

    data = processed_df.values
    dates = processed_df.index

    st.write(f"Data array shape (after converting DataFrame to numpy array): {data.shape}")

    # Scale the data
    # The scaler should be fit on ALL features, but inverse_transform will only use the first column (close)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Walk-Forward Validation Setup
    train_size_ratio = 0.7 # Initial training set size
    validation_window_size = 90 # Days for each validation fold
    
    # Calculate initial train and validation sizes in terms of rows
    initial_train_size = int(len(scaled_data) * train_size_ratio)
    
    # Ensure there's enough data for at least one sequence in train and validation
    # A sequence needs `sequence_length` steps for X and 1 step for y.
    # So, minimum data for training a model is `sequence_length + 1`.
    # For validation, you need at least `sequence_length + 1` data points as well.
    if initial_train_size < sequence_length + 1:
        st.error(f"Not enough data for initial training set to form sequences. Required: {sequence_length + 1} rows. Available: {initial_train_size} rows.")
        return
    
    # Determine the number of steps for walk-forward validation
    # Each step advances by the validation_window_size
    # The remaining data after initial_train_size must be sufficient for at least one validation window
    # and to form sequences within that window.
    remaining_data = len(scaled_data) - initial_train_size
    num_steps = (remaining_data - sequence_length) // validation_window_size
    
    if num_steps < num_folds:
        st.warning(f"Not enough data for {num_folds} walk-forward folds. Calculated {num_steps} possible folds. Running with {num_steps} folds instead.")
        num_folds = num_steps
    
    if num_folds <= 0:
        st.error("Not enough data to perform any walk-forward validation folds. Please provide more data or reduce sequence length/validation window size.")
        return

    wf_metrics = {'rmse': [], 'mae': [], 'r2': []}

    st.write(f"Starting Walk-Forward Validation with {num_folds} folds...")

    for i in range(num_folds):
        st.write(f"==== Starting Walk-Forward Fold {i + 1} ====")
        end_train_index = initial_train_size + i * validation_window_size
        end_val_index = end_train_index + validation_window_size

        # Check if validation data range is valid
        if end_val_index > len(scaled_data):
            st.warning(f"Fold {i+1} validation data would exceed total data bounds. Stopping walk-forward validation early.")
            break

        train_data = scaled_data[:end_train_index]
        val_data = scaled_data[end_train_index:end_val_index]

        X_train, y_train = create_sequences(train_data, sequence_length)
        X_val, y_val = create_sequences(val_data, sequence_length)

        st.write(f"  Train data shape for fold {i+1}: {train_data.shape}")
        st.write(f"  Validation data shape for fold {i+1}: {val_data.shape}")
        st.write(f"  Training sequences shape for fold {i+1}: {X_train.shape}")
        st.write(f"  Validation sequences shape for fold {i+1}: {X_val.shape}")


        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            st.warning(f"Skipping Fold {i+1}: Not enough sequences formed in train ({X_train.shape[0]} sequences) or validation set ({X_val.shape[0]} sequences).")
            continue

        # input_shape for the model should now reflect the number of features (all columns except 'close')
        # X_train.shape[2] gives the number of features (columns) in the sequences
        model = build_transformer_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )

        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )

        early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0 # Suppress verbose output in Streamlit
        )

        val_preds_scaled = model.predict(X_val)
        
        # To inverse transform, we need to create a dummy array with the correct number of features
        # and place the predicted 'close' values in the first column (index 0)
        # The number of features for inverse transform is `data.shape[1]` (original number of columns)
        dummy_val_preds = np.zeros((len(val_preds_scaled), data.shape[1]))
        dummy_val_preds[:, 0] = val_preds_scaled.flatten()
        val_preds = scaler.inverse_transform(dummy_val_preds)[:, 0]

        dummy_y_val = np.zeros((len(y_val), data.shape[1]))
        dummy_y_val[:, 0] = y_val.flatten()
        actual_val = scaler.inverse_transform(dummy_y_val)[:, 0]

        rmse = np.sqrt(mean_squared_error(actual_val, val_preds))
        mae = mean_absolute_error(actual_val, val_preds)
        r2 = r2_score(actual_val, val_preds)

        wf_metrics['rmse'].append(rmse)
        wf_metrics['mae'].append(mae)
        wf_metrics['r2'].append(r2)

        st.write(f"  Metrics for Fold {i + 1}:")
        st.write(f"    Validation RMSE: {rmse:.4f}")
        st.write(f"    Validation MAE: {mae:.4f}")
        st.write(f"    Validation R²: {r2:.4f}")

    if wf_metrics['rmse']:
        avg_rmse = np.mean(wf_metrics['rmse'])
        avg_mae = np.mean(wf_metrics['mae'])
        avg_r2 = np.mean(wf_metrics['r2'])
        st.success(f"\n==== Walk-Forward Validation Complete ({len(wf_metrics['rmse'])} folds) ====")
        st.write(f"Average Validation RMSE across folds: {avg_rmse:.4f}")
        st.write(f"Average Validation MAE across folds: {avg_mae:.4f}")
        st.write(f"Average Validation R²: {avg_r2:.4f}")
    else:
        st.warning("\nNo successful validation folds were completed.")
        return

    # --- Train Final Model on Full Dataset ---
    st.write("\n==== Training Final Model on Full Dataset ====")
    X_full, y_full = create_sequences(scaled_data, sequence_length)
    if X_full.shape[0] == 0:
        st.error("Not enough data to create sequences for final model training.")
        return

    final_model = build_transformer_model(
        input_shape=(X_full.shape[1], X_full.shape[2]),
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )
    final_model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    final_model.fit(X_full, y_full, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
    st.success("Final model trained on full dataset.")

    # --- Evaluate Final Model on Historical Data (for plotting) ---
    st.write("\n==== Evaluating Final Model on Historical Data ====")
    train_predict_scaled = final_model.predict(X_full)
    
    dummy_train_predict = np.zeros((len(train_predict_scaled), data.shape[1]))
    dummy_train_predict[:, 0] = train_predict_scaled.flatten()
    train_predict = scaler.inverse_transform(dummy_train_predict)[:, 0]

    # Shift train predictions for plotting
    train_predict_plot = np.empty_like(data[:, 0])
    train_predict_plot[:] = np.nan
    train_predict_plot[sequence_length:len(train_predict) + sequence_length] = train_predict

    # --- Future Predictions ---
    st.write(f"\n==== Predicting {prediction_days} Days into the Future for {ticker} ====")
    # The last sequence of features (excluding 'close') from the scaled data
    last_sequence_features = scaled_data[-sequence_length:, 1:] 
    future_predictions_scaled = []
    current_input_for_prediction = last_sequence_features.copy()

    for _ in range(prediction_days):
        # Predict the next 'close' price using the current sequence of features
        prediction_scaled_close = final_model.predict(current_input_for_prediction[np.newaxis, :, :])
        future_predictions_scaled.append(prediction_scaled_close[0, 0])
        
        # To update current_input_for_prediction for the next step:
        # We assume other features (from column 1 onwards) will be the same as the last observed features.
        # This is a simplification for future prediction.
        new_feature_row = current_input_for_prediction[-1, :].copy() 
        
        # Stack the new feature row, removing the oldest one
        current_input_for_prediction = np.vstack([current_input_for_prediction[1:], new_feature_row])

    dummy_future_predictions = np.zeros((len(future_predictions_scaled), data.shape[1]))
    dummy_future_predictions[:, 0] = np.array(future_predictions_scaled).flatten()
    future_preds = scaler.inverse_transform(dummy_future_predictions)[:, 0]

    last_date = dates[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]

    future_df = pd.DataFrame(future_preds, index=future_dates, columns=['predicted_close'])
    st.success(f"Future predictions for the next {prediction_days} days generated.")
    st.dataframe(future_df.head())

    # --- Visualize Predictions ---
    st.write("\n==== Visualizing Predictions ====")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, data[:, 0], label='Actual Close Price')
    ax.plot(dates[sequence_length:], train_predict, label='Historical Predicted Close Price', linestyle='--')
    ax.plot(future_dates, future_preds, label='Future Predicted Close Price', color='red')

    ax.set_title(f'Transformer Model Stock Price Prediction for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    st.success("Prediction visualization complete.")


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Stock Price Prediction with Transformer")

st.title("Stock Price Prediction with Transformer Model")
st.markdown("""
This application downloads historical stock and financial data,
calculates technical indicators, preprocesses the data,
and uses a Transformer model with Walk-Forward Validation to predict future stock prices.
""")

# User Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL").upper()
use_local_data = st.sidebar.checkbox("Use local data files?", False)
local_data_path = st.sidebar.text_input("Local Data Folder Path (e.g., data/):", "data/")

sequence_length = st.sidebar.slider("Sequence Length (for Transformer input):", 10, 120, 60)
prediction_days = st.sidebar.slider("Number of Future Prediction Days:", 1, 90, 30)
num_folds = st.sidebar.slider("Number of Walk-Forward Validation Folds:", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Status")

# Initialize session state for dataframes
if 'stock_df' not in st.session_state:
    st.session_state.stock_df = None
if 'financial_df' not in st.session_state:
    st.session_state.financial_df = None
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None
if 'df_with_indicators' not in st.session_state: # New: Store df after indicators for EDA
    st.session_state.df_with_indicators = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

if st.sidebar.button("Load and Process Data"):
    with st.spinner("Loading and processing data... This may take a moment."):
        # 1. Data Download
        st.session_state.stock_df = download_stock_data(ticker, use_local_data, local_data_path)
        st.session_state.financial_df = fetch_financial_data(ticker, FMP_API_KEY, FMP_STATEMENT_TYPES, use_local_data, local_data_path)

        if st.session_state.stock_df is not None and st.session_state.financial_df is not None:
            # 2. EDA Data Prep and Merge
            financial_df_prepared, stock_price_df_for_merge = load_and_prepare_data_eda(st.session_state.financial_df, st.session_state.stock_df)
            
            if financial_df_prepared is not None and stock_price_df_for_merge is not None:
                st.session_state.merged_df = merge_datasets_eda(financial_df_prepared, stock_price_df_for_merge, ticker)

                if st.session_state.merged_df is not None:
                    # 3. Calculate Technical Indicators
                    st.session_state.df_with_indicators = calculate_technical_indicators_eda(st.session_state.merged_df)
                    
                    if st.session_state.df_with_indicators is not None:
                        # 4. Preprocess for Model (now uses df_with_indicators)
                        st.session_state.processed_df = preprocess_data_for_model(st.session_state.df_with_indicators)
                        
                        if st.session_state.processed_df is not None:
                            st.success("All data loading and preprocessing steps completed!")
                        else:
                            st.error("Failed to preprocess data for the model.")
                    else:
                        st.error("Failed to calculate technical indicators.")
                else:
                    st.error("Failed to merge stock and financial data.")
            else:
                st.error("Failed to prepare data for EDA/merging.")
        else:
            st.error("Failed to download either stock or financial data. Please check ticker and API key.")

if st.session_state.stock_df is not None:
    st.sidebar.write(f"**Stock Data Loaded:** Yes (Shape: {st.session_state.stock_df.shape})")
else:
    st.sidebar.write("**Stock Data Loaded:** No")

if st.session_state.financial_df is not None:
    st.sidebar.write(f"**Financial Data Loaded:** Yes (Shape: {st.session_state.financial_df.shape})")
else:
    st.sidebar.write("**Financial Data Loaded:** No")

if st.session_state.merged_df is not None:
    st.sidebar.write(f"**Merged Data Loaded:** Yes (Shape: {st.session_state.merged_df.shape})")
else:
    st.sidebar.write("**Merged Data Loaded:** No")

if st.session_state.df_with_indicators is not None:
    st.sidebar.write(f"**Data with Indicators Loaded:** Yes (Shape: {st.session_state.df_with_indicators.shape})")
else:
    st.sidebar.write("**Data with Indicators Loaded:** No")

if st.session_state.processed_df is not None:
    st.sidebar.write(f"**Processed Data (for Model) Loaded:** Yes (Shape: {st.session_state.processed_df.shape})")
else:
    st.sidebar.write("**Processed Data (for Model) Loaded:** No")

st.markdown("---")

# --- EDA Section ---
st.header("Exploratory Data Analysis (EDA)")
if st.session_state.df_with_indicators is not None and not st.session_state.df_with_indicators.empty:
    with st.expander("View EDA Details"):
        st.subheader("Raw Data with Technical Indicators (Head)")
        st.dataframe(st.session_state.df_with_indicators.head())

        st.subheader("Descriptive Statistics")
        st.dataframe(st.session_state.df_with_indicators.describe())

        st.subheader("Close Price Over Time")
        fig_close_price, ax_close_price = plt.subplots(figsize=(12, 6))
        ax_close_price.plot(st.session_state.df_with_indicators.index, st.session_state.df_with_indicators['close'])
        ax_close_price.set_title(f'{ticker} Close Price Over Time')
        ax_close_price.set_xlabel('Date')
        ax_close_price.set_ylabel('Close Price')
        ax_close_price.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_close_price)

        st.subheader("Feature Correlation Heatmap")
        # Select only numeric columns for correlation
        numeric_df = st.session_state.df_with_indicators.select_dtypes(include=np.number)
        
        # Drop columns with all NaNs if any, before calculating correlation
        numeric_df_cleaned = numeric_df.dropna(axis=1, how='all')

        if not numeric_df_cleaned.empty and numeric_df_cleaned.shape[1] > 1:
            fig_corr, ax_corr = plt.subplots(figsize=(16, 12))
            sns.heatmap(numeric_df_cleaned.corr(), annot=False, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            ax_corr.set_title(f'Feature Correlation Heatmap for {ticker}')
            plt.tight_layout()
            st.pyplot(fig_corr)
        else:
            st.warning("Not enough numeric data or features to generate a correlation heatmap after cleaning.")
else:
    st.info("Load and process data to view EDA details.")

st.markdown("---")

if st.button("Run Transformer Prediction Pipeline"):
    if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
        try:
            run_transformer_prediction_pipeline(
                st.session_state.processed_df,
                ticker,
                sequence_length=sequence_length,
                prediction_days=prediction_days,
                num_folds=num_folds
            )
        except Exception as e:
            st.error(f"An error occurred during the Transformer prediction pipeline: {e}")
            st.exception(e) # Display full traceback
    else:
        st.warning("Please load and process data first before running the prediction pipeline.")

st.markdown("---")
st.markdown("### Important Notes:")
st.markdown("""
* **API Key:** The Financial Modeling Prep API key has been updated.
* **Local Data:** You can now choose to load data from local CSV files by checking the "Use local data files?" box and providing the "Local Data Folder Path".
* **Data Paths:** This script will first attempt to load from local files if the option is enabled. If local files are not found or the option is disabled, it will fall back to downloading data via APIs.
* **Model Training Time:** The Transformer model training can be computationally intensive, especially for larger datasets or more folds. Be patient!
* **Feature Selection:** The `run_transformer_prediction_pipeline` function now correctly excludes the 'close' price from the input features (X) and uses it only as the prediction target (y).
* **Troubleshooting Folds & Empty Data:** If you see fewer folds than expected or an empty DataFrame error, check the "Data Status" in the sidebar and the "Preprocessing Data for Model" messages. The `fillna` steps should significantly reduce data loss, but extreme data sparsity can still lead to issues.
""")
