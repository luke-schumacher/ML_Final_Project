import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # Import os for path manipulation

try:
    import tabulate
except ImportError:
    tabulate = None
    print("Warning: 'tabulate' library not found. DataFrame outputs will not be formatted as Markdown tables.")

# --- GLOBAL VARIABLES START ---
# Define the stock ticker and a base title as global variables
GLOBAL_STOCK_TICKER = "NVDA" # Assuming your data is for NVDA
GLOBAL_STOCK_TITLE = "Dataset Overview" # Base title for charts and outputs
# --- GLOBAL VARIABLES END ---

def print_df_markdown(df_to_print, title="", index=False):
    """Helper function to print DataFrame as Markdown or fallback to string."""
    if title:
        print(f"\n--- {title} ---")
    if tabulate:
        print(df_to_print.to_markdown(index=index))
    else:
        print(df_to_print.to_string(index=index))

def plot_dataframe_as_image(df_to_plot, title, filename, output_path, index=False):
    """
    Plots a DataFrame as a table image and saves it.

    Parameters:
    df_to_plot (pd.DataFrame): The DataFrame to plot.
    title (str): Title for the plot.
    filename (str): Base filename for the saved image (e.g., "missing_values").
    output_path (str): Directory where the image will be saved.
    index (bool): Whether to include DataFrame index in the table.
    """
    if df_to_plot.empty:
        print(f"Warning: DataFrame for '{title}' is empty. Skipping image generation.")
        return

    # Adjust figure size dynamically based on number of rows and columns
    # A base height of 0.4 per row, plus 2 for title/padding.
    # Width adjusts based on number of columns, with a cap for readability.
    fig_height = df_to_plot.shape[0] * 0.4 + 2
    # Heuristic for width: 1.5 units per column, min 6, max 18
    fig_width = max(6, min(18, df_to_plot.shape[1] * 1.5 + 2))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Hide axes
    ax.axis('off')
    ax.axis('tight')

    # Create the table
    table = ax.table(cellText=df_to_plot.values,
                     colLabels=df_to_plot.columns,
                     rowLabels=df_to_plot.index if index else None,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10) # Set a reasonable font size
    table.scale(1.2, 1.2) # Scale table to fill the figure

    ax.set_title(title, fontsize=14, pad=20) # Add title to the plot

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, f"{filename}_{GLOBAL_STOCK_TICKER}.png")

    try:
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Created table image: {full_path}")
    except Exception as e:
        print(f"Error saving table image '{full_path}': {e}")
    plt.close(fig) # Close the figure to free memory


def load_single_csv(file_path):
    """
    Load a single CSV file, parse dates, and ensure 'date' column exists.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    DataFrame: Loaded and preprocessed DataFrame, or None if an error occurs.
    """
    print(f"Loading data from {file_path}...")
    try:
        # Directly attempt to read with 'date' as the column to parse
        df = pd.read_csv(file_path, parse_dates=['date'])
        # The column is already named 'date' as per inspection, so no rename needed.

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    if 'date' not in df.columns:
        print("Critical Error: 'date' column is missing from the loaded data.")
        return None

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True) # Ensure sorted by date and reset index

    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {df.columns.tolist()}")

    return df

def visualize_raw_data_head(df, output_path):
    """
    Visualizes the head of the DataFrame as an image.
    """
    print("\nVisualizing DataFrame Head...")
    plot_dataframe_as_image(df.head(), "Raw Data - First 5 Rows", "raw_data_head", output_path, index=False)

def visualize_dataframe_info_image(df, output_path):
    """
    Visualizes DataFrame column info (dtypes, non-null counts) as an image.
    """
    print("\nVisualizing DataFrame Info...")
    info_data = []
    for col in df.columns:
        dtype = df[col].dtype
        non_null_count = df[col].count()
        total_count = len(df)
        missing_count = total_count - non_null_count
        missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0
        info_data.append([col, str(dtype), non_null_count, missing_count, f"{missing_percentage:.2f}%"])

    info_df = pd.DataFrame(info_data, columns=['Column Name', 'Data Type', 'Non-Null Count', 'Missing Count', 'Missing %'])
    plot_dataframe_as_image(info_df, "DataFrame Column Information", "dataframe_info", output_path, index=False)

def visualize_dataframe_shape_image(df, output_path):
    """
    Visualizes the shape (number of records/columns) of the DataFrame as an image.
    """
    print("\nVisualizing DataFrame Shape...")
    shape_data = pd.DataFrame({
        'Metric': ['Number of Records', 'Number of Columns'],
        'Value': [df.shape[0], df.shape[1]]
    })
    plot_dataframe_as_image(shape_data, "DataFrame Dimensions", "dataframe_shape", output_path, index=False)


def main(input_csv_path, output_path=None):
    """
    Main function to load CSV data and generate specified visualizations.

    Parameters:
    input_csv_path (str): Path to the combined CSV file.
    output_path (str, optional): Base path to save visualizations (e.g., "visuals").
    """
    try:
        # Load the single combined data file
        df = load_single_csv(input_csv_path)

        if df is None:
            print("Error loading data. Exiting.")
            return None

        # Ensure output_path is set
        if output_path is None:
            output_path = "raw_data_visuals" # Default output directory
            print(f"No output_path provided. Saving visuals to: {output_path}")

        # Create visualizations as images
        visualize_raw_data_head(df, output_path)
        visualize_dataframe_info_image(df, output_path)
        visualize_dataframe_shape_image(df, output_path)

        print("\nRaw data visualization complete.")
        return df

    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    # Define the path for your combined CSV file
    combined_csv_path = "ML_Final_Project/Lusofona/allFinData/NVDA_merged_dataset_NVDA.csv"

    # Define an output path for charts and tables
    # This directory will be created if it doesn't exist
    visuals_output_dir = "raw_data_visuals"

    # Run the main processing
    final_processed_df = main(combined_csv_path, visuals_output_dir)

    if final_processed_df is not None:
        print(f"\nSuccessfully generated raw data visuals in the '{visuals_output_dir}' directory.")
        print(f"Loaded DataFrame shape: {final_processed_df.shape}")
    else:
        print("\nFailed to generate raw data visuals.")
