"""
MarketData.py - Stock Market Data Fetcher
Fetches daily OHLCV data from YFinance for multiple sectors and stores them as CSV files.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# SECTOR DEFINITIONS
# =============================================================================

SECTORS = {
    "Technology": ["MSFT", "AAPL", "NVDA", "GOOGL", "AMD"],
    "Financial_Services": ["JPM", "BAC", "GS", "MS", "BLK"],
    "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV"],
    "Retail_Consumer_Discretionary": ["AMZN", "WMT", "COST", "NKE", "MCD"],
    "Energy_Oil_Gas": ["XOM", "CVX", "COP", "OXY", "SLB"],
    "ETFs_Benchmarks": ["SPY", "QQQ", "VTI", "URTH", "IWM"],
    "Blockchain_Crypto_Infrastructure": ["COIN", "MSTR", "MARA", "RIOT", "SQ"],
    "US_Defence_Aerospace": ["LMT", "NOC", "RTX", "GD", "BA"],
    "European_Defence": ["BA.L", "RHM.DE", "HO.PA", "LDO.MI", "SAAB-B.ST"],
    "Industrials": ["CAT", "GE", "HON", "UNP", "DE"],
    "Semiconductors": ["TSM", "INTC", "ASML", "AVGO", "QCOM"],
    "Consumer_Staples": ["PG", "KO", "PEP", "ULVR.L", "PM"],
    "Utilities": ["NEE", "DUK", "SO", "EXC", "AEP"],
    "Materials_Mining": ["BHP", "RIO", "FCX", "NEM", "ALB"],
    "Real_Estate_REITs": ["PLD", "AMT", "EQIX", "SPG", "O"],
    "Commodities_Futures": ["GC=F", "SI=F", "CL=F", "BZ=F", "NG=F"]
}

# Friendly names for commodities
COMMODITY_NAMES = {
    "GC=F": "Gold_Futures",
    "SI=F": "Silver_Futures",
    "CL=F": "WTI_Crude_Oil",
    "BZ=F": "Brent_Crude_Oil",
    "NG=F": "Natural_Gas"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_data_directory() -> Path:
    """Get the path to the MarketData directory."""
    # Navigate from src folder to MarketData folder
    current_dir = Path(__file__).parent.parent.parent
    data_dir = current_dir / "MarketData"
    return data_dir


def sanitize_filename(ticker: str) -> str:
    """
    Sanitize ticker symbol for use as filename.
    Handles special characters in tickers like BA.L, RHM.DE, GC=F
    """
    # Replace special characters
    sanitized = ticker.replace(".", "_").replace("=", "_").replace("-", "_")
    return sanitized


def get_csv_path(ticker: str, sector: str) -> Path:
    """Get the full path for a ticker's CSV file."""
    data_dir = get_data_directory()
    sector_dir = data_dir / sector
    filename = f"{sanitize_filename(ticker)}.csv"
    return sector_dir / filename


def ensure_sector_directories():
    """Create sector directories if they don't exist."""
    data_dir = get_data_directory()
    data_dir.mkdir(exist_ok=True)
    
    for sector in SECTORS.keys():
        sector_dir = data_dir / sector
        sector_dir.mkdir(exist_ok=True)
        logger.info(f"Directory ensured: {sector_dir}")


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_ticker_data(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker from YFinance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format (None for max history)
        end_date: End date in 'YYYY-MM-DD' format (None for today)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        
        if start_date is None:
            # Fetch maximum available history
            df = stock.history(period="max", interval="1d")
        else:
            # Fetch from specific date
            df = stock.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        # Keep only OHLCV columns
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[ohlcv_columns]
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Ensure Date column is properly formatted
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Add ticker column for reference
        df['Ticker'] = ticker
        
        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()


def save_ticker_data(df: pd.DataFrame, ticker: str, sector: str):
    """
    Save ticker data to CSV file.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker symbol
        sector: Sector name for directory organization
    """
    if df.empty:
        logger.warning(f"No data to save for {ticker}")
        return
    
    csv_path = get_csv_path(ticker, sector)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {ticker} data to {csv_path}")


def load_ticker_data(ticker: str, sector: str) -> pd.DataFrame:
    """
    Load existing ticker data from CSV file.
    
    Args:
        ticker: Stock ticker symbol
        sector: Sector name
    
    Returns:
        DataFrame with existing data or empty DataFrame
    """
    csv_path = get_csv_path(ticker, sector)
    
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        logger.info(f"Loaded {len(df)} existing rows for {ticker}")
        return df
    
    return pd.DataFrame()


def get_last_date(ticker: str, sector: str) -> datetime:
    """
    Get the last date of data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        sector: Sector name
    
    Returns:
        Last date in the data or None if no data exists
    """
    df = load_ticker_data(ticker, sector)
    
    if df.empty:
        return None
    
    return pd.to_datetime(df['Date']).max()


def update_ticker_data(ticker: str, sector: str) -> pd.DataFrame:
    """
    Update ticker data by fetching only new data since last stored date.
    
    Args:
        ticker: Stock ticker symbol
        sector: Sector name
    
    Returns:
        Updated DataFrame with all data
    """
    existing_df = load_ticker_data(ticker, sector)
    today = datetime.now().date()
    
    if existing_df.empty:
        # No existing data, fetch everything
        logger.info(f"No existing data for {ticker}, fetching full history")
        new_df = fetch_ticker_data(ticker)
        if not new_df.empty:
            save_ticker_data(new_df, ticker, sector)
        return new_df
    
    # Get the last date we have data for
    last_date = pd.to_datetime(existing_df['Date']).max().date()
    
    # Check if we need to update
    if last_date >= today - timedelta(days=1):
        logger.info(f"{ticker} data is up to date (last: {last_date})")
        return existing_df
    
    # Fetch new data starting from the day after last date
    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    logger.info(f"Updating {ticker} from {start_date}")
    
    new_df = fetch_ticker_data(ticker, start_date=start_date)
    
    if new_df.empty:
        logger.info(f"No new data available for {ticker}")
        return existing_df
    
    # Combine existing and new data
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates based on Date
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Save updated data
    save_ticker_data(combined_df, ticker, sector)
    
    logger.info(f"Updated {ticker}: added {len(new_df)} new rows, total {len(combined_df)} rows")
    return combined_df


# =============================================================================
# BATCH OPERATIONS
# =============================================================================

def fetch_all_sector_data(sector: str, update_only: bool = True):
    """
    Fetch data for all tickers in a sector.
    
    Args:
        sector: Sector name from SECTORS dictionary
        update_only: If True, only fetch new data; if False, fetch full history
    """
    if sector not in SECTORS:
        logger.error(f"Unknown sector: {sector}")
        return
    
    tickers = SECTORS[sector]
    logger.info(f"Fetching data for {sector}: {tickers}")
    
    for ticker in tickers:
        try:
            if update_only:
                update_ticker_data(ticker, sector)
            else:
                df = fetch_ticker_data(ticker)
                save_ticker_data(df, ticker, sector)
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {str(e)}")
            continue


def fetch_all_data(update_only: bool = True):
    """
    Fetch data for all tickers across all sectors.
    
    Args:
        update_only: If True, only fetch new data; if False, fetch full history
    """
    ensure_sector_directories()
    
    total_tickers = sum(len(tickers) for tickers in SECTORS.values())
    processed = 0
    
    logger.info(f"Starting data fetch for {total_tickers} tickers across {len(SECTORS)} sectors")
    
    for sector, tickers in SECTORS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing sector: {sector}")
        logger.info(f"{'='*50}")
        
        for ticker in tickers:
            processed += 1
            logger.info(f"[{processed}/{total_tickers}] Processing {ticker}")
            
            try:
                if update_only:
                    update_ticker_data(ticker, sector)
                else:
                    df = fetch_ticker_data(ticker)
                    save_ticker_data(df, ticker, sector)
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {str(e)}")
                continue
    
    logger.info(f"\nData fetch complete! Processed {processed} tickers.")


def check_data_status():
    """
    Check the status of all stored data files.
    Returns a summary of data availability and last update dates.
    """
    status = {}
    
    for sector, tickers in SECTORS.items():
        status[sector] = {}
        for ticker in tickers:
            csv_path = get_csv_path(ticker, sector)
            
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=['Date'])
                last_date = df['Date'].max()
                row_count = len(df)
                first_date = df['Date'].min()
                status[sector][ticker] = {
                    'exists': True,
                    'rows': row_count,
                    'first_date': first_date.strftime('%Y-%m-%d'),
                    'last_date': last_date.strftime('%Y-%m-%d')
                }
            else:
                status[sector][ticker] = {
                    'exists': False,
                    'rows': 0,
                    'first_date': None,
                    'last_date': None
                }
    
    return status


def print_data_status():
    """Print a formatted summary of data status."""
    status = check_data_status()
    
    print("\n" + "="*80)
    print("MARKET DATA STATUS REPORT")
    print("="*80)
    
    for sector, tickers in status.items():
        print(f"\n{sector}:")
        print("-" * 60)
        
        for ticker, info in tickers.items():
            if info['exists']:
                print(f"  {ticker:15} | {info['rows']:6} rows | {info['first_date']} to {info['last_date']}")
            else:
                print(f"  {ticker:15} | NO DATA")
    
    print("\n" + "="*80)


def get_all_tickers() -> list:
    """Get a flat list of all tickers."""
    all_tickers = []
    for tickers in SECTORS.values():
        all_tickers.extend(tickers)
    return all_tickers


def get_ticker_sector(ticker: str) -> str:
    """Get the sector for a given ticker."""
    for sector, tickers in SECTORS.items():
        if ticker in tickers:
            return sector
    return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch market data from YFinance')
    parser.add_argument('--full', action='store_true', help='Fetch full history (not just updates)')
    parser.add_argument('--sector', type=str, help='Fetch only specific sector')
    parser.add_argument('--status', action='store_true', help='Show data status only')
    
    args = parser.parse_args()
    
    if args.status:
        print_data_status()
    elif args.sector:
        fetch_all_sector_data(args.sector, update_only=not args.full)
    else:
        fetch_all_data(update_only=not args.full)
        print_data_status()
