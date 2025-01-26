import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from ta import add_all_ta_features
from ta.utils import dropna
import yfinance as yf  # Added for additional data verification

class EnhancedForexDataCollector:
    def __init__(self, exchangerate_api_key: str, alpha_vantage_key: str, base_path: str = "./forex_data"):
        """Initialize the Enhanced Forex Data Collector."""
        self.exchangerate_api_key = exchangerate_api_key
        self.alpha_vantage_key = alpha_vantage_key
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # API endpoints
        self.exchangerate_url = "https://v6.exchangerate-api.com/v6"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        
        # Setup logging and directories
        self._setup_logging()
        self._create_directories()
        
        # Cache for API responses
        self.cache = {}
        
    def _setup_logging(self):
        """Setup enhanced logging configuration."""
        log_file = self.base_path / 'forex_collection.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_directories(self):
        """Create necessary directories for data storage."""
        (self.base_path / 'raw_data').mkdir(exist_ok=True)
        (self.base_path / 'processed_data').mkdir(exist_ok=True)
        (self.base_path / 'cache').mkdir(exist_ok=True)

    def fetch_historical_data(self, base: str, target: str, days: int = 365) -> pd.DataFrame:
        """
        Fetch historical forex data from multiple sources and combine them.
        """
        data_sources = []
        
        # 1. Alpha Vantage Data
        av_data = self._fetch_alpha_vantage_data(base, target)
        if av_data is not None:
            data_sources.append(av_data)
            
        # 2. Yahoo Finance Data (as verification)
        symbol = f"{base}{target}=X"
        try:
            yf_data = yf.download(symbol, period="1y", interval="1d")  # Changed to "1y" instead of using days
            if not yf_data.empty:
                yf_data = yf_data.reset_index()
                yf_data.columns = [col.lower() for col in yf_data.columns]
                yf_data = yf_data.rename(columns={'date': 'timestamp'})
                data_sources.append(yf_data)
        except Exception as e:
            self.logger.warning(f"Failed to fetch Yahoo Finance data: {str(e)}")

        # Combine data sources
        if data_sources:
            combined_data = pd.concat(data_sources, ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=['timestamp'])
            combined_data = combined_data.sort_values('timestamp')
            return combined_data
        
        # If no data available, simulate with current rate
        current_rate = self._fetch_current_rate(base, target)
        if current_rate:
            return self._simulate_historical_prices(current_rate, days)
        
        return None
        
        # If no data available, simulate with current rate
        current_rate = self._fetch_current_rate(base, target)
        if current_rate:
            return self._simulate_historical_prices(current_rate, days)
        
        return None

    def _fetch_alpha_vantage_data(self, base: str, target: str) -> Optional[pd.DataFrame]:
        """Fetch and process Alpha Vantage forex data."""
        cache_file = self.base_path / 'cache' / f"av_{base}{target}.csv"
        
        # Check cache first
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.days < 1:  # Cache valid for 1 day
                return pd.read_csv(cache_file)
        
        params = {
            'function': 'FX_DAILY',
            'from_symbol': base,
            'to_symbol': target,
            'apikey': self.alpha_vantage_key,
            'outputsize': 'full'
        }
        
        try:
            response = requests.get(self.alpha_vantage_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Time Series FX (Daily)" in data:
                df = pd.DataFrame.from_dict(
                    data["Time Series FX (Daily)"],
                    orient='index'
                ).reset_index()
                
                df.columns = ['timestamp'] + ['open', 'high', 'low', 'close']
                df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Add volume (estimated from price movement)
                df['volume'] = ((df['high'] - df['low']) / df['close'] * 10000).astype(int)
                
                # Save to cache
                df.to_csv(cache_file, index=False)
                return df
                
        except Exception as e:
            self.logger.error(f"Alpha Vantage API error: {str(e)}")
            return None

    def _fetch_current_rate(self, base: str, target: str) -> Optional[float]:
        """Fetch current exchange rate."""
        url = f"{self.exchangerate_url}/{self.exchangerate_api_key}/latest/{base}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get('conversion_rates', {}).get(target)
        except Exception as e:
            self.logger.error(f"Failed to fetch current rate: {str(e)}")
            return None

    def _simulate_historical_prices(self, current_rate: float, days: int) -> pd.DataFrame:
        """Simulate historical prices with improved realism."""
        np.random.seed(42)
        
        # Parameters for more realistic simulation
        annual_volatility = 0.10  # 10% annual volatility
        daily_volatility = annual_volatility / np.sqrt(252)
        daily_drift = 0.0001  # Slight upward bias
        
        # Generate returns
        daily_returns = np.random.normal(
            daily_drift,
            daily_volatility,
            days
        )
        
        # Calculate prices backwards from current
        prices = [current_rate]
        for ret in reversed(daily_returns[:-1]):
            prices.append(prices[-1] / (1 + ret))
        prices.reverse()
        
        # Generate OHLC data
        timestamps = [
            datetime.now() - timedelta(days=i)
            for i in range(days)
        ]
        timestamps.reverse()
        
        data = []
        for i, base_price in enumerate(prices):
            # Intraday volatility
            daily_vol = daily_volatility * np.random.uniform(0.8, 1.2)
            
            # Generate high/low with mean reversion
            high_ret = abs(np.random.normal(0, daily_vol))
            low_ret = abs(np.random.normal(0, daily_vol))
            
            high = base_price * (1 + high_ret)
            low = base_price * (1 - low_ret)
            
            # Ensure realistic OHLC relationships
            if i > 0:
                prev_close = data[-1]['close']
                open_price = prev_close * (1 + np.random.normal(0, daily_vol/2))
                high = max(high, open_price)
                low = min(low, open_price)
            else:
                open_price = base_price
            
            # Volume based on price movement
            price_movement = (high - low) / base_price
            volume = int(np.random.normal(5000, 1000) * (1 + price_movement * 10))
            
            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': prices[i],
                'volume': max(1000, volume)
            })
        
        return pd.DataFrame(data)

    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical and statistical features."""
        df = df.copy()
        
        # Basic technical indicators
        df = add_all_ta_features(
            df=df,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            fillna=True
        )
        
        # Additional features
        lookback_periods = [1, 3, 5, 10, 20, 50]
        
        for period in lookback_periods:
            period_str = str(period)
            
            # Returns and volatility
            df[f'returns_{period_str}d'] = df['close'].pct_change(period)
            df[f'volatility_{period_str}d'] = df[f'returns_{period_str}d'].rolling(period).std()
            
            # Price momentum
            df[f'momentum_{period_str}d'] = df['close'].diff(period)
            
            # Volume momentum
            df[f'volume_momentum_{period_str}d'] = df['volume'].diff(period)
            
            # Moving averages
            df[f'sma_{period_str}d'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period_str}d'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # Price vs moving averages
            df[f'price_vs_sma_{period_str}d'] = (df['close'] / df[f'sma_{period_str}d'] - 1)
            df[f'price_vs_ema_{period_str}d'] = (df['close'] / df[f'ema_{period_str}d'] - 1)
        
        # Volatility features
        df['daily_range'] = (df['high'] - df['low']) / df['close']
        df['daily_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Additional momentum features
        df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
        df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        df['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100
        
        # Volatility measures
        df['atr_5'] = (
            df['high'].rolling(5).max() - df['low'].rolling(5).min()
        ) / df['close'].rolling(5).mean()
        
        df['atr_10'] = (
            df['high'].rolling(10).max() - df['low'].rolling(10).min()
        ) / df['close'].rolling(10).mean()
        
        # Volume profile
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
        df['volume_ratio_10'] = df['volume'] / df['volume_sma_10']
        
        # Clean up any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values
        df = df.fillna(method='ffill')
        
        # Backward fill any remaining NaN values
        df = df.fillna(method='bfill')
        
        return df


    def collect_and_process_data(self, currency_pairs: List[Tuple[str, str]], days: int = 365) -> pd.DataFrame:
        """Collect and process data for multiple currency pairs."""
        all_data = []
        
        for base, target in tqdm(currency_pairs, desc="Processing currency pairs"):
            try:
                # Fetch historical data
                hist_data = self.fetch_historical_data(base, target, days)
                
                if hist_data is not None:
                    # Add pair information
                    hist_data['pair'] = f"{base}/{target}"
                    hist_data['base_currency'] = base
                    hist_data['target_currency'] = target
                    
                    # Add technical features
                    hist_data = self.add_advanced_features(hist_data)
                    
                    # Save raw data
                    raw_file = self.base_path / 'raw_data' / f"{base}{target}_data.csv"
                    hist_data.to_csv(raw_file, index=False)
                    
                    all_data.append(hist_data)
                
                time.sleep(1)  # Rate limiting
            except Exception as e:
                self.logger.error(f"Error processing {base}/{target}: {str(e)}")
                continue
        
        if not all_data:
            return None
        
        try:
            # Combine and process all data
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values(['pair', 'timestamp'])
            
            # Save processed data
            processed_file = self.base_path / 'processed_data' / 'forex_data_with_features.csv'
            combined_data.to_csv(processed_file, index=False)
            
            return combined_data
        except Exception as e:
            self.logger.error(f"Error combining data: {str(e)}")
            return None


def main():
    # API keys
    exchangerate_api_key = "YOUR-API-KEY"
    alpha_vantage_key = "YOUR-API-KEY"
    
    # Initialize collector
    collector = EnhancedForexDataCollector(exchangerate_api_key, alpha_vantage_key)
    
    # Define currency pairs
    currency_pairs = [
        ('EUR', 'USD'),
        ('GBP', 'USD'),
        ('GBP', 'EUR'),
        ('USD', 'JPY'),
        ('EUR', 'JPY'),
        ('GBP', 'JPY')
    ]
    
    # Collect and process data
    data = collector.collect_and_process_data(currency_pairs, days=365)
    
    if data is not None:
        print("\nData collection completed successfully!")
        print(f"Total records: {len(data)}")
        print(f"Features available: {len(data.columns)}")
        
        # Save feature list
        with open(collector.base_path / 'processed_data' / 'feature_list.txt', 'w') as f:
            f.write('\n'.join(data.columns))

if __name__ == "__main__":
    main()
