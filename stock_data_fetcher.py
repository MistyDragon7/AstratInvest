import yfinance as yf
import pandas as pd
import numpy as np
from technical_indicators import TechnicalIndicators
from typing import List

class StockDataFetcher:
    """Fetch and preprocess stock data"""

    def __init__(self, stock_list: List[str] | None = None, start_date: str | None = None, end_date: str | None = None, interval="1d"):
        self.stock_list = stock_list if stock_list is not None else []
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.stock_data = {}
        self.returns_matrix = None
        self.market_caps = {}

    def fetch_all_stocks(self):
        """Fetch data for all stocks with improved market cap handling"""
        print(f"Fetching data for {len(self.stock_list)} stocks...")

        if not self.stock_list:
            print("No stocks provided to fetch data for. Skipping fetching.")
            return {}

        for i, ticker in enumerate(self.stock_list):
            try:
                print(f"Fetching {ticker} ({i+1}/{len(self.stock_list)})")
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date, interval=self.interval)

                if not df.empty and len(df) > 100:
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert(None).normalize()
                    else:
                        df.index = df.index.normalize()
                    df = df.dropna()
                    self.stock_data[ticker] = df

                    # ✅ IMPROVED: Better market cap fetching with fallbacks
                    market_cap = self._get_market_cap(stock, ticker)
                    self.market_caps[ticker] = market_cap

                    print(f"✓ {ticker}: {df.shape[0]} records, Market Cap: {market_cap:,.0f}")
                else:
                    print(f"✗ {ticker}: Insufficient data")

            except Exception as e:
                print(f"✗ Error fetching {ticker}: {str(e)}")

        print(f"Successfully fetched {len(self.stock_data)} stocks")
        print(f"Market caps summary: {len([k for k, v in self.market_caps.items() if v > 0])} valid market caps")
        return self.stock_data

    def _get_market_cap(self, stock, ticker):
        """Get market cap with multiple fallback methods"""
        try:
            info = stock.info
            market_cap = info.get('marketCap')
            
            if market_cap and market_cap > 0:
                return market_cap
            
            shares_outstanding = info.get('sharesOutstanding')
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if shares_outstanding and current_price:
                calculated_market_cap = shares_outstanding * current_price
                if calculated_market_cap > 0:
                    print(f"  📊 {ticker}: Calculated market cap from shares * price")
                    return calculated_market_cap
            
            enterprise_value = info.get('enterpriseValue')
            if enterprise_value and enterprise_value > 0:
                print(f"  📊 {ticker}: Using enterprise value as market cap proxy")
                return enterprise_value
            
            history = stock.history(period="5d")
            if not history.empty:
                avg_price = history['Close'].mean()
                estimated_market_cap = avg_price * 1e9
                print(f"{ticker}: Estimated market cap based on price")
                return estimated_market_cap
            
            
            default_cap = 5e10  # 50 billion INR default for Indian stocks
            
            print(f"  ⚠️ {ticker}: Using default market cap ({default_cap:,.0f})")
            return default_cap
            
        except Exception as e:
            print(f"  ⚠️ {ticker}: Error getting market cap ({e}), using default")
            return 1e12  # Default fallback

    def add_technical_indicators(self):
        """Add technical indicators to all stocks"""
        print("Adding technical indicators...")
    
        for ticker in self.stock_data.keys():
            df = self.stock_data[ticker].copy()
    
            if 'RSI' in df.columns:
                continue
            
            df['Returns'] = TechnicalIndicators.calculate_returns(df['Close'])
            df['Volatility'] = TechnicalIndicators.calculate_volatility(df['Returns'])
            df['MA_10'] = TechnicalIndicators.moving_average(df['Close'], 10)
            df['MA_20'] = TechnicalIndicators.moving_average(df['Close'], 20)
            df['EMA_12'] = TechnicalIndicators.exponential_moving_average(df['Close'], 12)
            df['RSI'] = TechnicalIndicators.rsi(df['Close'])
    
            macd, signal = TechnicalIndicators.macd(df['Close'])
            df['MACD'] = macd
            df['MACD_Signal'] = signal
    
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['Close'])
            df['BB_Upper'] = bb_upper
            df['BB_Lower'] = bb_lower
            df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
    
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Price_Momentum'] = df['Close'].pct_change(5)
    
            self.stock_data[ticker] = df

    def create_returns_matrix(self):
        """Create returns matrix for all stocks"""
        returns_data = {}
        common_dates = None # Initialize to None

        for ticker, df in self.stock_data.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        if common_dates is None:
            common_dates = [] # Ensure it's an empty list if no data was processed

        common_dates = sorted(list(common_dates))

        for ticker, df in self.stock_data.items():
            returns_data[ticker] = []
            for date in common_dates:
                if date in df.index and 'Returns' in df.columns:
                    ret = df.loc[date, 'Returns']
                    returns_data[ticker].append(ret if not pd.isna(ret) else 0)
                else:
                    returns_data[ticker].append(0)

        self.returns_matrix = pd.DataFrame(returns_data, index=pd.Index(common_dates))
        print(f"Debug: Number of common dates: {len(common_dates)}")
        if not common_dates:
            print("Debug: No common dates found across all stocks in the specified period.")
        return self.returns_matrix

    def debug_market_caps(self):
        """Debug method to check market cap data"""
        print("\n" + "="*50)
        print("MARKET CAPS DEBUG")
        print("="*50)
        
        for ticker, market_cap in self.market_caps.items():
            print(f"{ticker}: {market_cap:,.0f}")
        
        valid_caps = {k: v for k, v in self.market_caps.items() if v > 0}
        print(f"\nValid market caps: {len(valid_caps)}/{len(self.market_caps)}")
        
        if valid_caps:
            print(f"Min: {min(valid_caps.values()):,.0f}")
            print(f"Max: {max(valid_caps.values()):,.0f}")
            print(f"Avg: {sum(valid_caps.values())/len(valid_caps):,.0f}")
        
        return valid_caps