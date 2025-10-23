"""
Ultimate Portfolio Analytics, Backtesting, Simulation & ML Prediction Dashboard
Enhanced with Comprehensive Indian Market Coverage & Multiple ML Models
"""

import streamlit as st
import yfinance as yf
import quantstats as qs
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tempfile
import os
import io
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Ultimate Portfolio Analytics with Complete Indian Market Coverage",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🇮🇳 Ultimate Portfolio Analytics with Complete Indian Market Coverage")
st.markdown("---")

# Comprehensive Market Data Definitions
class IndianMarketData:
    """Comprehensive Indian market data including all major indices, stocks, and asset classes"""
    
    # NIFTY 50 Stocks
    NIFTY_50 = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'SBIN.NS',
        'BHARTIARTL.NS', 'ASIANPAINT.NS', 'HCLTECH.NS', 'AXISBANK.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'M&M.NS', 'BAJFINANCE.NS',
        'WIPRO.NS', 'NESTLEIND.NS', 'HDFC.NS', 'ONGC.NS', 'TATASTEEL.NS',
        'POWERGRID.NS', 'NTPC.NS', 'INDUSINDBK.NS', 'COALINDIA.NS', 'TECHM.NS',
        'BAJAJFINSV.NS', 'ADANIPORTS.NS', 'JSWSTEEL.NS', 'HDFCLIFE.NS', 'DRREDDY.NS',
        'BRITANNIA.NS', 'CIPLA.NS', 'GRASIM.NS', 'SHREECEM.NS', 'APOLLOHOSP.NS',
        'TATAMOTORS.NS', 'SBILIFE.NS', 'HINDALCO.NS', 'DIVISLAB.NS', 'BPCL.NS',
        'EICHERMOT.NS', 'UPL.NS', 'HEROMOTOCO.NS', 'ADANIENT.NS', 'ADANIGREEN.NS'
    ]
    
    # BANKNIFTY Stocks
    BANKNIFTY = [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS',
        'INDUSINDBK.NS', 'BANKBARODA.NS', 'PNB.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS',
        'AUBANK.NS', 'BANDHANBNK.NS'
    ]
    
    # NIFTY Midcap 150 representative stocks
    NIFTY_MIDCAP = [
        'ADANIPOWER.NS', 'ADANIGREEN.NS', 'ADANIENSOL.NS', 'ADANIENERGY.NS', 'ADANIPORTS.NS',
        'AMBUJACEM.NS', 'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'ASHOKLEY.NS', 'ASTRAL.NS',
        'AUBANK.NS', 'BAJAJELEC.NS', 'BALKRISIND.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS',
        'BANKINDIA.NS', 'BEL.NS', 'BERGEPAINT.NS', 'BHARATFORG.NS', 'BHEL.NS',
        'BIOCON.NS', 'BOSCHLTD.NS', 'CANBK.NS', 'CHOLAFIN.NS', 'CIPLA.NS',
        'COALINDIA.NS', 'COLPAL.NS', 'CONCOR.NS', 'CUMMINSIND.NS', 'DABUR.NS'
    ]
    
    # NIFTY Smallcap 250 representative stocks
    NIFTY_SMALLCAP = [
        'AARTIIND.NS', 'ABBOTINDIA.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'ACC.NS',
        'AIAENG.NS', 'AJANTPHARM.NS', 'ALBK.NS', 'ALKEM.NS', 'AMARAJABAT.NS',
        'APLLTD.NS', 'APOLLOHOSP.NS', 'ARVIND.NS', 'ASAHIINDIA.NS', 'ASHOKA.NS',
        'ASTRAL.NS', 'ATUL.NS', 'AUBANK.NS', 'AUROPHARMA.NS', 'BAJAJCON.NS',
        'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS',
        'BANKBARODA.NS', 'BANKINDIA.NS', 'BATAINDIA.NS', 'BBTC.NS', 'BEL.NS'
    ]
    
    # Indian Indices
    INDIAN_INDICES = {
        'Nifty 50': '^NSEI',
        'Bank Nifty': '^NSEBANK',
        'Nifty Midcap 100': 'NIFTY_MIDCAP_100.NS',
        'Nifty Smallcap 100': 'NIFTY_SMALLCAP_100.NS',
        'Nifty 500': '^CRSLDX',
        'Nifty Auto': 'NIFTY_AUTO.NS',
        'Nifty Pharma': 'NIFTY_PHARMA.NS',
        'Nifty IT': 'NIFTY_IT.NS',
        'Nifty FMCG': 'NIFTY_FMCG.NS',
        'Nifty Metal': 'NIFTY_METAL.NS',
        'Nifty Realty': 'NIFTY_REALTY.NS',
        'Nifty Energy': 'NIFTY_ENERGY.NS',
        'Nifty Infrastructure': 'NIFTY_INFRA.NS',
        'India VIX': '^INDIAVIX'
    }
    
    # Global Indices
    GLOBAL_INDICES = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'Dow Jones': '^DJI',
        'FTSE 100': '^FTSE',
        'DAX': '^GDAXI',
        'Nikkei 225': '^N225',
        'Hang Seng': '^HSI',
        'Shanghai Composite': '000001.SS',
        'CAC 40': '^FCHI'
    }
    
    # Forex Pairs
    FOREX = {
        'USD/INR': 'INR=X',
        'EUR/INR': 'EURINR=X',
        'GBP/INR': 'GBPINR=X',
        'JPY/INR': 'JPYINR=X',
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'JPY=X',
        'USD/CHF': 'CHF=X',
        'AUD/USD': 'AUDUSD=X',
        'USD/CAD': 'CAD=X'
    }
    
    # Commodities
    COMMODITIES = {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Crude Oil': 'CL=F',
        'Brent Crude': 'BZ=F',
        'Natural Gas': 'NG=F',
        'Copper': 'HG=F',
        'Aluminum': 'ALI=F',
        'Zinc': 'ZI=F',
        'Wheat': 'ZW=F',
        'Corn': 'ZC=F',
        'Soybean': 'ZS=F'
    }
    
    # Cryptocurrencies
    CRYPTO = {
        'Bitcoin USD': 'BTC-USD',
        'Ethereum USD': 'ETH-USD',
        'Binance Coin USD': 'BNB-USD',
        'Cardano USD': 'ADA-USD',
        'XRP USD': 'XRP-USD',
        'Solana USD': 'SOL-USD',
        'Polkadot USD': 'DOT-USD',
        'Dogecoin USD': 'DOGE-USD',
        'Shiba Inu USD': 'SHIB-USD',
        'Avalanche USD': 'AVAX-USD',
        'Polygon USD': 'MATIC-USD',
        'Litecoin USD': 'LTC-USD'
    }

class LSTMModel(nn.Module):
    """LSTM Model for Time Series Forecasting"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 50, output_dim: int = 1, num_layers: int = 2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out

class GRUModel(nn.Module):
    """GRU Model for Time Series Forecasting"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 50, output_dim: int = 1, num_layers: int = 2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out

class TimeSeriesDataset(Dataset):
    """Custom Dataset for Time Series Data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EnhancedPortfolioAnalyzer:
    """Enhanced Portfolio Analyzer with Complete Market Coverage and Multiple ML Models"""
    
    def __init__(self, tickers: List[str], weights: List[float], start_date: str, end_date: str, 
                 benchmark: str = '^NSEI', initial_capital: float = 100000):
        self.tickers = tickers
        self.weights = weights
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.initial_capital = initial_capital
        self.returns = None
        self.portfolio_value = None
        self.ml_models = {}
        self.scalers = {}
        self.technical_indicators = {}
        
    def fetch_extended_data(self) -> pd.DataFrame:
        """Fetch historical price data with extended period (10-15 years)"""
        try:
            st.info(f"Fetching data from {self.start_date} to {self.end_date} for {len(self.tickers)} assets...")
            
            # Adjust start date to ensure 10+ years of data
            start_dt = pd.to_datetime(self.start_date)
            end_dt = pd.to_datetime(self.end_date)
            
            # Ensure minimum 10 years of data
            if (end_dt - start_dt).days < 3650:
                start_dt = end_dt - pd.DateOffset(years=15)
                self.start_date = start_dt.strftime("%Y-%m-%d")
                st.warning(f"Adjusted start date to {self.start_date} to ensure sufficient historical data")
            
            # Fetch portfolio data
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=True)['Close']
            benchmark_data = yf.download(self.benchmark, start=self.start_date, end=self.end_date, auto_adjust=True)['Close']
            
            # Handle single ticker case
            if len(self.tickers) == 1:
                data = pd.DataFrame(data)
                data.columns = self.tickers
                
            # Align dates and handle missing data
            common_dates = data.index.intersection(benchmark_data.index)
            if len(common_dates) == 0:
                st.error("No common dates found between portfolio and benchmark data")
                return pd.DataFrame()
                
            data = data.loc[common_dates]
            benchmark_data = benchmark_data.loc[common_dates]
            
            # Forward fill missing data
            data = data.ffill().bfill()
            benchmark_data = benchmark_data.ffill().bfill()
            
            # Remove assets with too many missing values
            missing_threshold = 0.1  # 10% missing values
            assets_to_keep = []
            for asset in data.columns:
                missing_ratio = data[asset].isna().sum() / len(data)
                if missing_ratio < missing_threshold:
                    assets_to_keep.append(asset)
                else:
                    st.warning(f"Removing {asset} due to {missing_ratio:.1%} missing data")
            
            data = data[assets_to_keep]
            
            # Update weights for remaining assets
            if len(assets_to_keep) < len(self.tickers):
                remaining_indices = [self.tickers.index(asset) for asset in assets_to_keep]
                remaining_weights = [self.weights[i] for i in remaining_indices]
                # Normalize remaining weights
                total_weight = sum(remaining_weights)
                self.weights = [w/total_weight for w in remaining_weights]
                self.tickers = assets_to_keep
                st.info(f"Updated portfolio with {len(self.tickers)} assets after data cleaning")
            
            self.data = data
            self.benchmark_data = benchmark_data
            
            st.success(f"Successfully fetched data for {len(self.tickers)} assets with {len(data)} trading days")
            return data
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def compute_returns(self) -> Dict[str, pd.Series]:
        """Compute portfolio and benchmark returns"""
        if self.data.empty:
            return {}
            
        try:
            # Calculate returns
            portfolio_returns = (self.data.pct_change().dropna() * self.weights).sum(axis=1)
            benchmark_returns = self.benchmark_data.pct_change().dropna()
            
            # Align returns
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_returns = portfolio_returns.loc[common_dates]
            benchmark_returns = benchmark_returns.loc[common_dates]
            
            # Remove outliers (returns beyond 3 standard deviations)
            returns_std = portfolio_returns.std()
            portfolio_returns = portfolio_returns[np.abs(portfolio_returns) < 3 * returns_std]
            benchmark_returns = benchmark_returns.loc[portfolio_returns.index]
            
            # Calculate portfolio value
            portfolio_value = (1 + portfolio_returns).cumprod() * self.initial_capital
            
            self.returns = {
                'portfolio': portfolio_returns,
                'benchmark': benchmark_returns
            }
            self.portfolio_value = portfolio_value
            
            st.success(f"Computed returns for {len(portfolio_returns)} trading days")
            return self.returns
            
        except Exception as e:
            st.error(f"Error computing returns: {str(e)}")
            return {}
    
    def compute_technicals(self, price_series: pd.Series) -> pd.DataFrame:
        """Compute comprehensive technical indicators for ML features"""
        try:
            technicals = pd.DataFrame(index=price_series.index)
            
            # Price-based features
            technicals['returns'] = price_series.pct_change()
            technicals['log_returns'] = np.log(price_series / price_series.shift(1))
            technicals['price_sma_20'] = ta.sma(price_series, length=20)
            technicals['price_sma_50'] = ta.sma(price_series, length=50)
            technicals['price_ema_12'] = ta.ema(price_series, length=12)
            technicals['price_ema_26'] = ta.ema(price_series, length=26)
            
            # Momentum indicators
            technicals['rsi_14'] = ta.rsi(price_series, length=14)
            technicals['stoch_k'] = ta.stoch(price_series, price_series, price_series)['STOCHk_14_3_3']
            technicals['stoch_d'] = ta.stoch(price_series, price_series, price_series)['STOCHd_14_3_3']
            technicals['williams_r'] = ta.willr(price_series, price_series, price_series, length=14)
            technicals['cci'] = ta.cci(price_series, price_series, price_series, length=20)
            
            # MACD
            macd = ta.macd(price_series)
            if macd is not None:
                technicals['macd'] = macd['MACD_12_26_9']
                technicals['macd_signal'] = macd['MACDs_12_26_9']
                technicals['macd_hist'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands
            bb = ta.bbands(price_series, length=20)
            if bb is not None:
                technicals['bb_upper'] = bb['BBU_20_2.0']
                technicals['bb_middle'] = bb['BBM_20_2.0']
                technicals['bb_lower'] = bb['BBL_20_2.0']
                technicals['bb_width'] = (technicals['bb_upper'] - technicals['bb_lower']) / technicals['bb_middle']
                technicals['bb_position'] = (price_series - technicals['bb_lower']) / (technicals['bb_upper'] - technicals['bb_lower'])
            
            # Volatility
            technicals['volatility_20'] = technicals['returns'].rolling(window=20).std()
            technicals['volatility_50'] = technicals['returns'].rolling(window=50).std()
            technicals['atr'] = ta.atr(price_series, price_series, price_series, length=14)
            
            # Volume indicators (if volume data available)
            if hasattr(self, 'volume_data'):
                volume = self.volume_data.mean(axis=1) if len(self.tickers) > 1 else self.volume_data
                technicals['volume_sma_20'] = ta.sma(volume, length=20)
                technicals['volume_ratio'] = volume / technicals['volume_sma_20']
                technicals['obv'] = ta.obv(price_series, volume)
            
            # Additional features
            technicals['momentum_10'] = price_series / price_series.shift(10) - 1
            technicals['momentum_20'] = price_series / price_series.shift(20) - 1
            technicals['price_zscore_20'] = (price_series - technicals['price_sma_20']) / (price_series.rolling(20).std())
            
            # Drop NaN values and ensure sufficient data
            technicals = technicals.dropna()
            
            if len(technicals) < 100:
                st.warning(f"Limited technical indicators data: {len(technicals)} points")
            
            self.technical_indicators = technicals
            return technicals
            
        except Exception as e:
            st.warning(f"Technical indicators computation limited: {str(e)}")
            # Fallback to basic indicators
            technicals = pd.DataFrame(index=price_series.index)
            technicals['returns'] = price_series.pct_change()
            technicals['price_sma_20'] = price_series.rolling(20).mean()
            technicals['price_sma_50'] = price_series.rolling(50).mean()
            technicals = technicals.dropna()
            self.technical_indicators = technicals
            return technicals
    
    def compute_metrics(self, risk_free_rate: float = 0.04) -> Dict[str, float]:
        """Compute comprehensive portfolio performance metrics"""
        if self.returns is None:
            return {}
            
        portfolio_returns = self.returns['portfolio']
        benchmark_returns = self.returns['benchmark']
        
        try:
            # Basic metrics
            total_return = (self.portfolio_value.iloc[-1] / self.initial_capital - 1) * 100
            annual_return = portfolio_returns.mean() * 252 * 100
            annual_volatility = portfolio_returns.std() * np.sqrt(252) * 100
            
            # Risk-adjusted metrics
            excess_returns = portfolio_returns - (risk_free_rate / 252)
            sharpe_ratio = (annual_return - risk_free_rate * 100) / annual_volatility if annual_volatility > 0 else 0
            
            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # Beta calculation
            covariance = portfolio_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha
            benchmark_annual_return = benchmark_returns.mean() * 252 * 100
            alpha = annual_return - (risk_free_rate * 100 + beta * (benchmark_annual_return - risk_free_rate * 100))
            
            # Sortino Ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
            sortino_ratio = (annual_return - risk_free_rate * 100) / downside_volatility if downside_volatility > 0 else 0
            
            # Calmar Ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # VaR and CVaR (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5) * 100
            cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100
            
            # Additional metrics
            win_rate = (portfolio_returns > 0).mean() * 100
            profit_loss_ratio = abs(portfolio_returns[portfolio_returns > 0].mean() / portfolio_returns[portfolio_returns < 0].mean()) if portfolio_returns[portfolio_returns < 0].mean() != 0 else float('inf')
            
            # Skewness and Kurtosis
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurtosis()
            
            # Information Ratio
            active_returns = portfolio_returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252) * 100
            information_ratio = (annual_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0
            
            metrics = {
                'Total Return (%)': total_return,
                'Annual Return (%)': annual_return,
                'Annual Volatility (%)': annual_volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Max Drawdown (%)': max_drawdown,
                'Calmar Ratio': calmar_ratio,
                'Beta': beta,
                'Alpha (%)': alpha,
                'Information Ratio': information_ratio,
                'Tracking Error (%)': tracking_error,
                'VaR (95%) (%)': var_95,
                'CVaR (95%) (%)': cvar_95,
                'Win Rate (%)': win_rate,
                'Profit/Loss Ratio': profit_loss_ratio,
                'Skewness': skewness,
                'Kurtosis': kurtosis
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error computing metrics: {str(e)}")
            return {}
    
    def prepare_ml_features(self, sequence_length: int = 60, forecast_horizon: int = 1) -> Tuple[Any, Any, Any, Any]:
        """Prepare features and targets for ML models with enhanced feature engineering"""
        try:
            if self.returns is None:
                return None, None, None, None
            
            # Use portfolio value for technical indicators
            price_series = self.portfolio_value
            
            # Compute technical indicators
            technicals = self.compute_technicals(price_series)
            
            if len(technicals) < sequence_length + forecast_horizon:
                st.warning(f"Insufficient data for ML: {len(technicals)} points, need {sequence_length + forecast_horizon}")
                return None, None, None, None
            
            # Create features (X) and targets (y)
            features = []
            targets = []
            
            # Select feature columns (exclude target-related columns)
            feature_columns = [col for col in technicals.columns if col not in ['returns', 'log_returns']]
            
            for i in range(sequence_length, len(technicals) - forecast_horizon):
                # Sequence features
                seq_features = []
                for j in range(sequence_length):
                    idx = i - sequence_length + j
                    feature_row = []
                    for col in feature_columns:
                        feature_row.append(technicals[col].iloc[idx])
                    seq_features.append(feature_row)
                
                features.append(seq_features)
                
                # Target: future return (regression)
                future_return = (price_series.iloc[i + forecast_horizon] / price_series.iloc[i] - 1)
                targets.append(future_return)
            
            features = np.array(features)
            targets = np.array(targets)
            
            if len(features) == 0:
                st.warning("No features created - check data sufficiency")
                return None, None, None, None
            
            # Split data (80/20)
            split_idx = int(0.8 * len(features))
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = targets[:split_idx], targets[split_idx:]
            
            st.info(f"ML Features: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            st.error(f"Error preparing ML features: {str(e)}")
            return None, None, None, None
    
    def train_ml_models(self, model_types: List[str], sequence_length: int = 60, forecast_horizon: int = 1):
        """Train multiple ML models for portfolio forecasting with enhanced model selection"""
        try:
            # Prepare features
            X_train, X_test, y_train, y_test = self.prepare_ml_features(sequence_length, forecast_horizon)
            
            if X_train is None or len(X_train) == 0:
                st.warning("Insufficient data for ML training. Need more historical data.")
                return
            
            # Reshape data for different model types
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_flat)
            X_test_scaled = scaler.transform(X_test_flat)
            
            self.scalers['standard'] = scaler
            
            models = {}
            performance = {}
            
            st.info(f"Training {len(model_types)} ML models...")
            
            # Linear Regression
            if 'Linear Regression' in model_types:
                with st.spinner("Training Linear Regression..."):
                    lr_model = LinearRegression()
                    lr_model.fit(X_train_scaled, y_train)
                    models['Linear Regression'] = lr_model
                    lr_pred = lr_model.predict(X_test_scaled)
                    performance['Linear Regression'] = {
                        'mse': mean_squared_error(y_test, lr_pred),
                        'mae': mean_absolute_error(y_test, lr_pred),
                        'r2': lr_model.score(X_test_scaled, y_test)
                    }
            
            # Random Forest
            if 'Random Forest' in model_types:
                with st.spinner("Training Random Forest..."):
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    rf_model.fit(X_train_scaled, y_train)
                    models['Random Forest'] = rf_model
                    rf_pred = rf_model.predict(X_test_scaled)
                    performance['Random Forest'] = {
                        'mse': mean_squared_error(y_test, rf_pred),
                        'mae': mean_absolute_error(y_test, rf_pred),
                        'r2': rf_model.score(X_test_scaled, y_test)
                    }
            
            # Gradient Boosting
            if 'Gradient Boosting' in model_types:
                with st.spinner("Training Gradient Boosting..."):
                    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    gb_model.fit(X_train_scaled, y_train)
                    models['Gradient Boosting'] = gb_model
                    gb_pred = gb_model.predict(X_test_scaled)
                    performance['Gradient Boosting'] = {
                        'mse': mean_squared_error(y_test, gb_pred),
                        'mae': mean_absolute_error(y_test, gb_pred),
                        'r2': gb_model.score(X_test_scaled, y_test)
                    }
            
            # Support Vector Regression
            if 'SVR' in model_types:
                with st.spinner("Training SVR..."):
                    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                    svr_model.fit(X_train_scaled, y_train)
                    models['SVR'] = svr_model
                    svr_pred = svr_model.predict(X_test_scaled)
                    performance['SVR'] = {
                        'mse': mean_squared_error(y_test, svr_pred),
                        'mae': mean_absolute_error(y_test, svr_pred)
                    }
            
            # Neural Network
            if 'Neural Network' in model_types:
                with st.spinner("Training Neural Network..."):
                    nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', 
                                          solver='adam', random_state=42, max_iter=1000)
                    nn_model.fit(X_train_scaled, y_train)
                    models['Neural Network'] = nn_model
                    nn_pred = nn_model.predict(X_test_scaled)
                    performance['Neural Network'] = {
                        'mse': mean_squared_error(y_test, nn_pred),
                        'mae': mean_absolute_error(y_test, nn_pred),
                        'r2': nn_model.score(X_test_scaled, y_test)
                    }
            
            # LSTM
            if 'LSTM' in model_types and len(X_train) > 0:
                try:
                    with st.spinner("Training LSTM..."):
                        # Prepare LSTM data
                        X_train_lstm = torch.FloatTensor(X_train)
                        y_train_lstm = torch.FloatTensor(y_train)
                        X_test_lstm = torch.FloatTensor(X_test)
                        y_test_lstm = torch.FloatTensor(y_test)
                        
                        # Create data loaders
                        train_dataset = TimeSeriesDataset(X_train, y_train)
                        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                        
                        # Initialize model
                        input_dim = X_train.shape[2]
                        lstm_model = LSTMModel(input_dim=input_dim, hidden_dim=50, output_dim=1, num_layers=2)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
                        
                        # Training loop
                        lstm_model.train()
                        for epoch in range(50):
                            epoch_loss = 0
                            for batch_X, batch_y in train_loader:
                                optimizer.zero_grad()
                                outputs = lstm_model(batch_X)
                                loss = criterion(outputs.squeeze(), batch_y)
                                loss.backward()
                                optimizer.step()
                                epoch_loss += loss.item()
                        
                        models['LSTM'] = lstm_model
                        
                        # Evaluate
                        lstm_model.eval()
                        with torch.no_grad():
                            lstm_pred = lstm_model(X_test_lstm).squeeze().numpy()
                        
                        performance['LSTM'] = {
                            'mse': mean_squared_error(y_test, lstm_pred),
                            'mae': mean_absolute_error(y_test, lstm_pred)
                        }
                        
                except Exception as e:
                    st.warning(f"LSTM training skipped: {str(e)}")
            
            # GRU
            if 'GRU' in model_types and len(X_train) > 0:
                try:
                    with st.spinner("Training GRU..."):
                        X_train_gru = torch.FloatTensor(X_train)
                        y_train_gru = torch.FloatTensor(y_train)
                        X_test_gru = torch.FloatTensor(X_test)
                        y_test_gru = torch.FloatTensor(y_test)
                        
                        train_dataset = TimeSeriesDataset(X_train, y_train)
                        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                        
                        input_dim = X_train.shape[2]
                        gru_model = GRUModel(input_dim=input_dim, hidden_dim=50, output_dim=1, num_layers=2)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
                        
                        gru_model.train()
                        for epoch in range(50):
                            epoch_loss = 0
                            for batch_X, batch_y in train_loader:
                                optimizer.zero_grad()
                                outputs = gru_model(batch_X)
                                loss = criterion(outputs.squeeze(), batch_y)
                                loss.backward()
                                optimizer.step()
                                epoch_loss += loss.item()
                        
                        models['GRU'] = gru_model
                        
                        gru_model.eval()
                        with torch.no_grad():
                            gru_pred = gru_model(X_test_gru).squeeze().numpy()
                        
                        performance['GRU'] = {
                            'mse': mean_squared_error(y_test, gru_pred),
                            'mae': mean_absolute_error(y_test, gru_pred)
                        }
                        
                except Exception as e:
                    st.warning(f"GRU training skipped: {str(e)}")
            
            self.ml_models = models
            self.ml_performance = performance
            
            st.success(f"Successfully trained {len(models)} ML models")
            
        except Exception as e:
            st.error(f"Error training ML models: {str(e)}")
    
    def predict_future(self, horizon: int = 30, model_type: str = 'Random Forest') -> Dict[str, Any]:
        """Generate future predictions using trained ML models with enhanced forecasting"""
        try:
            if model_type not in self.ml_models:
                st.error(f"Model {model_type} not trained")
                return {}
            
            # Get the latest sequence for prediction
            technicals = self.technical_indicators
            price_series = self.portfolio_value
            
            if len(technicals) < 60:
                st.warning("Insufficient data for prediction")
                return {}
            
            # Use last sequence_length days for initial prediction
            sequence_length = 60
            latest_sequence = []
            
            feature_columns = [col for col in technicals.columns if col not in ['returns', 'log_returns']]
            
            for i in range(len(technicals) - sequence_length, len(technicals)):
                feature_row = []
                for col in feature_columns:
                    feature_row.append(technicals[col].iloc[i])
                latest_sequence.append(feature_row)
            
            latest_sequence = np.array([latest_sequence])
            
            predictions = []
            current_sequence = latest_sequence.copy()
            
            model = self.ml_models[model_type]
            scaler = self.scalers.get('standard')
            
            st.info(f"Generating {horizon}-day forecast using {model_type}...")
            
            for _ in range(horizon):
                if model_type in ['LSTM', 'GRU']:
                    # PyTorch models
                    with torch.no_grad():
                        if model_type == 'LSTM':
                            pred = model(torch.FloatTensor(current_sequence)).squeeze().numpy()
                        else:
                            pred = model(torch.FloatTensor(current_sequence)).squeeze().numpy()
                else:
                    # Scikit-learn models
                    current_flat = current_sequence.reshape(1, -1)
                    if scaler:
                        current_flat = scaler.transform(current_flat)
                    pred = model.predict(current_flat)[0]
                
                predictions.append(pred)
                
                # Update sequence for next prediction
                new_row = current_sequence[0, -1:].copy()
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1] = new_row
            
            # Convert predictions to returns and calculate equity curve
            predicted_returns = np.array(predictions)
            predicted_equity = (1 + predicted_returns).cumprod() * self.portfolio_value.iloc[-1]
            
            # Calculate confidence intervals using bootstrap
            n_bootstrap = 100
            bootstrap_curves = []
            
            for _ in range(n_bootstrap):
                # Add random noise to predictions
                noise = np.random.normal(0, np.std(predicted_returns) * 0.1, len(predicted_returns))
                noisy_returns = predicted_returns + noise
                bootstrap_curve = (1 + noisy_returns).cumprod() * self.portfolio_value.iloc[-1]
                bootstrap_curves.append(bootstrap_curve)
            
            bootstrap_curves = np.array(bootstrap_curves)
            ci_lower = np.percentile(bootstrap_curves, 5, axis=0)
            ci_upper = np.percentile(bootstrap_curves, 95, axis=0)
            
            # Create future dates
            last_date = self.portfolio_value.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
            
            results = {
                'predicted_returns': pd.Series(predicted_returns, index=future_dates),
                'equity_forecast': pd.Series(predicted_equity, index=future_dates),
                'ci_bands': (pd.Series(ci_lower, index=future_dates), 
                           pd.Series(ci_upper, index=future_dates)),
                'model_performance': self.ml_performance.get(model_type, {}),
                'total_forecast_return': (predicted_equity[-1] / self.portfolio_value.iloc[-1] - 1) * 100
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return {}
    
    def run_backtest(self, use_ml_signals: bool = False, ml_model: str = 'Random Forest') -> Dict[str, Any]:
        """Run backtest with optional ML signal integration"""
        try:
            if self.returns is None:
                return {}
            
            portfolio_returns = self.returns['portfolio']
            
            # Generate basic signals (simple momentum)
            signals = pd.Series(1, index=portfolio_returns.index)  # Default: buy and hold
            rolling_mean = portfolio_returns.rolling(window=20).mean()
            rolling_std = portfolio_returns.rolling(window=20).std()
            
            # Enhanced signal generation
            signals = pd.Series(0, index=portfolio_returns.index)
            
            # Momentum signal
            momentum_signal = (portfolio_returns.rolling(10).mean() > 0).astype(int)
            
            # Mean reversion signal
            z_score = (portfolio_returns - rolling_mean) / rolling_std
            mean_reversion_signal = (z_score < -1).astype(int) - (z_score > 1).astype(int)
            
            # Volatility signal
            volatility_signal = (rolling_std < rolling_std.rolling(50).mean()).astype(int)
            
            # Combine signals
            signals = momentum_signal + mean_reversion_signal + volatility_signal
            signals = np.where(signals > 1, 1, np.where(signals < -1, -1, signals))
            
            if use_ml_signals and ml_model in self.ml_models:
                # Integrate ML predictions for enhanced signals
                try:
                    # Generate short-term ML predictions
                    ml_horizon = 5
                    ml_results = self.predict_future(ml_horizon, ml_model)
                    
                    if ml_results:
                        ml_pred_returns = ml_results['predicted_returns']
                        ml_signal = 1 if ml_pred_returns.mean() > 0 else -1
                        
                        # Apply ML signal to recent periods
                        recent_dates = portfolio_returns.index[-10:]
                        signals.loc[recent_dates] = ml_signal
                        
                        st.info("ML signals integrated into backtest")
                        
                except Exception as e:
                    st.warning(f"ML signal integration failed: {str(e)}")
            
            # Calculate strategy returns
            strategy_returns = portfolio_returns * signals.shift(1)
            strategy_returns = strategy_returns.dropna()
            
            # Calculate strategy metrics
            strategy_value = (1 + strategy_returns).cumprod() * self.initial_capital
            
            # Strategy metrics
            total_return = (strategy_value.iloc[-1] / self.initial_capital - 1) * 100
            sharpe_ratio = (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
            
            backtest_results = {
                'strategy_returns': strategy_returns,
                'strategy_value': strategy_value,
                'signals': signals,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': self.calculate_max_drawdown(strategy_returns),
                'win_rate': (strategy_returns > 0).mean() * 100
            }
            
            return backtest_results
            
        except Exception as e:
            st.error(f"Error in backtest: {str(e)}")
            return {}
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def run_monte_carlo(self, num_simulations: int = 1000, years: int = 1, 
                       use_ml_params: bool = False, ml_model: str = 'Random Forest') -> Dict[str, Any]:
        """Run Monte Carlo simulation with optional ML-enhanced parameters"""
        try:
            if self.returns is None:
                return {}
            
            portfolio_returns = self.returns['portfolio']
            
            if use_ml_params and ml_model in self.ml_models:
                # Use ML predictions for parameters
                try:
                    ml_results = self.predict_future(30, ml_model)
                    if ml_results:
                        # Use predicted returns for simulation parameters
                        pred_returns = ml_results['predicted_returns']
                        mu = pred_returns.mean() * 252
                        sigma = pred_returns.std() * np.sqrt(252)
                    else:
                        mu = portfolio_returns.mean() * 252
                        sigma = portfolio_returns.std() * np.sqrt(252)
                except:
                    mu = portfolio_returns.mean() * 252
                    sigma = portfolio_returns.std() * np.sqrt(252)
            else:
                # Use historical parameters
                mu = portfolio_returns.mean() * 252
                sigma = portfolio_returns.std() * np.sqrt(252)
            
            # Run simulations
            days = years * 252
            simulations = np.zeros((days, num_simulations))
            initial_value = self.portfolio_value.iloc[-1]
            
            for i in range(num_simulations):
                # Geometric Brownian Motion with drift and volatility
                daily_returns = np.random.normal(mu/days, sigma/np.sqrt(days), days)
                price_path = initial_value * np.cumprod(1 + daily_returns)
                simulations[:, i] = price_path
            
            # Calculate statistics
            final_values = simulations[-1, :]
            mean_final_value = np.mean(final_values)
            median_final_value = np.median(final_values)
            var_95 = np.percentile(final_values, 5)
            cvar_95 = final_values[final_values <= var_95].mean()
            
            # Probability of loss
            prob_loss = (final_values < initial_value).mean() * 100
            
            mc_results = {
                'simulations': simulations,
                'final_values': final_values,
                'mean_final_value': mean_final_value,
                'median_final_value': median_final_value,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'prob_loss': prob_loss,
                'mu': mu,
                'sigma': sigma
            }
            
            return mc_results
            
        except Exception as e:
            st.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {}
    
    def generate_report(self, benchmark_returns: pd.Series, ml_results: Dict = None) -> str:
        """Generate comprehensive HTML report"""
        try:
            if self.returns is None:
                return ""
            
            portfolio_returns = self.returns['portfolio']
            
            # Generate QuantStats report
            qs.reports.html(portfolio_returns, benchmark=benchmark_returns,
                          output='temp_report.html', download_filename='portfolio_report.html')
            
            with open('temp_report.html', 'r') as f:
                html_content = f.read()
            
            # Clean up
            if os.path.exists('temp_report.html'):
                os.remove('temp_report.html')
            
            return html_content
            
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            return ""

# Helper functions
def plot_technical_indicators(technicals: pd.DataFrame) -> go.Figure:
    """Plot technical indicators"""
    fig = go.Figure()
    
    # Plot RSI
    fig.add_trace(go.Scatter(x=technicals.index, y=technicals.get('rsi_14', []), 
                           mode='lines', name='RSI', line=dict(color='blue')))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    fig.update_layout(title='Technical Indicators - RSI', xaxis_title='Date', yaxis_title='RSI')
    return fig

def plot_monte_carlo(simulations: np.ndarray, final_values: np.ndarray, initial_value: float) -> go.Figure:
    """Plot Monte Carlo simulation results"""
    fig = go.Figure()
    
    # Plot sample of simulation paths
    n_paths_to_plot = min(50, simulations.shape[1])
    for i in range(n_paths_to_plot):
        fig.add_trace(go.Scatter(x=list(range(simulations.shape[0])), 
                               y=simulations[:, i],
                               mode='lines', line=dict(width=1, color='lightblue'),
                               showlegend=False))
    
    # Plot mean path
    mean_path = simulations.mean(axis=1)
    fig.add_trace(go.Scatter(x=list(range(simulations.shape[0])), 
                           y=mean_path,
                           mode='lines', line=dict(width=3, color='red'),
                           name='Mean Path'))
    
    fig.update_layout(title='Monte Carlo Simulation Paths',
                    xaxis_title='Days', yaxis_title='Portfolio Value')
    
    return fig

def plot_predictions(historical_equity: pd.Series, forecast_equity: pd.Series, 
                    ci_bands: Tuple[pd.Series, pd.Series] = None) -> go.Figure:
    """Plot historical and forecasted equity curve"""
    fig = go.Figure()
    
    # Historical equity
    fig.add_trace(go.Scatter(x=historical_equity.index, y=historical_equity,
                           mode='lines', name='Historical', line=dict(color='blue')))
    
    # Forecasted equity
    fig.add_trace(go.Scatter(x=forecast_equity.index, y=forecast_equity,
                           mode='lines', name='Forecast', line=dict(color='red')))
    
    # Confidence intervals
    if ci_bands is not None:
        ci_lower, ci_upper = ci_bands
        fig.add_trace(go.Scatter(x=ci_upper.index, y=ci_upper,
                               mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=ci_lower.index, y=ci_lower,
                               mode='lines', line=dict(width=0), fill='tonexty',
                               fillcolor='rgba(255,0,0,0.2)', name='95% CI'))
    
    fig.update_layout(title='Portfolio Equity Curve with ML Forecast',
                    xaxis_title='Date', yaxis_title='Portfolio Value')
    
    return fig

# Main application
def main():
    """Main Streamlit application with enhanced market coverage"""
    
    # Sidebar inputs
    st.sidebar.header("🇮🇳 Portfolio Configuration")
    
    # Market Category Selection
    st.sidebar.subheader("Market Categories")
    
    # Initialize market data
    market_data = IndianMarketData()
    
    # Category selection
    selected_categories = st.sidebar.multiselect(
        "Select Market Categories",
        [
            'Nifty 50 Stocks', 
            'Bank Nifty Stocks', 
            'Nifty Midcap Stocks',
            'Nifty Smallcap Stocks',
            'Indian Indices',
            'Global Indices',
            'Forex',
            'Commodities',
            'Cryptocurrencies'
        ],
        default=['Nifty 50 Stocks']
    )
    
    # Asset selection based on categories
    selected_tickers = []
    
    if 'Nifty 50 Stocks' in selected_categories:
        nifty_selection = st.sidebar.multiselect(
            "Select Nifty 50 Stocks",
            market_data.NIFTY_50,
            default=market_data.NIFTY_50[:5]  # Default first 5 stocks
        )
        selected_tickers.extend(nifty_selection)
    
    if 'Bank Nifty Stocks' in selected_categories:
        banknifty_selection = st.sidebar.multiselect(
            "Select Bank Nifty Stocks",
            market_data.BANKNIFTY,
            default=market_data.BANKNIFTY[:3]
        )
        selected_tickers.extend(banknifty_selection)
    
    if 'Nifty Midcap Stocks' in selected_categories:
        midcap_selection = st.sidebar.multiselect(
            "Select Nifty Midcap Stocks",
            market_data.NIFTY_MIDCAP,
            default=market_data.NIFTY_MIDCAP[:3]
        )
        selected_tickers.extend(midcap_selection)
    
    if 'Nifty Smallcap Stocks' in selected_categories:
        smallcap_selection = st.sidebar.multiselect(
            "Select Nifty Smallcap Stocks",
            market_data.NIFTY_SMALLCAP,
            default=market_data.NIFTY_SMALLCAP[:2]
        )
        selected_tickers.extend(smallcap_selection)
    
    if 'Indian Indices' in selected_categories:
        indices_selection = st.sidebar.multiselect(
            "Select Indian Indices",
            list(market_data.INDIAN_INDICES.values()),
            default=['^NSEI']
        )
        selected_tickers.extend(indices_selection)
    
    if 'Global Indices' in selected_categories:
        global_selection = st.sidebar.multiselect(
            "Select Global Indices",
            list(market_data.GLOBAL_INDICES.values()),
            default=['^GSPC']
        )
        selected_tickers.extend(global_selection)
    
    if 'Forex' in selected_categories:
        forex_selection = st.sidebar.multiselect(
            "Select Forex Pairs",
            list(market_data.FOREX.values()),
            default=['INR=X']
        )
        selected_tickers.extend(forex_selection)
    
    if 'Commodities' in selected_categories:
        commodities_selection = st.sidebar.multiselect(
            "Select Commodities",
            list(market_data.COMMODITIES.values()),
            default=['GC=F']
        )
        selected_tickers.extend(commodities_selection)
    
    if 'Cryptocurrencies' in selected_categories:
        crypto_selection = st.sidebar.multiselect(
            "Select Cryptocurrencies",
            list(market_data.CRYPTO.values()),
            default=['BTC-USD']
        )
        selected_tickers.extend(crypto_selection)
    
    # Custom tickers
    custom_tickers = st.sidebar.text_input("Additional Custom Tickers (comma-separated)", "")
    if custom_tickers:
        custom_list = [ticker.strip() for ticker in custom_tickers.split(",")]
        selected_tickers.extend(custom_list)
    
    # Remove duplicates
    selected_tickers = list(dict.fromkeys(selected_tickers))
    
    if not selected_tickers:
        st.sidebar.warning("Please select at least one asset category")
        return
    
    # Portfolio weights
    st.sidebar.subheader("Portfolio Weights")
    if selected_tickers:
        st.sidebar.info(f"Selected {len(selected_tickers)} assets")
        
        # Equal weights by default
        equal_weight = 1.0 / len(selected_tickers)
        weights = [equal_weight] * len(selected_tickers)
        
        # Allow custom weights
        use_custom_weights = st.sidebar.checkbox("Use Custom Weights")
        if use_custom_weights:
            weight_input = st.sidebar.text_input(
                f"Custom Weights (comma-separated, sum to 1.0)",
                ",".join([f"{equal_weight:.3f}"] * len(selected_tickers))
            )
            try:
                weights = [float(w.strip()) for w in weight_input.split(",")]
                if len(weights) != len(selected_tickers):
                    st.sidebar.error("Number of weights must match number of assets")
                    return
                if abs(sum(weights) - 1.0) > 0.01:
                    st.sidebar.warning("Weights don't sum to 1.0, normalizing...")
                    weights = [w/sum(weights) for w in weights]
            except:
                st.sidebar.error("Invalid weights format")
                return
    
    # Date range with 10-15 years default
    st.sidebar.subheader("Date Range")
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    
    # Ensure 10+ years of data
    if (end_date - start_date).days < 3650:
        st.sidebar.warning("For better analysis, consider using at least 10 years of data")
    
    # Benchmark selection
    st.sidebar.subheader("Benchmark")
    benchmark = st.sidebar.selectbox(
        "Select Benchmark",
        list(market_data.INDIAN_INDICES.values()) + list(market_data.GLOBAL_INDICES.values()),
        index=0  # Default to Nifty 50
    )
    
    initial_capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, min_value=1000)
    
    # ML Configuration
    st.sidebar.header("🤖 Machine Learning Configuration")
    
    ml_models = st.sidebar.multiselect(
        "Select ML Models",
        ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'SVR', 'Neural Network', 'LSTM', 'GRU'],
        default=['Random Forest', 'Linear Regression', 'Gradient Boosting']
    )
    
    sequence_length = st.sidebar.slider("Sequence Length", 30, 100, 60)
    prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 30, 90, 30)
    enable_ml = st.sidebar.checkbox("Enable ML Predictions", True)
    
    # Backtest and Simulation
    st.sidebar.header("📊 Backtesting & Simulation")
    use_ml_signals = st.sidebar.checkbox("Use ML Signals in Backtest", False)
    num_simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 5000, 1000)
    projection_years = st.sidebar.slider("Projection Years", 1, 10, 5)
    
    # Risk-free rate for Indian context
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 2.0, 8.0, 4.0) / 100
    
    # Initialize analyzer
    analyzer = EnhancedPortfolioAnalyzer(
        tickers=selected_tickers,
        weights=weights,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        benchmark=benchmark,
        initial_capital=initial_capital
    )
    
    # Main analysis button
    if st.sidebar.button("🚀 Generate Comprehensive Analysis", type="primary"):
        with st.spinner("Fetching extensive market data and performing deep analysis..."):
            
            # Display selected portfolio
            st.header("📈 Selected Portfolio Composition")
            portfolio_df = pd.DataFrame({
                'Ticker': selected_tickers,
                'Weight': [f"{w*100:.2f}%" for w in weights]
            })
            st.dataframe(portfolio_df)
            
            # Fetch data
            data = analyzer.fetch_extended_data()
            if data.empty:
                st.error("Failed to fetch data. Please check ticker symbols and date range.")
                return
            
            # Compute returns
            returns = analyzer.compute_returns()
            if not returns:
                st.error("Failed to compute returns.")
                return
            
            # Compute metrics
            metrics = analyzer.compute_metrics(risk_free_rate)
            
            # Display comprehensive metrics
            st.header("📊 Portfolio Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{metrics.get('Total Return (%)', 0):.2f}%")
                st.metric("Annual Return", f"{metrics.get('Annual Return (%)', 0):.2f}%")
                st.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
                st.metric("Alpha", f"{metrics.get('Alpha (%)', 0):.2f}%")
            
            with col2:
                st.metric("Max Drawdown", f"{metrics.get('Max Drawdown (%)', 0):.2f}%")
                st.metric("Volatility", f"{metrics.get('Annual Volatility (%)', 0):.2f}%")
                st.metric("Sortino Ratio", f"{metrics.get('Sortino Ratio', 0):.2f}")
                st.metric("Beta", f"{metrics.get('Beta', 0):.2f}")
            
            with col3:
                st.metric("Calmar Ratio", f"{metrics.get('Calmar Ratio', 0):.2f}")
                st.metric("Information Ratio", f"{metrics.get('Information Ratio', 0):.2f}")
                st.metric("Win Rate", f"{metrics.get('Win Rate (%)', 0):.2f}%")
                st.metric("Profit/Loss Ratio", f"{metrics.get('Profit/Loss Ratio', 0):.2f}")
            
            with col4:
                st.metric("VaR (95%)", f"{metrics.get('VaR (95%) (%)', 0):.2f}%")
                st.metric("CVaR (95%)", f"{metrics.get('CVaR (95%) (%)', 0):.2f}%")
                st.metric("Skewness", f"{metrics.get('Skewness', 0):.3f}")
                st.metric("Kurtosis", f"{metrics.get('Kurtosis', 0):.3f}")
            
            # Portfolio visualization
            st.header("📈 Portfolio Visualization")
            
            # Cumulative returns comparison
            fig_returns = go.Figure()
            portfolio_cumulative = (1 + returns['portfolio']).cumprod()
            benchmark_cumulative = (1 + returns['benchmark']).cumprod()
            
            fig_returns.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative,
                                           mode='lines', name='Portfolio', line=dict(color='blue')))
            fig_returns.add_trace(go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative,
                                           mode='lines', name='Benchmark', line=dict(color='red')))
            fig_returns.update_layout(
                title='Portfolio vs Benchmark Cumulative Returns',
                xaxis_title='Date', 
                yaxis_title='Cumulative Return',
                height=500
            )
            st.plotly_chart(fig_returns, use_container_width=True)
            
            # Portfolio composition
            st.subheader("Portfolio Composition & Correlations")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    values=weights, 
                    names=selected_tickers, 
                    title="Portfolio Weights",
                    height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Correlation heatmap
                if len(selected_tickers) > 1:
                    returns_data = data.pct_change().dropna()
                    correlation_matrix = returns_data.corr()
                    fig_heatmap = px.imshow(
                        correlation_matrix, 
                        title="Asset Returns Correlation Heatmap",
                        aspect="auto",
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Technical indicators
            st.subheader("Technical Analysis")
            technicals = analyzer.compute_technicals(analyzer.portfolio_value)
            if not technicals.empty:
                fig_tech = plot_technical_indicators(technicals)
                st.plotly_chart(fig_tech, use_container_width=True)
            
            # Machine Learning Predictions
            if enable_ml and ml_models:
                st.header("🤖 Machine Learning Predictions")
                
                with st.spinner("Training multiple ML models with extensive market data..."):
                    analyzer.train_ml_models(ml_models, sequence_length)
                
                # Display model performance comparison
                if hasattr(analyzer, 'ml_performance'):
                    st.subheader("Model Performance Comparison")
                    
                    performance_data = []
                    for model_name, metrics in analyzer.ml_performance.items():
                        performance_data.append({
                            'Model': model_name,
                            'MSE': metrics.get('mse', 0),
                            'MAE': metrics.get('mae', 0),
                            'R²': metrics.get('r2', 0)
                        })
                    
                    if performance_data:
                        perf_df = pd.DataFrame(performance_data)
                        
                        # Display performance metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.dataframe(perf_df.style.format({
                                'MSE': '{:.6f}', 
                                'MAE': '{:.6f}',
                                'R²': '{:.4f}'
                            }))
                        
                        with col2:
                            # Plot performance comparison
                            fig_perf = px.bar(
                                perf_df, 
                                x='Model', 
                                y=['MSE', 'MAE'], 
                                title='Model Performance Comparison', 
                                barmode='group',
                                height=400
                            )
                            st.plotly_chart(fig_perf, use_container_width=True)
                
                # Generate predictions for each model
                st.subheader("Multi-Model Forecasting")
                
                for model_name in ml_models:
                    if model_name in analyzer.ml_models:
                        with st.spinner(f"Generating predictions with {model_name}..."):
                            ml_results = analyzer.predict_future(prediction_horizon, model_name)
                            
                            if ml_results:
                                # Display prediction metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        f"{model_name} Forecast Return", 
                                        f"{ml_results['total_forecast_return']:.2f}%"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Model MSE", 
                                        f"{ml_results['model_performance'].get('mse', 0):.6f}"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Model R²", 
                                        f"{ml_results['model_performance'].get('r2', 0):.4f}"
                                    )
                                
                                # Plot predictions
                                fig_pred = plot_predictions(
                                    analyzer.portfolio_value,
                                    ml_results['equity_forecast'],
                                    ml_results['ci_bands']
                                )
                                st.plotly_chart(fig_pred, use_container_width=True)
            
            # Backtesting
            st.header("🔍 Backtesting Results")
            backtest_results = analyzer.run_backtest(
                use_ml_signals, 
                ml_models[0] if ml_models and enable_ml else 'Random Forest'
            )
            
            if backtest_results:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Strategy Total Return", f"{backtest_results['total_return']:.2f}%")
                    st.metric("Strategy Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
                
                with col2:
                    st.metric("Strategy Max Drawdown", f"{backtest_results['max_drawdown']:.2f}%")
                    st.metric("Strategy Win Rate", f"{backtest_results['win_rate']:.2f}%")
                
                with col3:
                    buy_hold_return = metrics.get('Total Return (%)', 0)
                    strategy_advantage = backtest_results['total_return'] - buy_hold_return
                    st.metric("vs Buy & Hold", f"{strategy_advantage:.2f}%")
                
                # Plot backtest results
                fig_backtest = go.Figure()
                fig_backtest.add_trace(go.Scatter(
                    x=analyzer.portfolio_value.index, 
                    y=analyzer.portfolio_value,
                    mode='lines', 
                    name='Buy & Hold', 
                    line=dict(color='blue')
                ))
                fig_backtest.add_trace(go.Scatter(
                    x=backtest_results['strategy_value'].index,
                    y=backtest_results['strategy_value'],
                    mode='lines', 
                    name='Strategy', 
                    line=dict(color='red')
                ))
                fig_backtest.update_layout(
                    title='Backtest Results: Strategy vs Buy & Hold',
                    xaxis_title='Date', 
                    yaxis_title='Portfolio Value',
                    height=500
                )
                st.plotly_chart(fig_backtest, use_container_width=True)
            
            # Monte Carlo Simulation
            st.header("🎲 Monte Carlo Simulation")
            mc_results = analyzer.run_monte_carlo(
                num_simulations, 
                projection_years, 
                use_ml_params=enable_ml,
                ml_model=ml_models[0] if ml_models and enable_ml else 'Random Forest'
            )
            
            if mc_results:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Final Value", f"₹{mc_results['mean_final_value']:,.0f}")
                    st.metric("Probability of Loss", f"{mc_results['prob_loss']:.1f}%")
                
                with col2:
                    st.metric("Median Final Value", f"₹{mc_results['median_final_value']:,.0f}")
                    st.metric("Expected Return (μ)", f"{mc_results['mu']*100:.2f}%")
                
                with col3:
                    st.metric("VaR (95%)", f"₹{mc_results['var_95']:,.0f}")
                    st.metric("Volatility (σ)", f"{mc_results['sigma']*100:.2f}%")
                
                with col4:
                    st.metric("CVaR (95%)", f"₹{mc_results['cvar_95']:,.0f}")
                    current_value = analyzer.portfolio_value.iloc[-1]
                    st.metric("Current Value", f"₹{current_value:,.0f}")
                
                # Plot Monte Carlo results
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_mc = plot_monte_carlo(
                        mc_results['simulations'], 
                        mc_results['final_values'],
                        analyzer.portfolio_value.iloc[-1]
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)
                
                with col2:
                    # Final value distribution
                    fig_dist = px.histogram(
                        x=mc_results['final_values'], 
                        title='Monte Carlo Final Value Distribution',
                        nbins=50
                    )
                    fig_dist.add_vline(
                        x=mc_results['mean_final_value'], 
                        line_dash="dash", 
                        line_color="red", 
                        annotation_text="Mean"
                    )
                    fig_dist.add_vline(
                        x=mc_results['median_final_value'], 
                        line_dash="dash", 
                        line_color="green", 
                        annotation_text="Median"
                    )
                    fig_dist.add_vline(
                        x=analyzer.portfolio_value.iloc[-1], 
                        line_dash="dash", 
                        line_color="blue", 
                        annotation_text="Current"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            # Generate comprehensive report
            st.header("📄 Download Comprehensive Report")
            if st.button("Generate Detailed Analysis Report"):
                with st.spinner("Generating comprehensive report..."):
                    report_html = analyzer.generate_report(returns['benchmark'])
                    
                    if report_html:
                        st.download_button(
                            label="📥 Download HTML Report",
                            data=report_html,
                            file_name="comprehensive_portfolio_analysis_report.html",
                            mime="text/html"
                        )
                        st.success("Report generated successfully!")
                    else:
                        st.error("Failed to generate report")

# Add market overview section
def show_market_overview():
    """Display comprehensive market overview"""
    st.sidebar.header("🌐 Market Overview")
    
    market_data = IndianMarketData()
    
    # Quick market indices
    st.sidebar.subheader("Quick Market Indices")
    
    # Fetch current levels for major indices
    major_indices = {
        'Nifty 50': '^NSEI',
        'Bank Nifty': '^NSEBANK', 
        'S&P 500': '^GSPC',
        'Gold': 'GC=F',
        'USD/INR': 'INR=X'
    }
    
    for name, ticker in major_indices.items():
        try:
            data = yf.download(ticker, period='1d', progress=False)
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[0] if len(data) > 1 else current_price
                change_pct = ((current_price - prev_close) / prev_close) * 100
                
                st.sidebar.metric(
                    name,
                    f"{current_price:.2f}",
                    f"{change_pct:+.2f}%"
                )
        except:
            st.sidebar.text(f"{name}: N/A")

if __name__ == "__main__":
    show_market_overview()
    main()