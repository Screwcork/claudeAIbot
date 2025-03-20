#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced AI-Powered Trading Bot for Bybit API v5
------------------------------------------------
This trading system combines multiple AI models, adaptive position sizing,
market regime detection, and sentiment analysis to create a comprehensive
trading solution optimized for cryptocurrency markets.
10b2cb833cf8460fab929398b74939f1 newsapi
"""

import os
import sys
import time
import json
import hmac
import hashlib
import logging
import numpy as np
import pandas as pd
import requests
import ccxt
import argparse
import websocket
import threading
import math
import nx
import schedule
import pickle
import uuid
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque



os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings


# ML/AI libraries
import gc
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier


# NLP for sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Technical analysis
import ta
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

# Deep learning models
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input, Concatenate, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Limit TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    # Limit TensorFlow CPU memory
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)


# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TradingBot")


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CONSOLIDATING = "consolidating"
    UNKNOWN = "unknown"


class TradingSignal(Enum):
    """Trading signal types."""
    STRONG_BUY = 3
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -3


@dataclass
class TradePosition:
    """Represents an open trading position."""
    symbol: str
    side: str  # "buy" or "sell"
    entry_price: float
    amount: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    id: str = None
    order_id: str = None
    status: str = "open"  # "open", "closed", "canceled"
    exit_price: float = None
    exit_time: datetime = None
    pnl: float = None
    pnl_percentage: float = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.tags is None:
            self.tags = []


class Config:
    """Configuration class for the trading bot."""
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize with default or loaded configuration."""
        # Default configuration
        self.default_config = {
            "api_key": "",
            "api_secret": "",
            "base_url": "https://api.bybit.com",
            "testnet": False,
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframes": {
                "primary": "15m",
                "secondary": ["1h", "4h", "1d"],
            },
            "trade_settings": {
                "max_positions": 3,
                "account_risk_per_trade": 0.02,  # 2% of account per trade
                "default_stop_loss_pct": 0.02,   # 2% stop loss
                "default_take_profit_pct": 0.06, # 6% take profit
                "trailing_stop": True,           # Enable trailing stop
                "trailing_stop_activation": 0.02,# Activate trailing stop after 2% profit
                "trailing_stop_distance": 0.01,  # 1% trailing stop distance
            },
            "ml_settings": {
                "feature_window": 100,           # Look back window for feature generation
                "prediction_window": 24,         # Forward prediction window (in candles)
                "train_interval_days": 7,        # Re-train models every 7 days
                "confidence_threshold": 0.65,    # Minimum model confidence to trade
                "ensemble_method": "weighted",   # How to combine model predictions
                "model_weights": {
                    "xgboost": 0.25,
                    "lstm": 0.35,
                    "catboost": 0.25,
                    "transformer": 0.15,
                },
                # Added model_params to default config to avoid unknown key warning
                "model_params": {
                    "xgboost": {
                        "n_estimators": 50,
                        "learning_rate": 0.05,
                        "max_depth": 4,
                        "min_child_weight": 1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "gamma": 0
                    },
                    "catboost": {
                        "iterations": 50,
                        "learning_rate": 0.05,
                        "depth": 4,
                        "verbose": 0
                    },
                    "lstm": {
                        "epochs": 30,
                        "batch_size": 32
                    }
                }
            },
            "risk_management": {
                "max_daily_loss": 0.05,          # Stop trading after 5% daily loss
                "max_drawdown": 0.15,            # Maximum allowed drawdown
                "position_correlation_limit": 0.7 # Limit on correlated positions
            },
            "sentiment_analysis": {
                "enabled": True,
                "sources": ["twitter", "reddit", "news"],
                "refresh_interval_minutes": 30,
                "impact_weight": 0.2             # Weight for sentiment in final decision
            },
            "execution": {
                "order_type": "limit",           # "market" or "limit"
                "limit_order_distance": 0.001,   # 0.1% from current price
                "retry_attempts": 3,             # Number of retry attempts for failed orders
                "retry_delay": 1,                # Delay between retry attempts (seconds)
                "execution_cooldown": 60,        # Seconds to wait after execution before next trade
            },
            "performance_monitoring": {
                "metrics_interval_hours": 6,     # Calculate performance metrics every 6 hours
                "benchmark": "HODL",             # Benchmark for performance comparison
            },
            "logging": {
                "level": "INFO",                 # Logging level
                "save_predictions": True,        # Save model predictions
                "detailed_trade_logs": True      # Detailed logging for each trade
            }
        }
        
        self.config = self.default_config.copy()
        
        # Try to load configuration from file
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Recursively update the default config with loaded values
                    self._update_config(self.config, loaded_config)
                    logger.info(f"Configuration loaded from {config_file}")
            else:
                # Create a default config file if it doesn't exist
                with open(config_file, 'w') as f:
                    json.dump(self.default_config, f, indent=4)
                    logger.info(f"Default configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
        
        # Set logging level from config
        logging_level = getattr(logging, self.config['logging']['level'], logging.INFO)
        logger.setLevel(logging_level)
        
        # Validate configuration
        self._validate_config()
    
    def _update_config(self, default_dict, new_dict):
        """Recursively update the default configuration with new values."""
        for key, value in new_dict.items():
            if key in default_dict:
                if isinstance(value, dict) and isinstance(default_dict[key], dict):
                    self._update_config(default_dict[key], value)
                else:
                    default_dict[key] = value
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def _validate_config(self):
        """Validate the configuration to ensure all required fields are present."""
        if not self.config["api_key"] or not self.config["api_secret"]:
            logger.warning("API credentials not set in configuration")
        
        if not self.config["symbols"]:
            logger.error("No trading symbols specified in configuration")
            raise ValueError("At least one trading symbol must be specified")
        
        logger.info("Configuration validated successfully")
    
    def get(self, *keys):
        """Get a configuration value using dot notation."""
        config = self.config
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return None
        return config
    
    def update(self, config_update):
        """Update the configuration with new values."""
        self._update_config(self.config, config_update)
        # Re-validate the configuration
        self._validate_config()
        logger.info("Configuration updated")
    
    def save(self, config_file="config.json"):
        """Save the current configuration to a file."""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
                logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


class BybitAPI:
    """Production-ready wrapper for Bybit API v5 with robust error handling and rate limiting."""
    
    def __init__(self, config):
        """Initialize with config."""
        self.config = config
        self.api_key = config.get("api_key")
        self.api_secret = config.get("api_secret")
        self.base_url = config.get("base_url")
        self.testnet = config.get("testnet")
        
        # Rate limiting
        self.rate_limits = {
            "general": RateLimiter(max_calls=100, period=10),    # 100 calls per 10 sec for general endpoints
            "order": RateLimiter(max_calls=50, period=10),       # 50 calls per 10 sec for order endpoints
            "position": RateLimiter(max_calls=50, period=10),    # 50 calls per 10 sec for position endpoints
        }
        
        # Server time offset
        self.time_offset = 0
        
        # Initialize ccxt client for simpler interactions
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # Use USDT-margined futures
                'adjustForTimeDifference': True,
            }
        })
        
        if self.testnet:
            self.exchange.urls['api'] = 'https://api-testnet.bybit.com'
            self.base_url = 'https://api-testnet.bybit.com'
        
        # DNS cache
        self.dns_cache = {}
        self.dns_cache_time = {}
        
        # Health check
        self.health_status = True
        self.last_successful_call = time.time()
        
        # Test API connection and sync time
        self._test_connection()
    
    def _test_connection(self):
        """Test API connection, update rate limits, and sync server time."""
        try:
            # Get server time for time synchronization
            start_time = time.time()
            time_result = self.exchange.fetch_time()
            end_time = time.time()
            
            # Calculate network latency (round trip time / 2)
            latency = (end_time - start_time) / 2
            
            # Calculate time offset (server time - local time)
            self.time_offset = time_result - int((start_time + end_time) / 2 * 1000)
            
            logger.info(f"Bybit API connection successful. Server time: {time_result}, " +
                      f"latency: {latency*1000:.0f}ms, time offset: {self.time_offset}ms")
            
            # Get account information to verify authentication
            try:
                account_info = self.exchange.fetch_balance()
                logger.info("Authentication successful")
                self.health_status = True
                self.last_successful_call = time.time()
            except Exception as e:
                if "authenticate" in str(e).lower():
                    logger.error("Authentication failed. Please check API key and secret")
                    raise  # Re-raise authentication errors
                else:
                    logger.warning(f"Account info error: {e}")
            
            # Get rate limits from exchange info
            try:
                exchange_info = self._make_request("GET", "/v5/market/time", auth=False)
                logger.debug("Exchange info retrieved successfully")
            except Exception as e:
                logger.warning(f"Could not retrieve exchange info: {e}")
                
        except Exception as e:
            logger.error(f"Error connecting to Bybit API: {e}")
            self.health_status = False
            
            # Check if it's a temporary DNS or connection issue, which we can recover from
            if "dns" in str(e).lower() or "connect" in str(e).lower():
                logger.warning("Connection issue detected, will retry on next call")
            elif "authenticate" in str(e).lower():
                logger.error("Authentication failed. Please check API key and secret")
                raise  # Re-raise authentication errors
    
    def _resolve_dns(self, url):
        """Resolve DNS with caching to avoid repeated lookups."""
        hostname = url.split('://')[1].split('/')[0]
        
        # Check cache first
        current_time = time.time()
        if hostname in self.dns_cache and current_time - self.dns_cache_time.get(hostname, 0) < 3600:
            return self.dns_cache[hostname]
        
        try:
            import socket
            ip_address = socket.gethostbyname(hostname)
            self.dns_cache[hostname] = ip_address
            self.dns_cache_time[hostname] = current_time
            return ip_address
        except Exception as e:
            logger.warning(f"DNS resolution failed for {hostname}: {e}")
            return hostname  # Return hostname if resolution fails
    
    def _generate_signature(self, params: dict, timestamp: int) -> str:
        """Generate signature for authenticated requests with time offset."""
        # Adjust timestamp for server time difference
        adjusted_timestamp = timestamp + self.time_offset
        
        # Build query string from parameters
        param_str = str(adjusted_timestamp) + self.api_key + self._build_query_string(params)
        
        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(param_str, "utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _build_query_string(self, params: dict) -> str:
        """Build query string from parameters."""
        if not params:
            return ""
            
        # Sort parameters alphabetically by key
        keys = sorted(params.keys())
        
        # Build query string
        return "&".join([f"{key}={params[key]}" for key in keys])
    
    def _make_request(self, method: str, endpoint: str, params: dict = None, auth: bool = False, 
                     retry_count: int = 0, category: str = "general") -> dict:
        """Make a request to the Bybit API with comprehensive error handling and retries."""
        # Check health status for non-retry calls
        if retry_count == 0 and not self.health_status:
            # If more than 5 minutes since last successful call, try to recover
            if time.time() - self.last_successful_call > 300:
                logger.info("Attempting to recover API connection...")
                self._test_connection()
            
            # If still unhealthy, return error
            if not self.health_status:
                return {"success": False, "error": "API connection is unhealthy"}
        
        # Check rate limits
        if not self.rate_limits[category].can_call():
            wait_time = self.rate_limits[category].get_wait_time()
            logger.warning(f"Rate limit reached for {category}, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        # Mark rate limit usage
        self.rate_limits[category].call_made()
        
        # Initialize parameters if None
        params = params or {}
        
        # Build URL
        url = f"{self.base_url}{endpoint}"
        
        # Prepare headers
        headers = {}
        
        # Add authentication headers if required
        if auth:
            timestamp = int(time.time() * 1000)
            headers.update({
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": str(timestamp),
                "X-BAPI-SIGN": self._generate_signature(params, timestamp),
            })
        
        try:
            # Track request metrics
            start_time = time.time()
            
            # Make request based on HTTP method
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                headers["Content-Type"] = "application/json"
                response = requests.post(url, json=params, headers=headers, timeout=10)
            elif method == "DELETE":
                headers["Content-Type"] = "application/json"
                response = requests.delete(url, json=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Track response time
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response JSON
            result = response.json()
            
            # Check for API errors
            if 'retCode' in result and result['retCode'] != 0:
                logger.warning(f"API error: {result['retMsg']} (code: {result['retCode']})")
                
                # Handle specific error codes
                if result['retCode'] == 10002:  # Request expired
                    # Update time offset and retry
                    if retry_count < 3:
                        self._sync_time()
                        return self._make_request(method, endpoint, params, auth, retry_count + 1, category)
                
                elif result['retCode'] == 10018:  # IP rate limit
                    wait_time = 10  # Wait longer for IP rate limit
                    logger.warning(f"IP rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                    if retry_count < 3:
                        return self._make_request(method, endpoint, params, auth, retry_count + 1, category)
                
                # Return the error response
                return result
            
            # Update health status on successful call
            self.health_status = True
            self.last_successful_call = time.time()
            
            # Log slow responses
            if response_time > 1000:  # More than 1 second
                logger.warning(f"Slow API response: {response_time:.0f}ms for {method} {endpoint}")
            
            return result
            
        except requests.exceptions.Timeout as e:
            logger.warning(f"Request timeout: {endpoint} - {e}")
            
            # Retry with exponential backoff
            if retry_count < 3:
                wait_time = 2 ** retry_count  # 1, 2, 4 seconds
                logger.info(f"Retrying in {wait_time}s... (attempt {retry_count+1}/3)")
                time.sleep(wait_time)
                return self._make_request(method, endpoint, params, auth, retry_count + 1, category)
            
            return {"success": False, "error": f"Request timeout after 3 retries: {str(e)}"}
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error: {endpoint} - {e}")
            
            # Try to recover DNS or connection issues
            if retry_count < 3:
                wait_time = 2 ** retry_count  # 1, 2, 4 seconds
                logger.info(f"Retrying in {wait_time}s... (attempt {retry_count+1}/3)")
                time.sleep(wait_time)
                return self._make_request(method, endpoint, params, auth, retry_count + 1, category)
            
            # Mark API as unhealthy
            self.health_status = False
            
            return {"success": False, "error": f"Connection error after 3 retries: {str(e)}"}
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else 0
            
            logger.warning(f"HTTP error {status_code}: {endpoint} - {e}")
            
            # Retry server errors (5xx) but not client errors (4xx)
            if status_code >= 500 and retry_count < 3:
                wait_time = 2 ** retry_count  # 1, 2, 4 seconds
                logger.info(f"Retrying in {wait_time}s... (attempt {retry_count+1}/3)")
                time.sleep(wait_time)
                return self._make_request(method, endpoint, params, auth, retry_count + 1, category)
            
            # Try to parse error response
            error_detail = {}
            try:
                if hasattr(e, 'response') and e.response.text:
                    error_detail = json.loads(e.response.text)
            except:
                pass
                
            return {"success": False, "error": f"HTTP error {status_code}", "details": error_detail}
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {endpoint} - {e}")
            
            # Retry JSON decode errors
            if retry_count < 2:
                wait_time = 2 ** retry_count  # 1, 2 seconds
                time.sleep(wait_time)
                return self._make_request(method, endpoint, params, auth, retry_count + 1, category)
                
            return {"success": False, "error": f"Invalid JSON response: {str(e)}"}
            
        except Exception as e:
            logger.error(f"API request error: {endpoint} - {e}")
            
            # Retry other unexpected errors
            if retry_count < 2:
                wait_time = 2 ** retry_count  # 1, 2 seconds
                time.sleep(wait_time)
                return self._make_request(method, endpoint, params, auth, retry_count + 1, category)
                
            return {"success": False, "error": str(e)}
    
    def _sync_time(self):
        """Synchronize time with the exchange server."""
        try:
            # Get server time
            start_time = time.time()
            time_result = self.exchange.fetch_time()
            end_time = time.time()
            
            # Calculate network latency (round trip time / 2)
            latency = (end_time - start_time) / 2
            
            # Calculate time offset (server time - local time)
            self.time_offset = time_result - int((start_time + end_time) / 2 * 1000)
            
            logger.info(f"Time synchronized with server: offset {self.time_offset}ms, latency {latency*1000:.0f}ms")
            
        except Exception as e:
            logger.error(f"Error synchronizing time: {e}")
    
    # Market Data Methods with improved error handling and response normalization
    def get_klines(self, symbol: str, interval: str, limit: int = 200, start_time: int = None) -> pd.DataFrame:
        """Get candlestick data with comprehensive error handling and normalization."""
        try:
            # Normalize symbol format if needed
            if '/' in symbol:
                ccxt_symbol = symbol
            else:
                # Convert from plain format (e.g., BTCUSDT) to ccxt format (BTC/USDT)
                base, quote = symbol[:-4], symbol[-4:]
                ccxt_symbol = f"{base}/{quote}"
            
            # Use standard retry mechanism
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Use CCXT to get OHLCV data
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=ccxt_symbol,
                        timeframe=interval,
                        limit=limit,
                        since=start_time
                    )
                    
                    # Create DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Check for data quality
                    if len(df) < limit * 0.9 and start_time is None:
                        logger.warning(f"Incomplete klines data for {symbol}: {len(df)}/{limit} candles")
                    
                    # Check for duplicates
                    if len(df) != len(df.index.unique()):
                        logger.warning(f"Duplicate timestamps in klines for {symbol}")
                        # Remove duplicates
                        df = df[~df.index.duplicated(keep='last')]
                    
                    # Sort by timestamp
                    df = df.sort_index()
                    
                    return df
                    
                except ccxt.NetworkError as e:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.warning(f"Network error fetching klines for {symbol}, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                    
                    if retry_count < max_retries:
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch klines after {max_retries} retries")
                        return pd.DataFrame()
                        
                except ccxt.ExchangeError as e:
                    if 'rate limit' in str(e).lower():
                        retry_count += 1
                        wait_time = 5 * retry_count  # Longer wait for rate limiting
                        logger.warning(f"Rate limit hit fetching klines for {symbol}, retry {retry_count}/{max_retries} in {wait_time}s")
                        
                        if retry_count < max_retries:
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to fetch klines after {max_retries} retries due to rate limits")
                            return pd.DataFrame()
                    else:
                        logger.error(f"Exchange error fetching klines for {symbol}: {e}")
                        return pd.DataFrame()
                        
                except Exception as e:
                    logger.error(f"Unexpected error fetching klines for {symbol}: {e}")
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Error in get_klines for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_ticker(self, symbol: str) -> dict:
        """Get ticker information for a symbol with enhanced error handling."""
        try:
            # Normalize symbol format
            if '/' in symbol:
                ccxt_symbol = symbol
            else:
                # Convert from plain format (e.g., BTCUSDT) to ccxt format (BTC/USDT)
                base, quote = symbol[:-4], symbol[-4:]
                ccxt_symbol = f"{base}/{quote}"
            
            # Use CCXT with retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    ticker = self.exchange.fetch_ticker(ccxt_symbol)
                    
                    # Normalize the response
                    result = {
                        "symbol": ticker["symbol"],
                        "last_price": float(ticker["last"]) if ticker["last"] else None,
                        "bid": float(ticker["bid"]) if ticker["bid"] else None,
                        "ask": float(ticker["ask"]) if ticker["ask"] else None,
                        "high": float(ticker["high"]) if ticker["high"] else None,
                        "low": float(ticker["low"]) if ticker["low"] else None,
                        "volume": float(ticker["quoteVolume"]) if ticker["quoteVolume"] else None,
                        "timestamp": ticker["timestamp"],
                        "percentage": ticker["percentage"] if "percentage" in ticker else None,
                        "baseVolume": float(ticker["baseVolume"]) if "baseVolume" in ticker and ticker["baseVolume"] else None,
                    }
                    
                    return result
                    
                except ccxt.NetworkError as e:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.warning(f"Network error fetching ticker for {symbol}, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                    
                    if retry_count < max_retries:
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch ticker after {max_retries} retries")
                        return {}
                        
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error fetching ticker for {symbol}: {e}")
                    return {}
                    
                except Exception as e:
                    logger.error(f"Unexpected error fetching ticker for {symbol}: {e}")
                    return {}
            
        except Exception as e:
            logger.error(f"Error in get_ticker for {symbol}: {e}")
            return {}
    
    def get_order_book(self, symbol: str, limit: int = 50) -> dict:
        """Get order book for a symbol with robust error handling."""
        try:
            # Normalize symbol format
            if '/' in symbol:
                ccxt_symbol = symbol
            else:
                # Convert from plain format (e.g., BTCUSDT) to ccxt format (BTC/USDT)
                base, quote = symbol[:-4], symbol[-4:]
                ccxt_symbol = f"{base}/{quote}"
            
            # Use CCXT with retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    order_book = self.exchange.fetch_order_book(ccxt_symbol, limit)
                    
                    # Verify data integrity 
                    if not order_book or "bids" not in order_book or "asks" not in order_book:
                        raise ValueError("Invalid order book data")
                    
                    # Check for proper price ordering
                    if order_book["bids"] and len(order_book["bids"]) > 1:
                        if order_book["bids"][0][0] < order_book["bids"][1][0]:
                            logger.warning(f"Order book bids not properly sorted for {symbol}")
                            order_book["bids"] = sorted(order_book["bids"], key=lambda x: x[0], reverse=True)
                    
                    if order_book["asks"] and len(order_book["asks"]) > 1:
                        if order_book["asks"][0][0] > order_book["asks"][1][0]:
                            logger.warning(f"Order book asks not properly sorted for {symbol}")
                            order_book["asks"] = sorted(order_book["asks"], key=lambda x: x[0])
                    
                    # Normalize response
                    result = {
                        "symbol": symbol,
                        "bids": order_book["bids"],
                        "asks": order_book["asks"],
                        "timestamp": order_book["timestamp"],
                        "datetime": order_book["datetime"] if "datetime" in order_book else None,
                        "nonce": order_book["nonce"] if "nonce" in order_book else None,
                    }
                    
                    return result
                    
                except ccxt.NetworkError as e:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.warning(f"Network error fetching order book for {symbol}, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                    
                    if retry_count < max_retries:
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch order book after {max_retries} retries")
                        return {}
                        
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error fetching order book for {symbol}: {e}")
                    return {}
                    
                except Exception as e:
                    logger.error(f"Unexpected error fetching order book for {symbol}: {e}")
                    return {}
            
        except Exception as e:
            logger.error(f"Error in get_order_book for {symbol}: {e}")
            return {}
    
    # Account Methods with improved error handling and normalization
    def get_wallet_balance(self, coin: str = None) -> dict:
        """Get wallet balance with robust error handling and retry logic."""
        try:
            # Use CCXT with retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    balances = self.exchange.fetch_balance()
                    
                    # Check for specific coin
                    if coin:
                        normalized_coin = coin.upper()
                        if normalized_coin in balances['total']:
                            return {
                                "coin": normalized_coin,
                                "free": float(balances['free'].get(normalized_coin, 0)),
                                "used": float(balances['used'].get(normalized_coin, 0)),
                                "total": float(balances['total'].get(normalized_coin, 0)),
                            }
                        return {"coin": normalized_coin, "free": 0, "used": 0, "total": 0}
                    
                    # Return full balance
                    return balances
                    
                except ccxt.NetworkError as e:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.warning(f"Network error fetching wallet balance, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                    
                    if retry_count < max_retries:
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch wallet balance after {max_retries} retries")
                        return {}
                        
                except ccxt.ExchangeError as e:
                    if 'rate limit' in str(e).lower():
                        retry_count += 1
                        wait_time = 5 * retry_count  # Longer wait for rate limiting
                        logger.warning(f"Rate limit hit fetching wallet balance, retry {retry_count}/{max_retries} in {wait_time}s")
                        
                        if retry_count < max_retries:
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to fetch wallet balance after {max_retries} retries due to rate limits")
                            return {}
                    elif 'authentication' in str(e).lower():
                        logger.error(f"Authentication error fetching wallet balance: {e}")
                        self.health_status = False
                        return {}
                    else:
                        logger.error(f"Exchange error fetching wallet balance: {e}")
                        return {}
                        
                except Exception as e:
                    logger.error(f"Unexpected error fetching wallet balance: {e}")
                    return {}
            
        except Exception as e:
            logger.error(f"Error in get_wallet_balance: {e}")
            return {}
    
    def get_positions(self, symbol: str = None) -> List[dict]:
        """Get open positions with robust error handling and normalization."""
        try:
            # Rate limit check for position endpoint
            if not self.rate_limits["position"].can_call():
                wait_time = self.rate_limits["position"].get_wait_time()
                logger.warning(f"Rate limit reached for position endpoint, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            # Mark rate limit usage
            self.rate_limits["position"].call_made()
            
            # Use CCXT with retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Normalize symbol format if provided
                    ccxt_symbol = None
                    if symbol:
                        if '/' in symbol:
                            ccxt_symbol = symbol
                        else:
                            # Convert from plain format (e.g., BTCUSDT) to ccxt format (BTC/USDT)
                            base, quote = symbol[:-4], symbol[-4:]
                            ccxt_symbol = f"{base}/{quote}"
                    
                    # Fetch positions
                    positions = self.exchange.fetch_positions(ccxt_symbol)
                    
                    # Normalize the response
                    normalized_positions = []
                    for pos in positions:
                        # Only include positions with non-zero size
                        if float(pos['contracts']) != 0:
                            normalized_pos = {
                                "id": pos.get('id', str(uuid.uuid4())),
                                "symbol": pos['symbol'],
                                "side": pos['side'],
                                "size": float(pos['contracts']),
                                "value": float(pos['notional']) if 'notional' in pos else None,
                                "entry_price": float(pos['entryPrice']) if 'entryPrice' in pos else None,
                                "mark_price": float(pos['markPrice']) if 'markPrice' in pos else None,
                                "liquidation_price": float(pos['liquidationPrice']) if 'liquidationPrice' in pos and pos['liquidationPrice'] else None,
                                "margin_mode": pos.get('marginMode', 'cross'),
                                "leverage": float(pos['leverage']) if 'leverage' in pos else None,
                                "unrealized_pnl": float(pos['unrealizedPnl']) if 'unrealizedPnl' in pos else None,
                                "margin": float(pos['initialMargin']) if 'initialMargin' in pos else None,
                                "timestamp": pos.get('timestamp', int(time.time() * 1000)),
                                "stop_loss": float(pos['stopLoss']) if 'stopLoss' in pos and pos['stopLoss'] else None,
                                "take_profit": float(pos['takeProfit']) if 'takeProfit' in pos and pos['takeProfit'] else None,
                            }
                            normalized_positions.append(normalized_pos)
                    
                    return normalized_positions
                    
                except ccxt.NetworkError as e:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.warning(f"Network error fetching positions, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                    
                    if retry_count < max_retries:
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch positions after {max_retries} retries")
                        return []
                        
                except ccxt.ExchangeError as e:
                    if 'rate limit' in str(e).lower():
                        retry_count += 1
                        wait_time = 5 * retry_count  # Longer wait for rate limiting
                        logger.warning(f"Rate limit hit fetching positions, retry {retry_count}/{max_retries} in {wait_time}s")
                        
                        if retry_count < max_retries:
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to fetch positions after {max_retries} retries due to rate limits")
                            return []
                    else:
                        logger.error(f"Exchange error fetching positions: {e}")
                        return []
                        
                except Exception as e:
                    logger.error(f"Unexpected error fetching positions: {e}")
                    return []
            
        except Exception as e:
            logger.error(f"Error in get_positions: {e}")
            return []
    
    # Trading Methods with improved error handling, retry logic, and validation
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "LIMIT", 
                    price: float = None, time_in_force: str = "GTC", reduce_only: bool = False,
                    close_on_trigger: bool = False, stop_loss: float = None, 
                    take_profit: float = None) -> dict:
        """Place a new order with comprehensive validation and error handling."""
        try:
            # Rate limit check for order endpoint
            if not self.rate_limits["order"].can_call():
                wait_time = self.rate_limits["order"].get_wait_time()
                logger.warning(f"Rate limit reached for order endpoint, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            # Mark rate limit usage
            self.rate_limits["order"].call_made()
            
            # Validate inputs
            if not symbol or not side or quantity <= 0:
                return {"success": False, "error": "Invalid order parameters"}
            
            if order_type.upper() == "LIMIT" and (price is None or price <= 0):
                return {"success": False, "error": "Price is required for LIMIT orders"}
            
            # Normalize symbol format
            if '/' in symbol:
                ccxt_symbol = symbol
            else:
                # Convert from plain format (e.g., BTCUSDT) to ccxt format (BTC/USDT)
                base, quote = symbol[:-4], symbol[-4:]
                ccxt_symbol = f"{base}/{quote}"
            
            # Normalize side
            ccxt_side = side.lower()
            if ccxt_side not in ['buy', 'sell']:
                return {"success": False, "error": f"Invalid side: {side}"}
            
            # Prepare parameters
            params = {}
            if stop_loss:
                params["stopLoss"] = stop_loss
            if take_profit:
                params["takeProfit"] = take_profit
            if reduce_only:
                params["reduceOnly"] = reduce_only
            if close_on_trigger:
                params["closeOnTrigger"] = close_on_trigger
            if time_in_force:
                params["timeInForce"] = time_in_force
            
            # Use retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Execute order based on type
                    if order_type.upper() == "MARKET":
                        order = self.exchange.create_market_order(
                            symbol=ccxt_symbol,
                            side=ccxt_side,
                            amount=quantity,
                            params=params
                        )
                    elif order_type.upper() == "LIMIT":
                        order = self.exchange.create_limit_order(
                            symbol=ccxt_symbol,
                            side=ccxt_side,
                            amount=quantity,
                            price=price,
                            params=params
                        )
                    else:
                        return {"success": False, "error": f"Unsupported order type: {order_type}"}
                    
                    # Normalize response
                    normalized_order = {
                        "id": order.get('id', ''),
                        "symbol": order.get('symbol', ccxt_symbol),
                        "side": order.get('side', ccxt_side),
                        "type": order.get('type', order_type.lower()),
                        "price": float(order['price']) if 'price' in order and order['price'] else None,
                        "amount": float(order['amount']) if 'amount' in order else quantity,
                        "filled": float(order['filled']) if 'filled' in order else 0,
                        "remaining": float(order['remaining']) if 'remaining' in order else quantity,
                        "status": order.get('status', 'unknown'),
                        "timestamp": order.get('timestamp', int(time.time() * 1000)),
                        "fee": order.get('fee', {}),
                        "trades": order.get('trades', []),
                    }
                    
                    logger.info(f"Order placed: {normalized_order['id']} - {symbol} {side} {quantity} @ {price}")
                    return normalized_order
                    
                except ccxt.NetworkError as e:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.warning(f"Network error placing order, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                    
                    if retry_count < max_retries:
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to place order after {max_retries} retries")
                        return {"success": False, "error": str(e)}
                        
                except ccxt.InsufficientFunds as e:
                    logger.error(f"Insufficient funds to place order: {e}")
                    return {"success": False, "error": "Insufficient funds", "details": str(e)}
                    
                except ccxt.InvalidOrder as e:
                    logger.error(f"Invalid order parameters: {e}")
                    return {"success": False, "error": "Invalid order", "details": str(e)}
                    
                except ccxt.ExchangeError as e:
                    if 'rate limit' in str(e).lower():
                        retry_count += 1
                        wait_time = 5 * retry_count  # Longer wait for rate limiting
                        logger.warning(f"Rate limit hit placing order, retry {retry_count}/{max_retries} in {wait_time}s")
                        
                        if retry_count < max_retries:
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to place order after {max_retries} retries due to rate limits")
                            return {"success": False, "error": "Rate limit exceeded", "details": str(e)}
                    else:
                        logger.error(f"Exchange error placing order: {e}")
                        return {"success": False, "error": "Exchange error", "details": str(e)}
                        
                except Exception as e:
                    logger.error(f"Unexpected error placing order: {e}")
                    return {"success": False, "error": str(e)}
            
        except Exception as e:
            logger.error(f"Error in place_order: {e}")
            return {"success": False, "error": str(e)}
    
    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an order with robust error handling and validation."""
        try:
            # Rate limit check for order endpoint
            if not self.rate_limits["order"].can_call():
                wait_time = self.rate_limits["order"].get_wait_time()
                logger.warning(f"Rate limit reached for order endpoint, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            # Mark rate limit usage
            self.rate_limits["order"].call_made()
            
            # Validate inputs
            if not order_id or not symbol:
                return {"success": False, "error": "Order ID and symbol are required"}
            
            # Normalize symbol format
            if '/' in symbol:
                ccxt_symbol = symbol
            else:
                # Convert from plain format (e.g., BTCUSDT) to ccxt format (BTC/USDT)
                base, quote = symbol[:-4], symbol[-4:]
                ccxt_symbol = f"{base}/{quote}"
            
            # Use retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Cancel the order
                    result = self.exchange.cancel_order(order_id, ccxt_symbol)
                    
                    # Normalize response
                    normalized_result = {
                        "id": result.get('id', order_id),
                        "symbol": result.get('symbol', ccxt_symbol),
                        "status": result.get('status', 'canceled'),
                        "success": True
                    }
                    
                    logger.info(f"Order cancelled: {order_id} for {symbol}")
                    return normalized_result
                    
                except ccxt.OrderNotFound as e:
                    logger.warning(f"Order not found for cancellation: {order_id} - {e}")
                    return {"success": False, "error": "Order not found", "details": str(e)}
                    
                except ccxt.NetworkError as e:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.warning(f"Network error cancelling order, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                    
                    if retry_count < max_retries:
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to cancel order after {max_retries} retries")
                        return {"success": False, "error": str(e)}
                        
                except ccxt.ExchangeError as e:
                    if 'rate limit' in str(e).lower():
                        retry_count += 1
                        wait_time = 5 * retry_count  # Longer wait for rate limiting
                        logger.warning(f"Rate limit hit cancelling order, retry {retry_count}/{max_retries} in {wait_time}s")
                        
                        if retry_count < max_retries:
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to cancel order after {max_retries} retries due to rate limits")
                            return {"success": False, "error": "Rate limit exceeded", "details": str(e)}
                    else:
                        logger.error(f"Exchange error cancelling order: {e}")
                        return {"success": False, "error": "Exchange error", "details": str(e)}
                        
                except Exception as e:
                    logger.error(f"Unexpected error cancelling order: {e}")
                    return {"success": False, "error": str(e)}
            
        except Exception as e:
            logger.error(f"Error in cancel_order: {e}")
            return {"success": False, "error": str(e)}
    
    def health_check(self) -> bool:
        """Perform a health check on the API connection."""
        try:
            # Check if we've had a successful call recently
            if time.time() - self.last_successful_call < 300:  # Less than 5 minutes ago
                return self.health_status
            
            # Test the connection
            result = self._make_request("GET", "/v5/market/time", auth=False)
            
            if result and 'retCode' in result and result['retCode'] == 0:
                self.health_status = True
                self.last_successful_call = time.time()
                logger.info("API health check: OK")
                return True
            else:
                logger.warning(f"API health check failed: {result}")
                self.health_status = False
                return False
                
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            self.health_status = False
            return False
    
    def fetch_server_time(self) -> int:
        """Fetch current server time."""
        try:
            result = self._make_request("GET", "/v5/market/time", auth=False)
            
            if result and 'retCode' in result and result['retCode'] == 0 and 'result' in result:
                server_time = result['result'].get('timeNano', 0) // 1000000  # Convert ns to ms
                return server_time
            else:
                logger.warning(f"Failed to fetch server time: {result}")
                return int(time.time() * 1000)  # Fall back to local time
                
        except Exception as e:
            logger.error(f"Error fetching server time: {e}")
            return int(time.time() * 1000)  # Fall back to local time


class RateLimiter:
    """Helper class to manage API rate limits."""
    
    def __init__(self, max_calls: int, period: int):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
    
    def can_call(self) -> bool:
        """Check if a call can be made without exceeding rate limits."""
        with self.lock:
            now = time.time()
            
            # Remove expired calls
            self.calls = [t for t in self.calls if now - t < self.period]
            
            # Check if we can make a new call
            return len(self.calls) < self.max_calls
    
    def call_made(self):
        """Record that a call was made."""
        with self.lock:
            now = time.time()
            
            # Remove expired calls
            self.calls = [t for t in self.calls if now - t < self.period]
            
            # Add new call
            self.calls.append(now)
    
    def get_wait_time(self) -> float:
        """Get recommended wait time if rate limited."""
        with self.lock:
            now = time.time()
            
            # Remove expired calls
            self.calls = [t for t in self.calls if now - t < self.period]
            
            # If under the limit, no need to wait
            if len(self.calls) < self.max_calls:
                return 0
                
            # Calculate wait time until oldest call expires
            if self.calls:
                return max(0, self.period - (now - self.calls[0]))
            
            return 0

class FeatureEngineering:
    """Feature engineering for the trading bot."""
    
    def __init__(self, config):
        """Initialize feature engineering with configuration."""
        self.config = config
        self.feature_window = config.get("ml_settings", "feature_window")
        self.scalers = {}  # Store scaler for each symbol
    
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators on the DataFrame."""
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure we have enough data
        if len(result) < 50:
            logger.warning(f"Not enough data to compute indicators: {len(result)} rows")
            return result
        
        # Price transforms
        result['returns'] = result['close'].pct_change()
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        
        # Simple moving averages
        result['sma_5'] = result['close'].rolling(window=5).mean()
        result['sma_10'] = result['close'].rolling(window=10).mean()
        result['sma_20'] = result['close'].rolling(window=20).mean()
        result['sma_50'] = result['close'].rolling(window=50).mean()
        
        # Exponential moving averages
        result['ema_5'] = result['close'].ewm(span=5, adjust=False).mean()
        result['ema_10'] = result['close'].ewm(span=10, adjust=False).mean()
        result['ema_20'] = result['close'].ewm(span=20, adjust=False).mean()
        result['ema_50'] = result['close'].ewm(span=50, adjust=False).mean()
        
        # Bollinger Bands
        bb = BollingerBands(close=result["close"], window=20, window_dev=2)
        result['bb_upper'] = bb.bollinger_hband()
        result['bb_middle'] = bb.bollinger_mavg()
        result['bb_lower'] = bb.bollinger_lband()
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        result['bb_pct'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # MACD
        macd = MACD(close=result["close"])
        result['macd'] = macd.macd()
        result['macd_signal'] = macd.macd_signal()
        result['macd_diff'] = macd.macd_diff()
        
        # RSI
        rsi = RSIIndicator(close=result["close"])
        result['rsi'] = rsi.rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=result["high"], low=result["low"], close=result["close"])
        result['stoch_k'] = stoch.stoch()
        result['stoch_d'] = stoch.stoch_signal()
        
        # Average True Range
        atr = AverageTrueRange(high=result["high"], low=result["low"], close=result["close"])
        result['atr'] = atr.average_true_range()
        
        # CCI
        cci = CCIIndicator(high=result["high"], low=result["low"], close=result["close"])
        result['cci'] = cci.cci()
        
        # Volume indicators
        result['volume_ma_5'] = result['volume'].rolling(window=5).mean()
        result['volume_ma_10'] = result['volume'].rolling(window=10).mean()
        
        # On-balance volume
        obv = OnBalanceVolumeIndicator(close=result["close"], volume=result["volume"])
        result['obv'] = obv.on_balance_volume()
        
        # Money Flow Index
        mfi = MFIIndicator(high=result["high"], low=result["low"], close=result["close"], volume=result["volume"])
        result['mfi'] = mfi.money_flow_index()
        
        # Volatility measures
        result['volatility_5'] = result['returns'].rolling(window=5).std()
        result['volatility_10'] = result['returns'].rolling(window=10).std()
        result['volatility_20'] = result['returns'].rolling(window=20).std()
        
        # Price momentum
        result['momentum_5'] = result['close'] / result['close'].shift(5) - 1
        result['momentum_10'] = result['close'] / result['close'].shift(10) - 1
        result['momentum_20'] = result['close'] / result['close'].shift(20) - 1
        
        # Candle patterns
        result['body_size'] = abs(result['close'] - result['open']) / (result['high'] - result['low'])
        result['upper_shadow'] = (result['high'] - result['close'].clip(lower=result['open'])) / (result['high'] - result['low'])
        result['lower_shadow'] = (result['close'].clip(upper=result['open']) - result['low']) / (result['high'] - result['low'])
        
        # Ranges
        result['daily_range'] = (result['high'] - result['low']) / result['low']
        result['gap'] = (result['open'] - result['close'].shift(1)) / result['close'].shift(1)
        
        # Cross indicators
        result['sma_5_10_cross'] = (result['sma_5'] > result['sma_10']).astype(int) - (result['sma_5'] < result['sma_10']).astype(int)
        result['sma_10_20_cross'] = (result['sma_10'] > result['sma_20']).astype(int) - (result['sma_10'] < result['sma_20']).astype(int)
        result['ema_5_10_cross'] = (result['ema_5'] > result['ema_10']).astype(int) - (result['ema_5'] < result['ema_10']).astype(int)
        result['ema_10_20_cross'] = (result['ema_10'] > result['ema_20']).astype(int) - (result['ema_10'] < result['ema_20']).astype(int)
        
        # Support and resistance levels
        result['support_level'] = result['low'].rolling(window=20).min()
        result['resistance_level'] = result['high'].rolling(window=20).max()
        result['distance_to_support'] = (result['close'] - result['support_level']) / result['close']
        result['distance_to_resistance'] = (result['resistance_level'] - result['close']) / result['close']
        
        # Trend strength
        adx = ADXIndicator(high=result["high"], low=result["low"], close=result["close"])
        result['adx'] = adx.adx()
        result['adx_pos'] = adx.adx_pos()
        result['adx_neg'] = adx.adx_neg()
        
        # Market regime features
        result['regime_bull'] = ((result['sma_20'] > result['sma_50']) & 
                                 (result['ema_20'] > result['ema_50'])).astype(int)
        result['regime_bear'] = ((result['sma_20'] < result['sma_50']) & 
                                 (result['ema_20'] < result['ema_50'])).astype(int)
        result['regime_high_vol'] = (result['volatility_20'] > result['volatility_20'].rolling(window=100).mean()).astype(int)
        
        # Clean up NaN values
        result.fillna(method='bfill', inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def add_lagged_features(self, df: pd.DataFrame, lag_periods: List[int]) -> pd.DataFrame:
        """Add lagged features to the dataframe."""
        result = df.copy()
        
        # Select columns to lag (excluding those ending with '_lag_X')
        cols_to_lag = [col for col in df.columns if not any(f"_lag_{i}" in col for i in lag_periods)]
        
        # Create lagged features
        for period in lag_periods:
            for col in cols_to_lag:
                result[f"{col}_lag_{period}"] = result[col].shift(period)
        
        # Fill NaN values
        result.fillna(method='bfill', inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def add_target_labels(self, df: pd.DataFrame, forward_periods: List[int]) -> pd.DataFrame:
        """Add target labels for machine learning."""
        result = df.copy()
        
        for period in forward_periods:
            # Price change in the future
            result[f'future_return_{period}'] = result['close'].pct_change(periods=period).shift(-period)
            
            # Binary target: 1 if price will go up, 0 if down
            result[f'target_direction_{period}'] = (result[f'future_return_{period}'] > 0).astype(int)
            
            # Ternary target: 1 (significant up), 0 (sideways), -1 (significant down)
            # Define significant as more than 1% move
            result[f'target_ternary_{period}'] = 0
            result.loc[result[f'future_return_{period}'] > 0.01, f'target_ternary_{period}'] = 1
            result.loc[result[f'future_return_{period}'] < -0.01, f'target_ternary_{period}'] = -1
        
        return result
    
    def scale_features(self, df: pd.DataFrame, symbol: str, is_training: bool = False) -> Tuple[pd.DataFrame, dict]:
        """Scale numerical features for model training and prediction."""
        # Columns to exclude from scaling
        exclude_cols = [col for col in df.columns if 'target_' in col or 'future_' in col or col in ['timestamp', 'date']]
        cols_to_scale = [col for col in df.columns if col not in exclude_cols]
        
        result = df.copy()
        
        # Initialize or retrieve scaler
        if is_training or symbol not in self.scalers:
            scaler = StandardScaler()
            self.scalers[symbol] = {
                'scaler': scaler,
                'columns': cols_to_scale
            }
            if len(result) > 0:
                # Fit the scaler on the training data
                scaler.fit(result[cols_to_scale])
        else:
            scaler = self.scalers[symbol]['scaler']
            cols_to_scale = self.scalers[symbol]['columns']
        
        # Apply scaling
        if len(result) > 0 and len(cols_to_scale) > 0:
            # Only use columns that exist in the dataframe
            valid_cols = [col for col in cols_to_scale if col in result.columns]
            if valid_cols:
                result[valid_cols] = scaler.transform(result[valid_cols])
        
        return result
    
    def prepare_features(self, df: pd.DataFrame, symbol: str, is_training: bool = False) -> pd.DataFrame:
        """Prepare complete feature set for model training or prediction."""
        if df.empty:
            logger.warning(f"Empty dataframe provided to prepare_features for {symbol}")
            return pd.DataFrame()
        
        try:
            # Compute technical indicators
            df_with_indicators = self.compute_technical_indicators(df)
            
            # Add lagged features
            df_with_lags = self.add_lagged_features(df_with_indicators, [1, 2, 3, 5, 10])
            
            # Add target variables for training
            if is_training:
                df_with_targets = self.add_target_labels(df_with_lags, [1, 3, 5, 10])
            else:
                df_with_targets = df_with_lags
            
            # Scale features
            df_scaled = self.scale_features(df_with_targets, symbol, is_training)
            
            return df_scaled
        except Exception as e:
            logger.error(f"Error in feature preparation for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_scalers(self, filename: str = "scalers.pkl"):
        """Save feature scalers to a file."""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.scalers, f)
            logger.info(f"Feature scalers saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving feature scalers: {e}")
            return False
    
    def load_scalers(self, filename: str = "scalers.pkl"):
        """Load feature scalers from a file."""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info(f"Feature scalers loaded from {filename}")
                return True
            else:
                logger.warning(f"Scaler file {filename} not found")
                return False
        except Exception as e:
            logger.error(f"Error loading feature scalers: {e}")
            return False


class SentimentAnalyzer:
    """Analyze market sentiment from various sources with real API integrations."""
    
    def __init__(self, config):
        """Initialize sentiment analyzer with configuration."""
        self.config = config
        self.enabled = config.get("sentiment_analysis", "enabled")
        self.sources = config.get("sentiment_analysis", "sources")
        self.refresh_interval = config.get("sentiment_analysis", "refresh_interval_minutes")
        self.impact_weight = config.get("sentiment_analysis", "impact_weight")
        
        # API keys (should be in config)
        self.twitter_api_key = config.get("api_keys", "twitter_api_key", "")
        self.twitter_api_secret = config.get("api_keys", "twitter_api_secret", "")
        self.twitter_bearer_token = config.get("api_keys", "twitter_bearer_token", "")
        
        self.reddit_client_id = config.get("api_keys", "reddit_client_id", "")
        self.reddit_client_secret = config.get("api_keys", "reddit_client_secret", "")
        self.reddit_user_agent = config.get("api_keys", "reddit_user_agent", "")
        
        self.news_api_key = config.get("api_keys", "news_api_key", "")
        
        # Initialize sentiment cache with TTL
        self.sentiment_cache = {}
        self.last_update = {}
        
        # Initialize API clients
        self._init_api_clients()
        
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize finance-specific sentiment model
        self._init_finance_nlp()
        
        # Rate limiting
        self.rate_limits = {
            "twitter": RateLimiter(max_calls=300, period=900),  # 300 calls per 15 min
            "reddit": RateLimiter(max_calls=60, period=60),     # 60 calls per minute
            "news": RateLimiter(max_calls=100, period=86400)    # 100 calls per day
        }
    
    def _init_api_clients(self):
        """Initialize API clients for different sources."""
        self.api_clients = {}
        
        # Twitter API client
        if self.twitter_bearer_token:
            try:
                import tweepy
                self.api_clients["twitter"] = tweepy.Client(
                    bearer_token=self.twitter_bearer_token,
                    consumer_key=self.twitter_api_key,
                    consumer_secret=self.twitter_api_secret,
                    wait_on_rate_limit=True
                )
                logger.info("Twitter API client initialized")
            except Exception as e:
                logger.error(f"Error initializing Twitter API client: {e}")
        
        # Reddit API client
        if self.reddit_client_id and self.reddit_client_secret:
            try:
                import praw
                self.api_clients["reddit"] = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent
                )
                logger.info("Reddit API client initialized")
            except Exception as e:
                logger.error(f"Error initializing Reddit API client: {e}")
        
        # News API client
        if self.news_api_key:
            try:
                from newsapi import NewsApiClient
                self.api_clients["news"] = NewsApiClient(api_key=self.news_api_key)
                logger.info("News API client initialized")
            except Exception as e:
                logger.error(f"Error initializing News API client: {e}")
    
    def _init_finance_nlp(self):
        """Initialize finance-specific NLP model."""
        self.finance_nlp = None
        try:
            if torch.cuda.is_available():
                model_name = "ElKulako/cryptobert"
                self.finance_nlp = pipeline(
                    "sentiment-analysis", 
                    model=model_name,
                    tokenizer=model_name,
                    device=0  # Use GPU
                )
                logger.info("FinBERT sentiment model loaded successfully on GPU")
            else:
                # Fall back to a smaller model for CPU
                model_name = "ElKulako/cryptobert"
                self.finance_nlp = pipeline(
                    "sentiment-analysis", 
                    model=model_name,
                    tokenizer=model_name
                )
                logger.info("FinBERT sentiment model loaded successfully on CPU")
        except Exception as e:
            logger.warning(f"Could not load FinBERT model: {e}. Using VADER only.")
    
    def update_sentiment(self, symbol: str) -> bool:
        """Update sentiment data for a symbol with error handling and retries."""
        if not self.enabled:
            return False
        
        # Check if we need to update
        now = datetime.now()
        if symbol in self.last_update:
            time_diff = (now - self.last_update[symbol]).total_seconds() / 60
            if time_diff < self.refresh_interval:
                logger.debug(f"Sentiment for {symbol} is recent, skipping update")
                return False
        
        logger.info(f"Updating sentiment data for {symbol}")
        try:
            sentiment_data = {
                "timestamp": now,
                "symbol": symbol,
                "overall_score": 0,
                "sources": {}
            }
            
            processed_sources = 0
            total_score = 0
            
            # Process each source with retry logic
            for source in self.sources:
                try:
                    max_retries = 3
                    retry_count = 0
                    
                    while retry_count < max_retries:
                        try:
                            if source == "twitter":
                                if not self.rate_limits["twitter"].can_call():
                                    logger.warning("Twitter API rate limit reached, skipping")
                                    break
                                score, count = self._analyze_twitter_sentiment(symbol)
                            elif source == "reddit":
                                if not self.rate_limits["reddit"].can_call():
                                    logger.warning("Reddit API rate limit reached, skipping")
                                    break
                                score, count = self._analyze_reddit_sentiment(symbol)
                            elif source == "news":
                                if not self.rate_limits["news"].can_call():
                                    logger.warning("News API rate limit reached, skipping")
                                    break
                                score, count = self._analyze_news_sentiment(symbol)
                            else:
                                logger.warning(f"Unknown sentiment source: {source}")
                                break
                            
                            # Successfully got data
                            break
                        except Exception as e:
                            retry_count += 1
                            wait_time = 2 ** retry_count  # Exponential backoff
                            logger.warning(f"Error in {source} sentiment analysis, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                            
                            if retry_count < max_retries:
                                time.sleep(wait_time)
                            else:
                                logger.error(f"Failed to get {source} sentiment after {max_retries} retries")
                                score, count = 0, 0
                    
                    # Add to sentiment data
                    sentiment_data["sources"][source] = {
                        "score": score,
                        "count": count,
                        "timestamp": now.isoformat()
                    }
                    
                    # Only count sources that returned data
                    if count > 0:
                        processed_sources += 1
                        total_score += score
                
                except Exception as e:
                    logger.error(f"Error processing {source} sentiment: {e}")
            
            # Calculate overall score, averaging only sources that returned data
            if processed_sources > 0:
                sentiment_data["overall_score"] = total_score / processed_sources
            
            # Store in cache
            self.sentiment_cache[symbol] = sentiment_data
            self.last_update[symbol] = now
            
            # Persist sentiment data for analysis
            self._save_sentiment_data(symbol, sentiment_data)
            
            logger.info(f"Sentiment update for {symbol} complete: {sentiment_data['overall_score']}")
            return True
        except Exception as e:
            logger.error(f"Error updating sentiment for {symbol}: {e}")
            return False
    
    def _save_sentiment_data(self, symbol: str, sentiment_data: dict):
        """Save sentiment data to disk for later analysis."""
        try:
            # Create sentiment directory if it doesn't exist
            os.makedirs("sentiment_data", exist_ok=True)
            
            # Create symbol-specific file
            filename = f"sentiment_data/{symbol.replace('/', '_')}.json"
            
            # Load existing data if available
            existing_data = []
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        existing_data = json.load(f)
                except:
                    pass
            
            # Add new data and limit history
            existing_data.append(sentiment_data)
            
            # Keep last 1000 entries
            if len(existing_data) > 1000:
                existing_data = existing_data[-1000:]
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")
    
    def _analyze_twitter_sentiment(self, symbol: str) -> Tuple[float, int]:
        """Analyze Twitter sentiment for a symbol using the Twitter API."""
        if "twitter" not in self.api_clients:
            logger.warning("Twitter API client not initialized")
            return 0, 0
        
        try:
            # Mark API call for rate limiting
            self.rate_limits["twitter"].call_made()
            
            # Extract crypto symbol
            crypto_symbol = symbol.split('/')[0]
            
            # Build search query
            query = f"#{crypto_symbol} OR ${crypto_symbol} -is:retweet lang:en"
            
            # Get tweets from past 24 hours
            tweets = self.api_clients["twitter"].search_recent_tweets(
                query=query, 
                max_results=100,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets or not tweets.data:
                logger.info(f"No tweets found for {symbol}")
                return 0, 0
            
            tweet_count = len(tweets.data)
            total_sentiment = 0
            weighted_count = 0
            
            for tweet in tweets.data:
                # Get tweet text
                text = tweet.text
                
                # Get engagement metrics
                likes = tweet.public_metrics['like_count'] if 'like_count' in tweet.public_metrics else 0
                retweets = tweet.public_metrics['retweet_count'] if 'retweet_count' in tweet.public_metrics else 0
                
                # Calculate engagement weight (1 + log of engagement)
                engagement = likes + retweets
                weight = 1.0
                if engagement > 0:
                    weight = 1.0 + math.log(1 + engagement, 10)  # Log base 10 to dampen effect
                
                # Get sentiment
                sentiment = self.analyze_text(text)
                
                # Add weighted sentiment
                total_sentiment += sentiment * weight
                weighted_count += weight
            
            # Calculate weighted average sentiment
            if weighted_count > 0:
                avg_sentiment = total_sentiment / weighted_count
            else:
                avg_sentiment = 0
            
            logger.debug(f"Twitter sentiment for {symbol}: {avg_sentiment:.2f} (from {tweet_count} tweets)")
            return avg_sentiment, tweet_count
        
        except Exception as e:
            logger.error(f"Error analyzing Twitter sentiment: {e}")
            raise  # Re-raise for retry mechanism
    
    def _analyze_reddit_sentiment(self, symbol: str) -> Tuple[float, int]:
        """Analyze Reddit sentiment for a symbol using the Reddit API."""
        if "reddit" not in self.api_clients:
            logger.warning("Reddit API client not initialized")
            return 0, 0
        
        try:
            # Mark API call for rate limiting
            self.rate_limits["reddit"].call_made()
            
            # Extract crypto symbol
            crypto_symbol = symbol.split('/')[0]
            
            # Get posts from relevant subreddits
            subreddits = ["CryptoCurrency", f"{crypto_symbol}", "CryptoMarkets"]
            post_limit = 50
            
            valid_subreddits = []
            for sub_name in subreddits:
                try:
                    # Check if subreddit exists
                    subreddit = self.api_clients["reddit"].subreddit(sub_name)
                    # Try to access posts to confirm it's valid and accessible
                    next(subreddit.hot(limit=1))
                    valid_subreddits.append(subreddit)
                except Exception as e:
                    logger.debug(f"Subreddit r/{sub_name} not accessible: {e}")
            
            if not valid_subreddits:
                logger.info(f"No valid subreddits found for {symbol}")
                return 0, 0
            
            posts = []
            for subreddit in valid_subreddits:
                # Get hot posts
                try:
                    sub_posts = list(subreddit.hot(limit=post_limit))
                    
                    # Filter for relevant posts
                    if subreddit.display_name.lower() != crypto_symbol.lower():
                        sub_posts = [
                            post for post in sub_posts 
                            if crypto_symbol.lower() in post.title.lower() or
                            crypto_symbol.lower() in post.selftext.lower()
                        ]
                    
                    posts.extend(sub_posts)
                except Exception as e:
                    logger.error(f"Error fetching posts from r/{subreddit.display_name}: {e}")
            
            if not posts:
                logger.info(f"No Reddit posts found for {symbol}")
                return 0, 0
            
            post_count = len(posts)
            total_sentiment = 0
            weighted_count = 0
            
            for post in posts:
                # Combine title and text
                content = f"{post.title} {post.selftext}"
                
                # Calculate weight based on score and comments
                weight = 1.0 + math.log(1 + post.score + post.num_comments, 10)
                
                # Get sentiment
                sentiment = self.analyze_text(content)
                
                # Add weighted sentiment
                total_sentiment += sentiment * weight
                weighted_count += weight
                
                # Get top comments for additional sentiment
                try:
                    post.comments.replace_more(limit=0)  # Remove MoreComments
                    top_comments = list(post.comments)[:10]  # Get top 10 comments
                    
                    for comment in top_comments:
                        comment_sentiment = self.analyze_text(comment.body)
                        comment_weight = 1.0 + math.log(1 + comment.score, 10)
                        
                        total_sentiment += comment_sentiment * comment_weight
                        weighted_count += comment_weight
                except Exception as e:
                    logger.debug(f"Error processing comments: {e}")
            
            # Calculate weighted average sentiment
            if weighted_count > 0:
                avg_sentiment = total_sentiment / weighted_count
            else:
                avg_sentiment = 0
            
            logger.debug(f"Reddit sentiment for {symbol}: {avg_sentiment:.2f} (from {post_count} posts)")
            return avg_sentiment, post_count
        
        except Exception as e:
            logger.error(f"Error analyzing Reddit sentiment: {e}")
            raise  # Re-raise for retry mechanism
    
    def _analyze_news_sentiment(self, symbol: str) -> Tuple[float, int]:
        """Analyze news sentiment for a symbol using news APIs."""
        if "news" not in self.api_clients:
            logger.warning("News API client not initialized")
            return 0, 0
        
        try:
            # Mark API call for rate limiting
            self.rate_limits["news"].call_made()
            
            # Extract crypto symbol
            crypto_symbol = symbol.split('/')[0]
            
            # Create keyword variations
            # E.g., for BTC: "Bitcoin", "BTC"
            crypto_names = {
                "BTC": ["Bitcoin", "BTC"],
                "ETH": ["Ethereum", "ETH"],
                "SOL": ["Solana", "SOL"],
                "ADA": ["Cardano", "ADA"],
                "XRP": ["Ripple", "XRP"],
                "DOT": ["Polkadot", "DOT"],
                "DOGE": ["Dogecoin", "DOGE"],
                "AVAX": ["Avalanche", "AVAX"],
                "LTC": ["Litecoin", "LTC"],
                "LINK": ["Chainlink", "LINK"],
                "UNI": ["Uniswap", "UNI"],
                "BNB": ["Binance Coin", "BNB"],
                "MATIC": ["Polygon", "MATIC"],
                "XLM": ["Stellar", "XLM"],
                "VET": ["VeChain", "VET"],
                "TRX": ["TRON", "TRX"],
                "EOS": ["EOS", "EOS"],
                "ATOM": ["Cosmos", "ATOM"],
                "ALGO": ["Algorand", "ALGO"],
                "XTZ": ["Tezos", "XTZ"],
                "FIL": ["Filecoin", "FIL"],
                "AAVE": ["Aave", "AAVE"],
                "MKR": ["Maker", "MKR"],
                "COMP": ["Compound", "COMP"]
                # Add more as needed
            }
            
            keywords = [crypto_symbol]
            if crypto_symbol in crypto_names:
                keywords.extend(crypto_names[crypto_symbol])
            
            # Build search query (comma-separated for OR in News API)
            query = " OR ".join(keywords)
            
            # Get news from past week
            one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            news_results = self.api_clients["news"].get_everything(
                q=query,
                from_param=one_week_ago,
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            if not news_results or "articles" not in news_results or not news_results["articles"]:
                logger.info(f"No news articles found for {symbol}")
                return 0, 0
            
            articles = news_results["articles"]
            article_count = len(articles)
            
            total_sentiment = 0
            weighted_count = 0
            
            for article in articles:
                # Combine title and description
                content = f"{article['title']} {article['description'] or ''}"
                
                # Simple age-based weighting (newer articles have more weight)
                pub_date = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                days_old = (datetime.now() - pub_date).days + 1  # Add 1 to avoid division by zero
                
                # Weight decreases with age (newest = 1.0, week old = 0.14)
                age_weight = 1.0 / days_old
                
                # Source credibility weight (could be expanded with a source credibility database)
                source_weight = 1.0
                
                # Combined weight
                weight = age_weight * source_weight
                
                # Get sentiment
                sentiment = self.analyze_text(content)
                
                # Add weighted sentiment
                total_sentiment += sentiment * weight
                weighted_count += weight
            
            # Calculate weighted average sentiment
            if weighted_count > 0:
                avg_sentiment = total_sentiment / weighted_count
            else:
                avg_sentiment = 0
            
            logger.debug(f"News sentiment for {symbol}: {avg_sentiment:.2f} (from {article_count} articles)")
            return avg_sentiment, article_count
        
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            raise  # Re-raise for retry mechanism
    
    def get_sentiment(self, symbol: str) -> dict:
        """Get the current sentiment for a symbol with cache management."""
        if not self.enabled:
            return {"overall_score": 0, "enabled": False}
        
        # Check cache freshness
        now = datetime.now()
        if symbol in self.last_update:
            time_diff = (now - self.last_update[symbol]).total_seconds() / 60
            
            # If cache is stale but not expired (>80% of refresh interval)
            if time_diff > self.refresh_interval * 0.8 and time_diff < self.refresh_interval:
                # Trigger async update if possible
                threading.Thread(target=self.update_sentiment, args=(symbol,), daemon=True).start()
        
        # Update if needed (and not already triggered above)
        if symbol not in self.sentiment_cache:
            self.update_sentiment(symbol)
        
        # Return from cache (even if it's stale, the update will happen in background)
        sentiment_data = self.sentiment_cache.get(symbol, {"overall_score": 0})
        
        # Add cache age information
        if symbol in self.last_update:
            sentiment_data["cache_age_minutes"] = (now - self.last_update[symbol]).total_seconds() / 60
        
        return sentiment_data
    
    def get_sentiment_signal(self, symbol: str) -> float:
        """Get a trading signal from sentiment analysis (-1 to 1)."""
        if not self.enabled:
            return 0
        
        sentiment_data = self.get_sentiment(symbol)
        raw_score = sentiment_data.get("overall_score", 0)
        
        # Apply a sigmoid transform to keep in range and reduce extreme values
        # This makes the signal more stable while preserving direction
        transformed_score = 2 / (1 + math.exp(-2 * raw_score)) - 1
        
        # Apply impact weight from config
        return transformed_score * self.impact_weight
    
    def analyze_text(self, text: str) -> float:
        """Analyze the sentiment of a text snippet with error handling."""
        if not text:
            return 0
        
        try:
            # Use both VADER and FinBERT for better accuracy
            vader_scores = self.vader.polarity_scores(text)
            vader_compound = vader_scores['compound']
            
            if self.finance_nlp:
                try:
                    # Limit text length for transformer models
                    text_chunk = text[:512]  # Truncate to max length
                    
                    # Use finance-specific model
                    result = self.finance_nlp(text_chunk)
                    
                    if result[0]['label'] == 'POSITIVE':
                        finbert_score = result[0]['score']
                    elif result[0]['label'] == 'NEGATIVE':
                        finbert_score = -result[0]['score']
                    else:
                        finbert_score = 0
                    
                    # Average the scores, with more weight to the specialized model
                    weighted_score = (vader_compound + 2 * finbert_score) / 3
                    return weighted_score
                except Exception as e:
                    logger.debug(f"Error with FinBERT analysis, falling back to VADER: {e}")
                    return vader_compound
            else:
                # Fall back to just VADER if FinBERT isn't available
                return vader_compound
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return 0


class RateLimiter:
    """Helper class to manage API rate limits."""
    
    def __init__(self, max_calls: int, period: int):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
    
    def can_call(self) -> bool:
        """Check if a call can be made without exceeding rate limits."""
        with self.lock:
            now = time.time()
            
            # Remove expired calls
            self.calls = [t for t in self.calls if now - t < self.period]
            
            # Check if we can make a new call
            return len(self.calls) < self.max_calls
    
    def call_made(self):
        """Record that a call was made."""
        with self.lock:
            now = time.time()
            
            # Remove expired calls
            self.calls = [t for t in self.calls if now - t < self.period]
            
            # Add new call
            self.calls.append(now)

class MLModels:
    """Machine learning models for trading signals."""
    
    def __init__(self, config, features):
        """Initialize ML models with configuration."""
        self.config = config
        self.features = features
        self.models = {}  # Store models for each symbol and timeframe
        self.last_trained = {}  # Track when models were last trained
        self.train_interval = config.get("ml_settings", "train_interval_days")
        self.prediction_window = config.get("ml_settings", "prediction_window")
        self.confidence_threshold = config.get("ml_settings", "confidence_threshold")
        self.ensemble_method = config.get("ml_settings", "ensemble_method")
        self.model_weights = config.get("ml_settings", "model_weights")
        
        # Model metadata
        self.feature_importance = {}
        self.model_performance = {}
        
        # Create model directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
    
    def needs_training(self, symbol: str, timeframe: str) -> bool:
        """Check if a model needs to be trained."""
        model_key = f"{symbol}_{timeframe}"
        
        # Check if model exists
        if model_key not in self.models or not self.models[model_key]:
            return True
        
        # Check if it's time to retrain
        if model_key in self.last_trained:
            days_since_trained = (datetime.now() - self.last_trained[model_key]).days
            return days_since_trained >= self.train_interval
        
        return True
    
    def train_models(self, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        """Train machine learning models for a symbol and timeframe."""
        model_key = f"{symbol}_{timeframe}"
        
        logger.info(f"Training models for {symbol} on {timeframe} timeframe")
        try:
            # Check if we have enough data
            min_rows = 250  # Reduced from 500 to require less data
            if len(df) < min_rows:
                logger.warning(f"Not enough data to train models for {symbol} ({len(df)} rows, need {min_rows})")
                return False
            
            # Free memory before starting
            gc.collect()
            
            # Prepare features for training - with reduced feature window
            df_features = self.features.prepare_features(df, symbol, is_training=True)
            
            # Define target columns
            target_cols = [col for col in df_features.columns if col.startswith('target_')]
            
            if not target_cols:
                logger.error(f"No target columns found in data for {symbol}")
                return False
            
            # Create time series split for validation - with fewer splits
            tscv = TimeSeriesSplit(n_splits=3)  # Reduced from 5
            
            # Drop any non-feature columns
            feature_cols = [col for col in df_features.columns if not col.startswith('target_') and not col.startswith('future_')]
            
            # Limit the number of features to reduce memory usage
            if len(feature_cols) > 500:  # Cap if too many features
                # Sort features by importance if we have previous data
                if f"{model_key}_target_direction_1" in self.feature_importance:
                    # Use previous feature importance to select features
                    importance_dict = self.feature_importance[f"{model_key}_target_direction_1"].get('xgboost', {})
                    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    top_features = [f[0] for f in sorted_features[:500] if f[0] in feature_cols]
                    
                    # Ensure we have 200 features even if some aren't in the importance dict
                    remaining = 500 - len(top_features)
                    if remaining > 0:
                        remaining_features = [f for f in feature_cols if f not in top_features][:remaining]
                        feature_cols = top_features + remaining_features
                    else:
                        feature_cols = top_features
                else:
                    # Without importance data, just take the first 200
                    feature_cols = feature_cols[:500]
                    
                logger.info(f"Limited feature count to {len(feature_cols)} for {symbol} to reduce memory usage")
            
            # Store the feature columns used for training
            if not hasattr(self, 'feature_cols'):
                self.feature_cols = {}
            self.feature_cols[model_key] = feature_cols
            
            self.models[model_key] = {}
            self.model_performance[model_key] = {}
            
            # Try to get model parameters from config, fall back to defaults if not found
            try:
                xgb_params = self.config.config.get("ml_settings", {}).get("model_params", {}).get("xgboost", {})
                if not xgb_params:
                    xgb_params = {
                        "n_estimators": 50,
                        "learning_rate": 0.05,
                        "max_depth": 4,
                        "min_child_weight": 1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "gamma": 0
                    }
            except:
                xgb_params = {
                    "n_estimators": 50,
                    "learning_rate": 0.05,
                    "max_depth": 4,
                    "min_child_weight": 1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "gamma": 0
                }
            
            try:
                cat_params = self.config.config.get("ml_settings", {}).get("model_params", {}).get("catboost", {})
                if not cat_params:
                    cat_params = {
                        "iterations": 50,
                        "learning_rate": 0.05,
                        "depth": 4,
                        "verbose": 0
                    }
            except:
                cat_params = {
                    "iterations": 50,
                    "learning_rate": 0.05,
                    "depth": 4,
                    "verbose": 0
                }
            
            try:
                lstm_params = self.config.config.get("ml_settings", {}).get("model_params", {}).get("lstm", {})
                if not lstm_params:
                    lstm_params = {
                        "epochs": 30,
                        "batch_size": 32
                    }
            except:
                lstm_params = {
                    "epochs": 30,
                    "batch_size": 32
                }
            
            # For each target, train multiple model types
            for target in target_cols:
                # Only process a subset of targets to reduce memory usage
                if "target_direction_1" not in target and "target_direction_5" not in target:
                    continue  # Skip other targets
                
                # Drop rows with NaN in target
                df_target = df_features.dropna(subset=[target])
                
                # Skip if not enough data after dropping NaNs
                if len(df_target) < min_rows:
                    logger.warning(f"Not enough valid data for target {target}")
                    continue
                
                X = df_target[feature_cols]
                y = df_target[target]
                
                # Skip if all target values are the same
                if len(y.unique()) < 2:
                    logger.warning(f"Target {target} has only one class, skipping")
                    continue
                
                # Determine if regression or classification
                is_regression = not (target.startswith('target_direction_') or target.startswith('target_ternary_'))
                
                model_results = {}
                
                # Train XGBoost model with lower resource parameters
                if is_regression:
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=xgb_params.get("n_estimators", 50),
                        learning_rate=xgb_params.get("learning_rate", 0.05),
                        max_depth=xgb_params.get("max_depth", 4),
                        min_child_weight=xgb_params.get("min_child_weight", 1),
                        subsample=xgb_params.get("subsample", 0.8),
                        colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
                        gamma=xgb_params.get("gamma", 0),
                        random_state=42
                    )
                else:
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=xgb_params.get("n_estimators", 50),
                        learning_rate=xgb_params.get("learning_rate", 0.05),
                        max_depth=xgb_params.get("max_depth", 4),
                        min_child_weight=xgb_params.get("min_child_weight", 1),
                        subsample=xgb_params.get("subsample", 0.8),
                        colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
                        gamma=xgb_params.get("gamma", 0),
                        random_state=42
                    )
                
                # Use smaller samples for training to reduce memory use
                train_sample_size = min(1000, len(X))
                
                # Train XGBoost with cross-validation
                xgb_scores = []
                for train_idx, val_idx in tscv.split(X):
                    # Sample training data to reduce memory
                    if len(train_idx) > train_sample_size:
                        np.random.seed(42)
                        train_idx = np.random.choice(train_idx, train_sample_size, replace=False)
                    
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    xgb_model.fit(X_train, y_train)
                    
                    if is_regression:
                        y_pred = xgb_model.predict(X_val)
                        score = mean_squared_error(y_val, y_pred, squared=False)  # RMSE
                    else:
                        y_pred = xgb_model.predict(X_val)
                        score = accuracy_score(y_val, y_pred)
                    
                    xgb_scores.append(score)
                    
                    # Force garbage collection after each fold
                    gc.collect()
                
                # Train final XGBoost model on all data
                if len(X) > train_sample_size:
                    # Sample for final training to reduce memory
                    sample_idx = np.random.choice(len(X), train_sample_size, replace=False)
                    X_sample, y_sample = X.iloc[sample_idx], y.iloc[sample_idx]
                    xgb_model.fit(X_sample, y_sample)
                else:
                    xgb_model.fit(X, y)
                    
                model_results['xgboost'] = {
                    'model': xgb_model,
                    'score': np.mean(xgb_scores),
                    'is_regression': is_regression
                }
                
                gc.collect()  # Another GC after XGBoost
                
                # Train CatBoost model
                if is_regression:
                    cat_model = CatBoostRegressor(
                        iterations=cat_params.get("iterations", 50),
                        learning_rate=cat_params.get("learning_rate", 0.05),
                        depth=cat_params.get("depth", 4),
                        loss_function='RMSE',  # Fixed loss function for regression
                        random_seed=42,
                        verbose=cat_params.get("verbose", 0)
                    )
                else:
                    cat_model = CatBoostClassifier(
                        iterations=cat_params.get("iterations", 50),
                        learning_rate=cat_params.get("learning_rate", 0.05),
                        depth=cat_params.get("depth", 4),
                        loss_function='Logloss',  # Fixed loss function for classification
                        random_seed=42,
                        verbose=cat_params.get("verbose", 0)
                    )
                
                # Train CatBoost with cross-validation
                cat_scores = []
                for train_idx, val_idx in tscv.split(X):
                    # Sample training data to reduce memory
                    if len(train_idx) > train_sample_size:
                        np.random.seed(43)  # Different seed
                        train_idx = np.random.choice(train_idx, train_sample_size, replace=False)
                    
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    cat_model.fit(X_train, y_train)
                    
                    if is_regression:
                        y_pred = cat_model.predict(X_val)
                        score = mean_squared_error(y_val, y_pred, squared=False)  # RMSE
                    else:
                        y_pred = cat_model.predict(X_val)
                        score = accuracy_score(y_val, y_pred)
                    
                    cat_scores.append(score)
                    
                    # Force garbage collection after each fold
                    gc.collect()
                
                # Train final CatBoost model on all data
                if len(X) > train_sample_size:
                    # Sample for final training to reduce memory
                    sample_idx = np.random.choice(len(X), train_sample_size, replace=False)
                    X_sample, y_sample = X.iloc[sample_idx], y.iloc[sample_idx]
                    cat_model.fit(X_sample, y_sample)
                else:
                    cat_model.fit(X, y)
                    
                model_results['catboost'] = {
                    'model': cat_model,
                    'score': np.mean(cat_scores),
                    'is_regression': is_regression
                }
                
                gc.collect()  # Another GC after CatBoost
                
                # Train LSTM model for time series prediction
                lstm_model = self._build_lstm_model(
                    input_dim=X.shape[1], 
                    is_regression=is_regression,
                    lstm_params=lstm_params
                )
                
                # Train LSTM with time series validation
                lstm_scores = []
                for train_idx, val_idx in tscv.split(X):
                    # Sample training data to reduce memory
                    if len(train_idx) > train_sample_size:
                        np.random.seed(44)  # Different seed
                        train_idx = np.random.choice(train_idx, train_sample_size, replace=False)
                    
                    X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
                    y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values
                    
                    # Reshape for LSTM [samples, timesteps, features]
                    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
                    
                    lstm_model.fit(
                        X_train, y_train,
                        epochs=lstm_params.get("epochs", 30),
                        batch_size=lstm_params.get("batch_size", 32),
                        validation_data=(X_val, y_val),
                        verbose=0,
                        callbacks=[
                            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        ]
                    )
                    
                    if is_regression:
                        y_pred = lstm_model.predict(X_val)
                        score = mean_squared_error(y_val, y_pred, squared=False)  # RMSE
                    else:
                        y_pred = (lstm_model.predict(X_val) > 0.5).astype(int)
                        score = accuracy_score(y_val, y_pred)
                    
                    lstm_scores.append(score)
                    
                    # Force garbage collection after each fold
                    tf.keras.backend.clear_session()
                    gc.collect()
                
                # Train final LSTM model on all data
                if len(X) > train_sample_size:
                    # Sample for final training to reduce memory
                    sample_idx = np.random.choice(len(X), train_sample_size, replace=False)
                    X_sample = X.iloc[sample_idx].values.reshape((train_sample_size, 1, X.shape[1]))
                    y_sample = y.iloc[sample_idx].values
                    
                    lstm_model.fit(
                        X_sample, y_sample,
                        epochs=lstm_params.get("epochs", 30),
                        batch_size=lstm_params.get("batch_size", 32),
                        validation_split=0.2,  # Use 20% of data as validation
                        verbose=0,
                        callbacks=[
                            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        ]
                    )
                else:
                    X_reshaped = X.values.reshape((X.shape[0], 1, X.shape[1]))
                    
                    # If dataset is very small, don't use validation split
                    if len(X) >= 10:
                        lstm_model.fit(
                            X_reshaped, y.values,
                            epochs=lstm_params.get("epochs", 30),
                            batch_size=lstm_params.get("batch_size", 32),
                            validation_split=0.2,  # Use 20% of data as validation
                            verbose=0,
                            callbacks=[
                                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                            ]
                        )
                    else:
                        # For very small datasets, monitor loss instead of val_loss
                        lstm_model.fit(
                            X_reshaped, y.values,
                            epochs=lstm_params.get("epochs", 30),
                            batch_size=lstm_params.get("batch_size", 32),
                            verbose=0,
                            callbacks=[
                                EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                            ]
                        )
                
                model_results['lstm'] = {
                    'model': lstm_model,
                    'score': np.mean(lstm_scores),
                    'is_regression': is_regression
                }
                
                # Clear TensorFlow session to free memory
                tf.keras.backend.clear_session()
                
                # Store for this target
                self.models[model_key][target] = model_results
                
                # Store performance metrics
                self.model_performance[model_key][target] = {
                    'xgboost': np.mean(xgb_scores),
                    'catboost': np.mean(cat_scores),
                    'lstm': np.mean(lstm_scores)
                }
                
                # Store feature importance from tree-based models
                if not is_regression:
                    try:
                        feature_imp = {
                            'xgboost': dict(zip(feature_cols, xgb_model.feature_importances_)),
                            'catboost': dict(zip(feature_cols, cat_model.feature_importances_))
                        }
                        self.feature_importance[f"{model_key}_{target}"] = feature_imp
                    except:
                        pass
                
                # Force garbage collection after each target
                gc.collect()
            
            # Save models
            self._save_models(model_key)
            
            # Update last trained time
            self.last_trained[model_key] = datetime.now()
            
            # Final memory cleanup
            gc.collect()
            
            logger.info(f"Model training complete for {symbol} on {timeframe} timeframe")
            return True
        except Exception as e:
            logger.error(f"Error training models for {symbol} on {timeframe}: {e}")
            return False
        
        
    def _build_lstm_model(self, input_dim: int, is_regression: bool, lstm_params: dict = None) -> tf.keras.Model:
        """Build an LSTM model for time series prediction."""
        # Use default params if none provided
        if lstm_params is None:
            lstm_params = {
                "epochs": 30,
                "batch_size": 32
            }
            
        model = Sequential()
        
        # Input layer with shape [batch, timesteps, features]
        model.add(LSTM(64, activation='tanh', return_sequences=True,  # Reduced from 128
                    input_shape=(1, input_dim)))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(32, activation='tanh'))  # Reduced from 64
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(16, activation='relu'))  # Reduced from 32
        model.add(BatchNormalization())
        
        # Output layer
        if is_regression:
            model.add(Dense(1))
        else:
            model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        if is_regression:
            model.compile(optimizer=Adam(0.001), loss='mse')
        else:
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _save_models(self, model_key: str):
        """Save models to disk with improved error handling."""
        try:
            # Convert model_key to safe directory name
            safe_dir_key = model_key.replace("/", "_")
            
            # Create directory if it doesn't exist
            os.makedirs(f"models/{safe_dir_key}", exist_ok=True)
            
            # Keep track of successful saves
            saved_models = 0
            
            # Save each model
            for target, models in self.models[model_key].items():
                try:
                    target_safe = target.replace("/", "_")
                    
                    # Save tree-based models
                    try:
                        with open(f"models/{safe_dir_key}/{target_safe}_xgboost.pkl", 'wb') as f:
                            pickle.dump(models['xgboost']['model'], f)
                            saved_models += 1
                            logger.debug(f"XGBoost model saved for {model_key}/{target_safe}")
                    except Exception as e:
                        logger.error(f"Error saving XGBoost model for {target_safe}: {e}")
                    
                    try:
                        with open(f"models/{safe_dir_key}/{target_safe}_catboost.pkl", 'wb') as f:
                            pickle.dump(models['catboost']['model'], f)
                            saved_models += 1
                            logger.debug(f"CatBoost model saved for {model_key}/{target_safe}")
                    except Exception as e:
                        logger.error(f"Error saving CatBoost model for {target_safe}: {e}")
                    
                    # Save LSTM model
                    try:
                        models['lstm']['model'].save(f"models/{safe_dir_key}/{target_safe}_lstm")
                        saved_models += 1
                        logger.debug(f"LSTM model saved for {model_key}/{target_safe}")
                    except Exception as e:
                        logger.error(f"Error saving LSTM model for {target_safe}: {e}")
                
                except Exception as e:
                    logger.error(f"Error processing target {target} for {model_key}: {e}")
            
            # Save metadata in a robust way
            try:
                metadata = {
                    "last_trained": self.last_trained[model_key].isoformat(),
                    "performance": self.model_performance[model_key]
                }
                
                metadata_file = f"models/{safe_dir_key}/metadata.json"
                
                # Write with explicit encoding and flush
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4)
                    f.flush()
                
                logger.info(f"Metadata saved for {model_key}")
            except Exception as e:
                logger.error(f"Error saving metadata for {model_key}: {str(e)}")
            
            logger.info(f"Models saved for {model_key} ({saved_models} models)")
            return saved_models > 0
        except Exception as e:
            logger.error(f"Error in _save_models for {model_key}: {str(e)}")
            return False
        
    def _load_models(self, model_key: str):
        """Load models from disk."""
        try:
            # Convert model_key to safe directory name
            safe_dir_key = model_key.replace("/", "_")
            
            if not os.path.exists(f"models/{safe_dir_key}"):
                logger.warning(f"No saved models found for {model_key}")
                return False
            
            # Load metadata
            try:
                with open(f"models/{safe_dir_key}/metadata.json", 'r') as f:
                    metadata = json.load(f)
                    self.last_trained[model_key] = datetime.fromisoformat(metadata["last_trained"])
                    self.model_performance[model_key] = metadata["performance"]
            except Exception as e:
                logger.error(f"Error loading metadata for {model_key}: {e}")
                # Continue anyway, we can still try to load models
            
            # Initialize model dictionary
            self.models[model_key] = {}
            
            # Load all models in the directory
            model_files = os.listdir(f"models/{safe_dir_key}")
            for model_file in model_files:
                if model_file.endswith("_xgboost.pkl"):
                    target = model_file.replace("_xgboost.pkl", "").replace("_", "/")
                    
                    # Initialize target in models dict
                    if target not in self.models[model_key]:
                        self.models[model_key][target] = {}
                    
                    # Load XGBoost
                    try:
                        with open(f"models/{safe_dir_key}/{model_file}", 'rb') as f:
                            xgb_model = pickle.load(f)
                            is_regression = not hasattr(xgb_model, 'classes_')
                            self.models[model_key][target]['xgboost'] = {
                                'model': xgb_model,
                                'is_regression': is_regression
                            }
                    except Exception as e:
                        logger.error(f"Error loading XGBoost model for {target}: {e}")
                        continue
                    
                    # Load CatBoost
                    cat_file = model_file.replace("xgboost", "catboost")
                    try:
                        with open(f"models/{safe_dir_key}/{cat_file}", 'rb') as f:
                            cat_model = pickle.load(f)
                            self.models[model_key][target]['catboost'] = {
                                'model': cat_model,
                                'is_regression': is_regression
                            }
                    except Exception as e:
                        logger.error(f"Error loading CatBoost model for {target}: {e}")
                    
                    # Load LSTM
                    lstm_dir = f"models/{safe_dir_key}/{target.replace('/', '_')}_lstm"
                    if os.path.exists(lstm_dir):
                        try:
                            lstm_model = load_model(lstm_dir)
                            self.models[model_key][target]['lstm'] = {
                                'model': lstm_model,
                                'is_regression': is_regression
                            }
                        except Exception as e:
                            logger.error(f"Error loading LSTM model for {target}: {e}")
            
            logger.info(f"Models loaded for {model_key}")
            return bool(self.models[model_key])  # Return True if any models were loaded
        except Exception as e:
            logger.error(f"Error loading models for {model_key}: {e}")
            return False
            
            # Load metadata
            with open(f"models/{model_key.replace('/', '_')}/metadata.json", 'r') as f:
                metadata = json.load(f)
                self.last_trained[model_key] = datetime.fromisoformat(metadata["last_trained"])
                self.model_performance[model_key] = metadata["performance"]
            
            # Load feature columns if available
            if os.path.exists(f"models/{model_key.replace('/', '_')}/feature_columns.pkl"):
                with open(f"models/{model_key.replace('/', '_')}/feature_columns.pkl", 'rb') as f:
                    if not hasattr(self, 'feature_cols'):
                        self.feature_cols = {}
                    self.feature_cols[model_key] = pickle.load(f)
            
            # Initialize model dictionary
            self.models[model_key] = {}
            
            # Load all models in the directory
            for target_file in os.listdir(f"models/{model_key.replace('/', '_')}"):
                if target_file.endswith("_xgboost.pkl"):
                    target = target_file.replace("_xgboost.pkl", "").replace('_', '/')
                    
                    # Initialize target in models dict
                    if target not in self.models[model_key]:
                        self.models[model_key][target] = {}
                    
                    # Load XGBoost
                    with open(f"models/{model_key.replace('/', '_')}/{target_file}", 'rb') as f:
                        xgb_model = pickle.load(f)
                        is_regression = not hasattr(xgb_model, 'classes_')
                        self.models[model_key][target]['xgboost'] = {
                            'model': xgb_model,
                            'is_regression': is_regression
                        }
                    
                    # Load CatBoost
                    with open(f"models/{model_key.replace('/', '_')}/{target.replace('/', '_')}_catboost.pkl", 'rb') as f:
                        cat_model = pickle.load(f)
                        self.models[model_key][target]['catboost'] = {
                            'model': cat_model,
                            'is_regression': is_regression
                        }
                    
                    # Load LSTM
                    lstm_model = load_model(f"models/{model_key.replace('/', '_')}/{target.replace('/', '_')}_lstm")
                    self.models[model_key][target]['lstm'] = {
                        'model': lstm_model,
                        'is_regression': is_regression
                    }
            
            logger.info(f"Models loaded for {model_key}")
            return True
        except Exception as e:
            logger.error(f"Error loading models for {model_key}: {e}")
            return False
        
    def get_predictions(self, symbol: str, timeframe: str, df: pd.DataFrame) -> dict:
        """Get predictions from models for a symbol and timeframe."""
        model_key = f"{symbol}_{timeframe}"
        
        try:
            # Check if models are loaded
            if model_key not in self.models:
                loaded = self._load_models(model_key)
                if not loaded:
                    logger.warning(f"No models available for {model_key}")
                    return {}
            
            # Prepare features for prediction
            df_features = self.features.prepare_features(df, symbol, is_training=False)
            
            if df_features.empty:
                logger.warning(f"No features available for prediction for {symbol}")
                return {}
            
            # Get the latest data point for prediction
            latest_features = df_features.iloc[-1:].copy()
            
            # Make sure we're using the same feature columns that were used in training
            if hasattr(self, 'feature_cols') and model_key in self.feature_cols:
                training_features = self.feature_cols[model_key]
                
                # Check which features are available in the current data
                available_features = [col for col in training_features if col in latest_features.columns]
                
                if len(available_features) < len(training_features):
                    logger.warning(f"Some training features are missing from prediction data: "
                                f"{len(training_features) - len(available_features)} missing features")
                    
                    # Fill missing features with zeros
                    for col in training_features:
                        if col not in latest_features.columns:
                            latest_features[col] = 0
                
                # Use only the training features and in the same order
                X = latest_features[training_features]
            else:
                # If we don't have saved feature columns, just use all non-target columns
                feature_cols = [col for col in latest_features.columns 
                            if not col.startswith('target_') and not col.startswith('future_')]
                X = latest_features[feature_cols]
            
            # Get predictions for each target and model
            predictions = {}
            
            for target, models in self.models[model_key].items():
                target_predictions = {}
                
                # XGBoost prediction
                try:
                    xgb_model = models['xgboost']['model']
                    is_regression = models['xgboost']['is_regression']
                    
                    if is_regression:
                        xgb_pred = xgb_model.predict(X)[0]
                    else:
                        xgb_proba = xgb_model.predict_proba(X)[0]
                        if len(xgb_proba) == 2:  # Binary classification
                            xgb_pred = xgb_proba[1]  # Probability of positive class
                        else:  # Multi-class
                            xgb_pred = xgb_model.predict(X)[0]
                    
                    target_predictions['xgboost'] = {
                        'prediction': float(xgb_pred),
                        'weight': self.model_weights['xgboost']
                    }
                except Exception as e:
                    logger.error(f"Error with XGBoost prediction for {target}: {e}")
                
                # CatBoost prediction
                try:
                    cat_model = models['catboost']['model']
                    is_regression = models['catboost']['is_regression']
                    
                    if is_regression:
                        cat_pred = cat_model.predict(X)[0]
                    else:
                        cat_proba = cat_model.predict_proba(X)[0]
                        if len(cat_proba) == 2:  # Binary classification
                            cat_pred = cat_proba[1]  # Probability of positive class
                        else:  # Multi-class
                            cat_pred = cat_model.predict(X)[0]
                    
                    target_predictions['catboost'] = {
                        'prediction': float(cat_pred),
                        'weight': self.model_weights['catboost']
                    }
                except Exception as e:
                    logger.error(f"Error with CatBoost prediction for {target}: {e}")
                
                # LSTM prediction
                try:
                    lstm_model = models['lstm']['model']
                    is_regression = models['lstm']['is_regression']
                    
                    # Reshape for LSTM [samples, timesteps, features]
                    # Make sure we have the right number of features for the LSTM model
                    if hasattr(self, 'feature_cols') and model_key in self.feature_cols:
                        # Get the expected input shape from the model
                        input_shape = lstm_model.layers[0].input_shape
                        expected_features = input_shape[2] if len(input_shape) >= 3 else X.shape[1]
                        
                        # For LSTM we need to make sure the feature dimensionality exactly matches
                        X_lstm = X.values  # Use exactly the same features as saved during training
                    else:
                        X_lstm = X.values
                    
                    X_reshaped = X_lstm.reshape((X_lstm.shape[0], 1, X_lstm.shape[1]))
                    
                    lstm_pred = lstm_model.predict(X_reshaped)[0][0]
                    
                    target_predictions['lstm'] = {
                        'prediction': float(lstm_pred),
                        'weight': self.model_weights['lstm']
                    }
                except Exception as e:
                    logger.error(f"Error with LSTM prediction for {target}: {e}")
                
                # Store all model predictions for this target
                predictions[target] = target_predictions
            
            return predictions
        except Exception as e:
            logger.error(f"Error generating predictions for {symbol} on {timeframe}: {e}")
            return {}
    def get_trading_signals(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, float]:
        """Get trading signals from model predictions."""
        try:
            # Get raw predictions
            predictions = self.get_predictions(symbol, timeframe, df)
            
            if not predictions:
                return {"signal": 0, "confidence": 0}
            
            # Combine predictions into trading signals
            signals = {}
            
            # Process each prediction target
            for target, target_preds in predictions.items():
                # Skip if no predictions
                if not target_preds:
                    continue
                
                # Calculate weighted ensemble prediction
                total_weight = sum(pred['weight'] for pred in target_preds.values())
                if total_weight == 0:
                    continue
                
                weighted_pred = sum(pred['prediction'] * pred['weight'] for pred in target_preds.values()) / total_weight
                
                # Convert to signal based on prediction type
                if "target_direction" in target:
                    # For directional prediction (binary classification)
                    # Signal is between -1 and 1 based on confidence
                    signal = (weighted_pred - 0.5) * 2
                    confidence = abs(weighted_pred - 0.5) * 2
                elif "target_ternary" in target:
                    # For ternary prediction (3-class classification)
                    # Already returns -1, 0, or 1
                    signal = weighted_pred
                    confidence = 1.0  # Confidence is embedded in the signal
                else:
                    # For regression prediction (future returns)
                    # Scale returns to a signal between -1 and 1
                    # Typical daily returns range from -0.05 to 0.05 (5%)
                    signal = max(-1, min(1, weighted_pred * 20))
                    confidence = min(1.0, abs(weighted_pred * 20))
                
                # Extract time horizon from target name
                horizon_match = target.split('_')[-1]
                if horizon_match.isdigit():
                    horizon = int(horizon_match)
                    signals[f"horizon_{horizon}"] = {
                        "signal": signal,
                        "confidence": confidence
                    }
            
            # If we have signals for multiple horizons, combine them
            # with more weight to shorter horizons
            if not signals:
                return {"signal": 0, "confidence": 0}
            
            # Calculate overall signal with weighted horizons
            combined_signal = 0
            combined_confidence = 0
            total_horizon_weight = 0
            
            for horizon_key, signal_dict in signals.items():
                horizon = int(horizon_key.split('_')[1])
                # Weight is inversely proportional to horizon
                horizon_weight = 1 / horizon
                
                combined_signal += signal_dict["signal"] * horizon_weight
                combined_confidence += signal_dict["confidence"] * horizon_weight
                total_horizon_weight += horizon_weight
            
            if total_horizon_weight > 0:
                combined_signal /= total_horizon_weight
                combined_confidence /= total_horizon_weight
            
            # Only return confident signals
            if combined_confidence < self.confidence_threshold:
                combined_signal *= (combined_confidence / self.confidence_threshold)
            
            return {
                "signal": combined_signal,
                "confidence": combined_confidence,
                "horizon_signals": signals
            }
        except Exception as e:
            logger.error(f"Error generating trading signals for {symbol} on {timeframe}: {e}")
            return {"signal": 0, "confidence": 0}


class MarketAnalysis:
    """Advanced market analysis including regime detection."""
    
    def __init__(self, config):
        """Initialize market analysis with configuration."""
        self.config = config
        self.regime_history = {}  # Store history of market regimes
    
    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect the current market regime based on comprehensive price action analysis."""
        try:
            # Need enough data for analysis
            if len(df) < 100:
                logger.warning(f"Insufficient data for market regime detection: {len(df)} rows < 100 required")
                return MarketRegime.UNKNOWN
            
            # Extract relevant price series
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # Calculate various moving averages
            sma10 = close.rolling(window=10).mean()
            sma20 = close.rolling(window=20).mean()
            sma50 = close.rolling(window=50).mean()
            sma100 = close.rolling(window=100).mean()
            
            # Calculate EMAs for trend detection
            ema8 = close.ewm(span=8, adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()
            ema34 = close.ewm(span=34, adjust=False).mean()
            ema55 = close.ewm(span=55, adjust=False).mean()
            
            # ATR for volatility
            atr = ta.volatility.average_true_range(high, low, close, window=14)
            atr_pct = atr / close * 100  # ATR as percentage of price
            
            # Calculate MACD for trend strength and transitions
            macd = ta.trend.MACD(close)
            macd_line = macd.macd()
            signal_line = macd.macd_signal()
            macd_hist = macd.macd_diff()
            
            # RSI for overbought/oversold
            rsi = ta.momentum.RSIIndicator(close).rsi()
            
            # ADX for trend strength
            adx_indicator = ta.trend.ADXIndicator(high, low, close)
            adx = adx_indicator.adx()
            plus_di = adx_indicator.adx_pos()
            minus_di = adx_indicator.adx_neg()
            
            # Bollinger Bands for volatility and mean reversion
            bollinger = ta.volatility.BollingerBands(close)
            bb_upper = bollinger.bollinger_hband()
            bb_middle = bollinger.bollinger_mavg()
            bb_lower = bollinger.bollinger_lband()
            bb_width = (bb_upper - bb_lower) / bb_middle
            
            # Ichimoku Cloud for trend and support/resistance
            ichimoku = ta.trend.IchimokuIndicator(high, low)
            tenkan = ichimoku.ichimoku_conversion_line()
            kijun = ichimoku.ichimoku_base_line()
            senkou_a = ichimoku.ichimoku_a()
            senkou_b = ichimoku.ichimoku_b()
            
            # Recent values (last 5 candles)
            recent_close = close.iloc[-1]
            recent_adx = adx.iloc[-1]
            recent_plus_di = plus_di.iloc[-1]
            recent_minus_di = minus_di.iloc[-1]
            recent_rsi = rsi.iloc[-1]
            recent_bb_width = bb_width.iloc[-1]
            recent_atr_pct = atr_pct.iloc[-1]
            
            # Historical volatility using 20-day window
            returns = close.pct_change()
            hist_vol = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
            recent_hist_vol = hist_vol.iloc[-1]
            
            # Volume analysis
            vol_sma20 = volume.rolling(window=20).mean()
            recent_vol_ratio = volume.iloc[-1] / vol_sma20.iloc[-1]
            
            # Check for price directionality
            recent_candles = 20
            up_candles = sum(1 for i in range(-1, -recent_candles, -1) if i >= -len(close) and close.iloc[i] > close.iloc[i-1])
            down_candles = recent_candles - up_candles
            directional_strength = abs(up_candles - down_candles) / recent_candles
            
            # Compute trend score (-100 to +100) combining multiple indicators
            trend_score = 0
            
            # Moving average alignment (weighted by recency)
            if ema8.iloc[-1] > ema21.iloc[-1] > ema34.iloc[-1]:
                trend_score += 20
            elif ema8.iloc[-1] < ema21.iloc[-1] < ema34.iloc[-1]:
                trend_score -= 20
                
            # MACD signal
            if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_hist.iloc[-1] > 0:
                trend_score += 15
            elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_hist.iloc[-1] < 0:
                trend_score -= 15
                
            # ADX trend strength with direction
            if recent_adx > 25:
                if recent_plus_di > recent_minus_di:
                    trend_score += min(25, recent_adx / 2)
                else:
                    trend_score -= min(25, recent_adx / 2)
                    
            # Price relation to Ichimoku Cloud
            if recent_close > senkou_a.iloc[-1] and recent_close > senkou_b.iloc[-1]:
                trend_score += 15
            elif recent_close < senkou_a.iloc[-1] and recent_close < senkou_b.iloc[-1]:
                trend_score -= 15
                
            # RSI trend bias
            if recent_rsi > 60:
                trend_score += 10
            elif recent_rsi < 40:
                trend_score -= 10
                
            # Recent directional movement
            if up_candles > down_candles:
                trend_score += (directional_strength * 15)
            else:
                trend_score -= (directional_strength * 15)
                
            # Compute volatility score (0 to 100)
            volatility_score = 0
            
            # ATR percentage
            if recent_atr_pct > 5:  # Very high volatility
                volatility_score += 40
            elif recent_atr_pct > 3:  # High volatility
                volatility_score += 30
            elif recent_atr_pct > 2:  # Moderate volatility
                volatility_score += 20
            elif recent_atr_pct > 1:  # Low volatility
                volatility_score += 10
                
            # Bollinger Band width
            vol_percentile = pd.Series(bb_width).rank(pct=True).iloc[-1] * 100
            volatility_score += min(30, vol_percentile / 3.33)  # Max 30 points
            
            # Historical volatility percentile
            hist_vol_series = pd.Series(hist_vol.dropna())
            if not hist_vol_series.empty:
                hist_vol_percentile = hist_vol_series.rank(pct=True).iloc[-1] * 100
                volatility_score += min(20, hist_vol_percentile / 5)  # Max 20 points
                
            # Volume volatility
            volume_std = volume.rolling(window=20).std() / volume.rolling(window=20).mean()
            if not volume_std.empty and not np.isnan(volume_std.iloc[-1]):
                volume_vol = volume_std.iloc[-1]
                if volume_vol > 1.5:  # Very volatile volume
                    volatility_score += 10
                elif volume_vol > 1.0:  # Moderately volatile volume
                    volatility_score += 5
                    
            # Calculate range-bound indication
            range_bound_score = 0
            
            # Price movement relative to ATR
            price_range_20d = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100
            if abs(price_range_20d) < recent_atr_pct * 5:  # Price hasn't moved much relative to ATR
                range_bound_score += 30
                
            # RSI in middle range
            if 40 <= recent_rsi <= 60:
                range_bound_score += 20
                
            # Low directional strength
            if directional_strength < 0.3:
                range_bound_score += 20
                
            # Low ADX (weak trend)
            if recent_adx < 20:
                range_bound_score += 30
                
            # Check for consolidation pattern
            consolidation_score = 0
            
            # Narrowing Bollinger Bands
            bb_width_change = bb_width.iloc[-1] / bb_width.iloc[-10] - 1
            if bb_width_change < -0.2:  # BB width decreasing by 20% or more
                consolidation_score += 30
                
            # Decreasing ATR
            atr_change = atr.iloc[-1] / atr.iloc[-10] - 1
            if atr_change < -0.15:  # ATR decreasing by 15% or more
                consolidation_score += 30
                
            # Decreasing volume
            vol_change = volume.iloc[-5:].mean() / volume.iloc[-15:-5].mean() - 1
            if vol_change < -0.1:  # Volume decreasing by 10% or more
                consolidation_score += 20
                
            # Low price volatility recently
            recent_true_ranges = [max(high.iloc[i] - low.iloc[i], 
                                    abs(high.iloc[i] - close.iloc[i-1]),
                                    abs(low.iloc[i] - close.iloc[i-1])) / close.iloc[i] * 100
                                for i in range(-5, 0)]
            if np.mean(recent_true_ranges) < recent_atr_pct * 0.7:  # Recent TR below 70% of ATR
                consolidation_score += 20
                
            # Determine market regime based on scores
            logger.debug(f"Market regime scores - Trend: {trend_score:.1f}, Volatility: {volatility_score:.1f}, "
                        f"Range: {range_bound_score:.1f}, Consolidation: {consolidation_score:.1f}")
            
            # Strong trend detection
            if abs(trend_score) > 60:
                if trend_score > 0:
                    logger.info(f"Detected TRENDING_UP regime (score: {trend_score:.1f})")
                    return MarketRegime.TRENDING_UP
                else:
                    logger.info(f"Detected TRENDING_DOWN regime (score: {trend_score:.1f})")
                    return MarketRegime.TRENDING_DOWN
                    
            # High volatility detection
            if volatility_score > 60:
                logger.info(f"Detected VOLATILE regime (score: {volatility_score:.1f})")
                return MarketRegime.VOLATILE
                
            # Consolidation detection
            if consolidation_score > 70:
                logger.info(f"Detected CONSOLIDATING regime (score: {consolidation_score:.1f})")
                return MarketRegime.CONSOLIDATING
                
            # Range-bound detection
            if range_bound_score > 60 and abs(trend_score) < 40:
                logger.info(f"Detected RANGING regime (score: {range_bound_score:.1f})")
                return MarketRegime.RANGING
                
            # Moderate trend
            if abs(trend_score) > 40:
                if trend_score > 0:
                    logger.info(f"Detected moderate TRENDING_UP regime (score: {trend_score:.1f})")
                    return MarketRegime.TRENDING_UP
                else:
                    logger.info(f"Detected moderate TRENDING_DOWN regime (score: {trend_score:.1f})")
                    return MarketRegime.TRENDING_DOWN
                    
            # Default to ranging if no strong signals
            logger.info("Detected default RANGING regime (no strong signals)")
            return MarketRegime.RANGING
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.UNKNOWN
    def update_regime_history(self, symbol: str, timeframe: str, regime: MarketRegime):
        """Update the market regime history."""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.regime_history:
            self.regime_history[key] = []
        
        # Add the current regime with timestamp
        self.regime_history[key].append({
            "regime": regime,
            "timestamp": datetime.now()
        })
        
        # Keep only the last 100
        if len(self.regime_history[key]) > 100:
            self.regime_history[key] = self.regime_history[key][-100:]
    
    def get_regime_adjusted_signal(self, symbol: str, timeframe: str, base_signal: float, 
                                   regime: MarketRegime) -> float:
        """Adjust a trading signal based on the current market regime."""
        # No adjustment for unknown regime
        if regime == MarketRegime.UNKNOWN:
            return base_signal
        
        # Amplify signals in trending markets
        if regime == MarketRegime.TRENDING_UP:
            if base_signal > 0:
                return min(1.0, base_signal * 1.5)  # Amplify positive signals
            else:
                return base_signal * 0.5  # Reduce negative signals
        
        if regime == MarketRegime.TRENDING_DOWN:
            if base_signal < 0:
                return max(-1.0, base_signal * 1.5)  # Amplify negative signals
            else:
                return base_signal * 0.5  # Reduce positive signals
        
        # Reduce signal strength in ranging markets
        if regime == MarketRegime.RANGING:
            return base_signal * 0.7
        
        # Be cautious in volatile markets
        if regime == MarketRegime.VOLATILE:
            return base_signal * 0.5
        
        # Consolidating markets often precede breakouts, maintain signal
        if regime == MarketRegime.CONSOLIDATING:
            return base_signal
        
        return base_signal
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """Calculate dynamic support and resistance levels."""
        try:
            if len(df) < window:
                return None, None
            
            # Use recent price data
            recent_df = df.iloc[-window:]
            
            # Method 1: Using pivots
            pivot_high = recent_df['high'].max()
            pivot_low = recent_df['low'].min()
            
            # Method 2: Using price clusters
            # Bin prices into clusters to find areas of price congestion
            price_range = recent_df['high'].max() - recent_df['low'].min()
            bin_size = price_range / 10
            
            # Create bins for high and low prices
            high_bins = np.floor(recent_df['high'] / bin_size) * bin_size
            low_bins = np.floor(recent_df['low'] / bin_size) * bin_size
            
            # Find the most common bins (price levels with most touches)
            high_bin_counts = high_bins.value_counts().sort_index()
            low_bin_counts = low_bins.value_counts().sort_index()
            
            # Get the bins with the most counts
            if not high_bin_counts.empty:
                resistance_bin = high_bin_counts.idxmax()
                resistance = resistance_bin + bin_size  # Top of the bin
            else:
                resistance = pivot_high
            
            if not low_bin_counts.empty:
                support_bin = low_bin_counts.idxmax()
                support = support_bin  # Bottom of the bin
            else:
                support = pivot_low
            
            # Combine methods (weight based on recency)
            recent_close = df['close'].iloc[-1]
            
            # If price is closer to pivot high, use pivot high as resistance
            if abs(recent_close - pivot_high) < abs(recent_close - resistance):
                final_resistance = pivot_high
            else:
                final_resistance = resistance
            
            # If price is closer to pivot low, use pivot low as support
            if abs(recent_close - pivot_low) < abs(recent_close - support):
                final_support = pivot_low
            else:
                final_support = support
            
            return final_support, final_resistance
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return None, None
    
    def identify_patterns(self, df: pd.DataFrame) -> List[str]:
        """Identify chart patterns in the price data using robust detection algorithms."""
        patterns = []
        
        try:
            if len(df) < 50:
                logger.warning(f"Insufficient data for pattern detection: {len(df)} rows < 50 required")
                return patterns
            
            # Get recent data (last 100 candles)
            recent_df = df.iloc[-100:].copy()
            
            # Calculate basic indicators
            recent_df['sma20'] = recent_df['close'].rolling(window=20).mean()
            recent_df['sma50'] = recent_df['close'].rolling(window=50).mean()
            recent_df['atr'] = ta.volatility.average_true_range(recent_df['high'], recent_df['low'], recent_df['close'], window=14)
            
            # Calculate pivot points for pattern detection
            high_pivots = self._find_pivot_highs(recent_df)
            low_pivots = self._find_pivot_lows(recent_df)
            
            # 1. Detect head and shoulders pattern
            head_shoulders = self._detect_head_and_shoulders(recent_df, high_pivots)
            if head_shoulders:
                patterns.append("head_and_shoulders")
            
            # 2. Detect inverse head and shoulders pattern
            inv_head_shoulders = self._detect_inverse_head_and_shoulders(recent_df, low_pivots)
            if inv_head_shoulders:
                patterns.append("inverse_head_and_shoulders")
            
            # 3. Detect double top pattern
            double_top = self._detect_double_top(recent_df, high_pivots)
            if double_top:
                patterns.append("double_top")
            
            # 4. Detect double bottom pattern
            double_bottom = self._detect_double_bottom(recent_df, low_pivots)
            if double_bottom:
                patterns.append("double_bottom")
            
            # 5. Detect triple top pattern
            triple_top = self._detect_triple_top(recent_df, high_pivots)
            if triple_top:
                patterns.append("triple_top")
            
            # 6. Detect triple bottom pattern
            triple_bottom = self._detect_triple_bottom(recent_df, low_pivots)
            if triple_bottom:
                patterns.append("triple_bottom")
            
            # 7. Detect ascending triangle
            asc_triangle = self._detect_ascending_triangle(recent_df, high_pivots, low_pivots)
            if asc_triangle:
                patterns.append("ascending_triangle")
            
            # 8. Detect descending triangle
            desc_triangle = self._detect_descending_triangle(recent_df, high_pivots, low_pivots)
            if desc_triangle:
                patterns.append("descending_triangle")
            
            # 9. Detect symmetrical triangle
            sym_triangle = self._detect_symmetrical_triangle(recent_df, high_pivots, low_pivots)
            if sym_triangle:
                patterns.append("symmetrical_triangle")
            
            # 10. Detect bull flag pattern
            bull_flag = self._detect_bull_flag(recent_df)
            if bull_flag:
                patterns.append("bull_flag")
            
            # 11. Detect bear flag pattern
            bear_flag = self._detect_bear_flag(recent_df)
            if bear_flag:
                patterns.append("bear_flag")
            
            # 12. Detect engulfing patterns (bullish and bearish)
            for i in range(1, len(recent_df)):
                # Get current and previous candle data
                prev_open, prev_close = recent_df['open'].iloc[i-1], recent_df['close'].iloc[i-1]
                curr_open, curr_close = recent_df['open'].iloc[i], recent_df['close'].iloc[i]
                
                # Detect bullish engulfing
                if (prev_close < prev_open and  # Previous candle is bearish (red)
                    curr_close > curr_open and  # Current candle is bullish (green)
                    curr_open <= prev_close and  # Current open is below or equal to previous close
                    curr_close >= prev_open):    # Current close is above or equal to previous open
                    
                    # Only detect if it's the most recent candle
                    if i == len(recent_df) - 1:
                        patterns.append("bullish_engulfing")
                
                # Detect bearish engulfing
                if (prev_close > prev_open and  # Previous candle is bullish (green)
                    curr_close < curr_open and  # Current candle is bearish (red)
                    curr_open >= prev_close and  # Current open is above or equal to previous close
                    curr_close <= prev_open):    # Current close is below or equal to previous open
                    
                    # Only detect if it's the most recent candle
                    if i == len(recent_df) - 1:
                        patterns.append("bearish_engulfing")
            
            # Log detected patterns
            if patterns:
                logger.info(f"Detected chart patterns: {patterns}")
            
            return patterns
        
        except Exception as e:
            logger.error(f"Error identifying chart patterns: {e}")
            return []

    def _find_pivot_highs(self, df: pd.DataFrame, window: int = 5) -> List[int]:
        """Find pivot high points in the data with optimized detection."""
        pivot_highs = []
        
        # Need at least 2*window+1 candles
        if len(df) < 2 * window + 1:
            return pivot_highs
        
        # Get high prices array for faster operations
        highs = df['high'].values
        
        # Find pivot highs
        for i in range(window, len(df) - window):
            # Get window around current point
            left_window = highs[i-window:i]
            right_window = highs[i+1:i+window+1]
            current = highs[i]
            
            # Check if current point is a pivot high
            if current > np.max(left_window) and current > np.max(right_window):
                pivot_highs.append(i)
        
        return pivot_highs

    def _find_pivot_lows(self, df: pd.DataFrame, window: int = 5) -> List[int]:
        """Find pivot low points in the data with optimized detection."""
        pivot_lows = []
        
        # Need at least 2*window+1 candles
        if len(df) < 2 * window + 1:
            return pivot_lows
        
        # Get low prices array for faster operations
        lows = df['low'].values
        
        # Find pivot lows
        for i in range(window, len(df) - window):
            # Get window around current point
            left_window = lows[i-window:i]
            right_window = lows[i+1:i+window+1]
            current = lows[i]
            
            # Check if current point is a pivot low
            if current < np.min(left_window) and current < np.min(right_window):
                pivot_lows.append(i)
        
        return pivot_lows

    def _detect_head_and_shoulders(self, df: pd.DataFrame, pivot_highs: List[int]) -> bool:
        """Detect head and shoulders pattern using pivot points."""
        if len(pivot_highs) < 3:
            return False
        
        # Need at least 3 recent pivot highs
        recent_pivots = pivot_highs[-3:]
        if len(recent_pivots) < 3:
            return False
        
        # Get pivot high values
        try:
            left_shoulder = df['high'].iloc[recent_pivots[0]]
            head = df['high'].iloc[recent_pivots[1]]
            right_shoulder = df['high'].iloc[recent_pivots[2]]
            
            # Check if middle pivot is higher (the head)
            if head > left_shoulder and head > right_shoulder:
                # Check if shoulders are at similar levels (within 3% of each other)
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                if shoulder_diff < 0.03:
                    # Check if pattern is complete (right shoulder formed)
                    if recent_pivots[2] >= len(df) - 5:
                        return True
        except Exception as e:
            logger.debug(f"Error detecting head and shoulders: {e}")
        
        return False

    def _detect_inverse_head_and_shoulders(self, df: pd.DataFrame, pivot_lows: List[int]) -> bool:
        """Detect inverse head and shoulders pattern using pivot points."""
        if len(pivot_lows) < 3:
            return False
        
        # Need at least 3 recent pivot lows
        recent_pivots = pivot_lows[-3:]
        if len(recent_pivots) < 3:
            return False
        
        # Get pivot low values
        try:
            left_shoulder = df['low'].iloc[recent_pivots[0]]
            head = df['low'].iloc[recent_pivots[1]]
            right_shoulder = df['low'].iloc[recent_pivots[2]]
            
            # Check if middle pivot is lower (the head)
            if head < left_shoulder and head < right_shoulder:
                # Check if shoulders are at similar levels (within 3% of each other)
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                if shoulder_diff < 0.03:
                    # Check if pattern is complete (right shoulder formed)
                    if recent_pivots[2] >= len(df) - 5:
                        return True
        except Exception as e:
            logger.debug(f"Error detecting inverse head and shoulders: {e}")
        
        return False

    def _detect_double_top(self, df: pd.DataFrame, pivot_highs: List[int]) -> bool:
        """Detect double top pattern using pivot points."""
        if len(pivot_highs) < 2:
            return False
        
        # Need at least 2 recent pivot highs
        recent_pivots = pivot_highs[-2:]
        if len(recent_pivots) < 2:
            return False
        
        # Get pivot high values
        try:
            first_top = df['high'].iloc[recent_pivots[0]]
            second_top = df['high'].iloc[recent_pivots[1]]
            
            # Check if tops are at similar levels (within 2% of each other)
            top_diff = abs(first_top - second_top) / first_top
            if top_diff < 0.02:
                # Check time distance (should be at least 10 candles apart)
                time_distance = recent_pivots[1] - recent_pivots[0]
                if time_distance >= 10 and time_distance <= 50:
                    # Check if pattern is complete (second top formed recently)
                    if recent_pivots[1] >= len(df) - 5:
                        return True
        except Exception as e:
            logger.debug(f"Error detecting double top: {e}")
        
        return False

    def _detect_double_bottom(self, df: pd.DataFrame, pivot_lows: List[int]) -> bool:
        """Detect double bottom pattern using pivot points."""
        if len(pivot_lows) < 2:
            return False
        
        # Need at least 2 recent pivot lows
        recent_pivots = pivot_lows[-2:]
        if len(recent_pivots) < 2:
            return False
        
        # Get pivot low values
        try:
            first_bottom = df['low'].iloc[recent_pivots[0]]
            second_bottom = df['low'].iloc[recent_pivots[1]]
            
            # Check if bottoms are at similar levels (within 2% of each other)
            bottom_diff = abs(first_bottom - second_bottom) / first_bottom
            if bottom_diff < 0.02:
                # Check time distance (should be at least 10 candles apart)
                time_distance = recent_pivots[1] - recent_pivots[0]
                if time_distance >= 10 and time_distance <= 50:
                    # Check if pattern is complete (second bottom formed recently)
                    if recent_pivots[1] >= len(df) - 5:
                        return True
        except Exception as e:
            logger.debug(f"Error detecting double bottom: {e}")
        
        return False

    def _detect_triple_top(self, df: pd.DataFrame, pivot_highs: List[int]) -> bool:
        """Detect triple top pattern using pivot points."""
        if len(pivot_highs) < 3:
            return False
        
        # Need at least 3 recent pivot highs
        recent_pivots = pivot_highs[-3:]
        if len(recent_pivots) < 3:
            return False
        
        # Get pivot high values
        try:
            first_top = df['high'].iloc[recent_pivots[0]]
            second_top = df['high'].iloc[recent_pivots[1]]
            third_top = df['high'].iloc[recent_pivots[2]]
            
            # Check if all tops are at similar levels (within 2% of average)
            avg_top = (first_top + second_top + third_top) / 3
            if (abs(first_top - avg_top) / avg_top < 0.02 and
                abs(second_top - avg_top) / avg_top < 0.02 and
                abs(third_top - avg_top) / avg_top < 0.02):
                
                # Check time distances (should be reasonably spaced)
                if (recent_pivots[1] - recent_pivots[0] >= 5 and
                    recent_pivots[2] - recent_pivots[1] >= 5 and
                    recent_pivots[2] - recent_pivots[0] <= 60):
                    
                    # Check if pattern is complete (third top formed recently)
                    if recent_pivots[2] >= len(df) - 5:
                        return True
        except Exception as e:
            logger.debug(f"Error detecting triple top: {e}")
        
        return False

    def _detect_triple_bottom(self, df: pd.DataFrame, pivot_lows: List[int]) -> bool:
        """Detect triple bottom pattern using pivot points."""
        if len(pivot_lows) < 3:
            return False
        
        # Need at least 3 recent pivot lows
        recent_pivots = pivot_lows[-3:]
        if len(recent_pivots) < 3:
            return False
        
        # Get pivot low values
        try:
            first_bottom = df['low'].iloc[recent_pivots[0]]
            second_bottom = df['low'].iloc[recent_pivots[1]]
            third_bottom = df['low'].iloc[recent_pivots[2]]
            
            # Check if all bottoms are at similar levels (within 2% of average)
            avg_bottom = (first_bottom + second_bottom + third_bottom) / 3
            if (abs(first_bottom - avg_bottom) / avg_bottom < 0.02 and
                abs(second_bottom - avg_bottom) / avg_bottom < 0.02 and
                abs(third_bottom - avg_bottom) / avg_bottom < 0.02):
                
                # Check time distances (should be reasonably spaced)
                if (recent_pivots[1] - recent_pivots[0] >= 5 and
                    recent_pivots[2] - recent_pivots[1] >= 5 and
                    recent_pivots[2] - recent_pivots[0] <= 60):
                    
                    # Check if pattern is complete (third bottom formed recently)
                    if recent_pivots[2] >= len(df) - 5:
                        return True
        except Exception as e:
            logger.debug(f"Error detecting triple bottom: {e}")
        
        return False

    def _detect_ascending_triangle(self, df: pd.DataFrame, pivot_highs: List[int], pivot_lows: List[int]) -> bool:
        """Detect ascending triangle pattern."""
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return False
        
        try:
            # Get recent pivots
            recent_high_pivots = pivot_highs[-3:]
            recent_low_pivots = pivot_lows[-3:]
            
            if len(recent_high_pivots) < 2 or len(recent_low_pivots) < 2:
                return False
            
            # Check for horizontal resistance (similar highs)
            high_values = [df['high'].iloc[i] for i in recent_high_pivots]
            avg_high = sum(high_values) / len(high_values)
            
            if all(abs(h - avg_high) / avg_high < 0.02 for h in high_values):
                # Check for rising support (increasing lows)
                low_values = [df['low'].iloc[i] for i in recent_low_pivots]
                
                # Must have rising lows
                if low_values[-1] > low_values[0]:
                    # Pattern should narrow (highs and lows converging)
                    if (recent_high_pivots[-1] - recent_low_pivots[-1]) < (recent_high_pivots[0] - recent_low_pivots[0]):
                        return True
        except Exception as e:
            logger.debug(f"Error detecting ascending triangle: {e}")
        
        return False

    def _detect_descending_triangle(self, df: pd.DataFrame, pivot_highs: List[int], pivot_lows: List[int]) -> bool:
        """Detect descending triangle pattern."""
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return False
        
        try:
            # Get recent pivots
            recent_high_pivots = pivot_highs[-3:]
            recent_low_pivots = pivot_lows[-3:]
            
            if len(recent_high_pivots) < 2 or len(recent_low_pivots) < 2:
                return False
            
            # Check for horizontal support (similar lows)
            low_values = [df['low'].iloc[i] for i in recent_low_pivots]
            avg_low = sum(low_values) / len(low_values)
            
            if all(abs(l - avg_low) / avg_low < 0.02 for l in low_values):
                # Check for declining resistance (decreasing highs)
                high_values = [df['high'].iloc[i] for i in recent_high_pivots]
                
                # Must have declining highs
                if high_values[-1] < high_values[0]:
                    # Pattern should narrow (highs and lows converging)
                    if (recent_high_pivots[-1] - recent_low_pivots[-1]) < (recent_high_pivots[0] - recent_low_pivots[0]):
                        return True
        except Exception as e:
            logger.debug(f"Error detecting descending triangle: {e}")
        
        return False

    def _detect_symmetrical_triangle(self, df: pd.DataFrame, pivot_highs: List[int], pivot_lows: List[int]) -> bool:
        """Detect symmetrical triangle pattern."""
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return False
        
        try:
            # Get recent pivots
            recent_high_pivots = pivot_highs[-3:]
            recent_low_pivots = pivot_lows[-3:]
            
            if len(recent_high_pivots) < 2 or len(recent_low_pivots) < 2:
                return False
            
            # Check for declining highs
            high_values = [df['high'].iloc[i] for i in recent_high_pivots]
            if high_values[-1] < high_values[0]:
                # Check for rising lows
                low_values = [df['low'].iloc[i] for i in recent_low_pivots]
                
                if low_values[-1] > low_values[0]:
                    # Pattern should narrow (highs and lows converging)
                    initial_range = high_values[0] - low_values[0]
                    current_range = high_values[-1] - low_values[-1]
                    
                    if current_range < initial_range * 0.7:  # At least 30% narrower
                        return True
        except Exception as e:
            logger.debug(f"Error detecting symmetrical triangle: {e}")
        
        return False

    def _detect_bull_flag(self, df: pd.DataFrame) -> bool:
        """Detect bull flag pattern."""
        try:
            # Need enough data
            if len(df) < 20:
                return False
            
            # Check for strong uptrend before flag
            lookback = 20  # Look at last 20 candles
            
            # Split into pole and flag
            pole_period = 10
            flag_period = 10
            
            # Get pole data (earlier period)
            pole_data = df.iloc[-lookback:-flag_period]
            
            # Get flag data (recent period)
            flag_data = df.iloc[-flag_period:]
            
            # Pole should be a strong uptrend
            pole_start_price = pole_data['close'].iloc[0]
            pole_end_price = pole_data['close'].iloc[-1]
            pole_gain = (pole_end_price / pole_start_price - 1) * 100
            
            # Flag should be a shallow downtrend or consolidation
            flag_start_price = flag_data['close'].iloc[0]
            flag_end_price = flag_data['close'].iloc[-1]
            flag_change = (flag_end_price / flag_start_price - 1) * 100
            
            # Check criteria
            if (pole_gain > 10 and  # Strong uptrend in pole (at least 10%)
                flag_change < 0 and  # Flag should be down or flat
                flag_change > -pole_gain * 0.5):  # Flag retracement less than 50% of pole gain
                
                # Check flag volume (should be lower than pole)
                pole_avg_volume = pole_data['volume'].mean()
                flag_avg_volume = flag_data['volume'].mean()
                
                if flag_avg_volume < pole_avg_volume:
                    return True
        except Exception as e:
            logger.debug(f"Error detecting bull flag: {e}")
        
        return False

    def _detect_bear_flag(self, df: pd.DataFrame) -> bool:
        """Detect bear flag pattern."""
        try:
            # Need enough data
            if len(df) < 20:
                return False
            
            # Check for strong downtrend before flag
            lookback = 20  # Look at last 20 candles
            
            # Split into pole and flag
            pole_period = 10
            flag_period = 10
            
            # Get pole data (earlier period)
            pole_data = df.iloc[-lookback:-flag_period]
            
            # Get flag data (recent period)
            flag_data = df.iloc[-flag_period:]
            
            # Pole should be a strong downtrend
            pole_start_price = pole_data['close'].iloc[0]
            pole_end_price = pole_data['close'].iloc[-1]
            pole_loss = (pole_end_price / pole_start_price - 1) * 100
            
            # Flag should be a shallow uptrend or consolidation
            flag_start_price = flag_data['close'].iloc[0]
            flag_end_price = flag_data['close'].iloc[-1]
            flag_change = (flag_end_price / flag_start_price - 1) * 100
            
            # Check criteria
            if (pole_loss < -10 and  # Strong downtrend in pole (at least 10%)
                flag_change > 0 and  # Flag should be up or flat
                flag_change < abs(pole_loss) * 0.5):  # Flag retracement less than 50% of pole loss
                
                # Check flag volume (should be lower than pole)
                pole_avg_volume = pole_data['volume'].mean()
                flag_avg_volume = flag_data['volume'].mean()
                
                if flag_avg_volume < pole_avg_volume:
                    return True
        except Exception as e:
            logger.debug(f"Error detecting bear flag: {e}")
        
        return False

    def get_pattern_signal(self, patterns: List[str]) -> float:
        """Convert detected patterns to a trading signal with strength assessment."""
        if not patterns:
            return 0
        
        # Pattern signals with strength values
        pattern_signals = {
            "head_and_shoulders": -0.7,        # Strong bearish
            "inverse_head_and_shoulders": 0.7,  # Strong bullish
            "double_top": -0.5,                # Moderately bearish
            "double_bottom": 0.5,              # Moderately bullish
            "triple_top": -0.6,                # Strongly bearish
            "triple_bottom": 0.6,              # Strongly bullish
            "ascending_triangle": 0.4,         # Moderately bullish
            "descending_triangle": -0.4,       # Moderately bearish
            "symmetrical_triangle": 0,         # Neutral (needs direction confirmation)
            "bull_flag": 0.5,                  # Moderately bullish
            "bear_flag": -0.5,                 # Moderately bearish
            "bullish_engulfing": 0.3,          # Mildly bullish
            "bearish_engulfing": -0.3          # Mildly bearish
        }
        
        # Calculate total signal
        signal = 0
        pattern_count = 0
        
        for pattern in patterns:
            if pattern in pattern_signals:
                signal += pattern_signals[pattern]
                pattern_count += 1
                logger.debug(f"Pattern {pattern} contributes {pattern_signals[pattern]} to signal")
        
        # Average the signal if we have multiple patterns
        if pattern_count > 0:
            # Use a weighted approach: first pattern has more weight
            weighted_signal = signal / math.sqrt(pattern_count)
            
            # Cap signal at -1 to 1
            final_signal = max(-1, min(1, weighted_signal))
            logger.info(f"Final pattern signal: {final_signal:.2f} from {pattern_count} patterns")
            return final_signal
        
        return 0

class RiskManagement:
    """Advanced risk management and position sizing."""
    
    def __init__(self, config):
        """Initialize risk management with configuration."""
        self.config = config
        self.max_positions = config.get("trade_settings", "max_positions")
        self.account_risk_per_trade = config.get("trade_settings", "account_risk_per_trade")
        self.default_stop_loss_pct = config.get("trade_settings", "default_stop_loss_pct")
        self.default_take_profit_pct = config.get("trade_settings", "default_take_profit_pct")
        self.trailing_stop = config.get("trade_settings", "trailing_stop")
        self.trailing_stop_activation = config.get("trade_settings", "trailing_stop_activation")
        self.trailing_stop_distance = config.get("trade_settings", "trailing_stop_distance")
        self.max_daily_loss = config.get("risk_management", "max_daily_loss")
        self.max_drawdown = config.get("risk_management", "max_drawdown")
        self.position_correlation_limit = config.get("risk_management", "position_correlation_limit")
        
        # Daily tracking
        self.daily_stats = {
            "date": datetime.now().date(),
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0,
            "loss": 0,
            "net_pnl": 0,
            "starting_balance": 0,
            "current_balance": 0,
        }
        
        # Performance tracking
        self.trade_history = []
        self.equity_curve = []
        self.max_equity = 0
        self.current_drawdown = 0
    
    def reset_daily_stats(self, current_balance):
        """Reset daily statistics."""
        today = datetime.now().date()
        
        # Only reset if it's a new day
        if self.daily_stats["date"] != today:
            # Save previous day's stats to history
            self._save_daily_stats()
            
            # Reset for new day
            self.daily_stats = {
                "date": today,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "profit": 0,
                "loss": 0,
                "net_pnl": 0,
                "starting_balance": current_balance,
                "current_balance": current_balance,
            }
            
            logger.info(f"Daily stats reset for {today}")
    
    def _save_daily_stats(self):
        """Save daily stats to history file."""
        try:
            stats_file = "daily_stats.json"
            
            # Load existing stats
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats_history = json.load(f)
            else:
                stats_history = []
            
            # Add current day stats
            stats_history.append({
                "date": str(self.daily_stats["date"]),
                "trades": self.daily_stats["trades"],
                "wins": self.daily_stats["wins"],
                "losses": self.daily_stats["losses"],
                "profit": self.daily_stats["profit"],
                "loss": self.daily_stats["loss"],
                "net_pnl": self.daily_stats["net_pnl"],
                "starting_balance": self.daily_stats["starting_balance"],
                "ending_balance": self.daily_stats["current_balance"],
            })
            
            # Save updated history
            with open(stats_file, 'w') as f:
                json.dump(stats_history, f, indent=4)
            
            logger.info(f"Daily stats saved for {self.daily_stats['date']}")
        except Exception as e:
            logger.error(f"Error saving daily stats: {e}")
    
    def update_equity_curve(self, current_balance):
        """Update equity curve and drawdown metrics."""
        timestamp = datetime.now()
        
        # Add point to equity curve
        self.equity_curve.append({
            "timestamp": timestamp,
            "balance": current_balance
        })
        
        # Update max equity and drawdown
        if current_balance > self.max_equity:
            self.max_equity = current_balance
        
        if self.max_equity > 0:
            self.current_drawdown = (self.max_equity - current_balance) / self.max_equity
        else:
            self.current_drawdown = 0
    
    def record_trade(self, trade: TradePosition):
        """Record a completed trade in history."""
        if trade.status != "closed":
            return
        
        # Add to trade history
        self.trade_history.append({
            "id": trade.id,
            "symbol": trade.symbol,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "amount": trade.amount,
            "pnl": trade.pnl,
            "pnl_percentage": trade.pnl_percentage,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "duration_minutes": (trade.exit_time - trade.entry_time).total_seconds() / 60,
            "tags": trade.tags
        })
        
        # Update daily stats
        self.daily_stats["trades"] += 1
        if trade.pnl > 0:
            self.daily_stats["wins"] += 1
            self.daily_stats["profit"] += trade.pnl
        else:
            self.daily_stats["losses"] += 1
            self.daily_stats["loss"] += abs(trade.pnl)
        
        self.daily_stats["net_pnl"] += trade.pnl
        
        # Save trade history periodically
        if len(self.trade_history) % 10 == 0:
            self._save_trade_history()
    
    def _save_trade_history(self):
        """Save trade history to file."""
        try:
            with open("trade_history.json", 'w') as f:
                json.dump(self.trade_history, f, indent=4)
            logger.info(f"Trade history saved ({len(self.trade_history)} trades)")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, 
                               account_balance: float, signal_strength: float) -> float:
        """Calculate position size based on risk parameters and signal strength."""
        # Validate inputs
        if entry_price <= 0 or stop_loss <= 0 or account_balance <= 0:
            logger.error(f"Invalid inputs for position sizing: entry={entry_price}, stop={stop_loss}, balance={account_balance}")
            return 0
        
        # Calculate risk amount
        base_risk_pct = self.account_risk_per_trade
        
        # Scale risk based on signal strength (0.5 to 1.5 times base risk)
        adjusted_risk_pct = base_risk_pct * (0.5 + abs(signal_strength))
        
        # Cap at 1.5x base risk
        adjusted_risk_pct = min(base_risk_pct * 1.5, adjusted_risk_pct)
        
        # Daily loss limit check
        if self.daily_stats["net_pnl"] < -self.max_daily_loss * self.daily_stats["starting_balance"]:
            # We've hit daily loss limit, reduce position size
            adjusted_risk_pct = adjusted_risk_pct * 0.5
            logger.warning(f"Daily loss limit reached, reducing position size by 50%")
        
        # Drawdown check
        if self.current_drawdown > self.max_drawdown * 0.7:
            # Getting close to max drawdown, reduce position size
            drawdown_factor = 1 - (self.current_drawdown / (self.max_drawdown))
            adjusted_risk_pct = adjusted_risk_pct * max(0.25, drawdown_factor)
            logger.warning(f"Approaching max drawdown ({self.current_drawdown:.1%}), reducing position size")
        
        # Calculate risk per unit
        risk_amount = account_balance * adjusted_risk_pct
        
        # Calculate position size based on stop loss distance
        if signal_strength > 0:  # Long position
            risk_per_unit = entry_price - stop_loss
        else:  # Short position
            risk_per_unit = stop_loss - entry_price
        
        # Protect against invalid stop loss
        if risk_per_unit <= 0:
            logger.error(f"Invalid stop loss placement: entry={entry_price}, stop={stop_loss}")
            return 0
        
        # Calculate position size in base currency (e.g., BTC)
        position_size = risk_amount / risk_per_unit
        
        # Convert to position size in quote currency (e.g., USDT)
        position_value = position_size * entry_price
        
        # Ensure position value doesn't exceed a percentage of account
        max_position_value = account_balance * 0.3  # Maximum 30% of account per position
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            logger.info(f"Position size capped at 30% of account")
        
        return position_size
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, side: str, 
                           df: pd.DataFrame, atr_multiplier: float = 2.0) -> float:
        """Calculate dynamic stop loss price based on market volatility."""
        try:
            # Default stop loss percentage
            default_stop_pct = self.default_stop_loss_pct
            
            # Get ATR if we have price data
            if df is not None and len(df) > 14:
                # Calculate Average True Range
                atr = ta.volatility.average_true_range(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    window=14
                ).iloc[-1]
                
                # Convert ATR to percentage of price
                atr_pct = atr / entry_price
                
                # Use ATR for dynamic stop loss, with multiplier
                stop_pct = atr_pct * atr_multiplier
                
                # Ensure minimum stop distance
                stop_pct = max(stop_pct, default_stop_pct * 0.5)
                
                # Cap maximum stop distance
                stop_pct = min(stop_pct, default_stop_pct * 2.0)
            else:
                # Use default if no price data
                stop_pct = default_stop_pct
            
            # Calculate stop loss price based on side
            if side.lower() == "buy":
                stop_price = entry_price * (1 - stop_pct)
            else:  # sell
                stop_price = entry_price * (1 + stop_pct)
            
            # Round to appropriate precision
            tick_size = 0.5  # Default, should be fetched from exchange
            stop_price = round(stop_price / tick_size) * tick_size
            
            return stop_price
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            
            # Fall back to default stop loss
            if side.lower() == "buy":
                return entry_price * (1 - self.default_stop_loss_pct)
            else:
                return entry_price * (1 + self.default_stop_loss_pct)
    
    def calculate_take_profit(self, symbol: str, entry_price: float, side: str, 
                             stop_loss: float) -> float:
        """Calculate take profit based on risk-reward ratio."""
        try:
            # Calculate risk (distance to stop loss)
            if side.lower() == "buy":
                risk = entry_price - stop_loss
                # Default take profit at 2R (risk:reward ratio of 1:2)
                take_profit = entry_price + (risk * 2)
            else:  # sell
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * 2)
            
            # Ensure minimum take profit
            min_tp_pct = self.default_take_profit_pct
            
            if side.lower() == "buy":
                min_tp = entry_price * (1 + min_tp_pct)
                take_profit = max(take_profit, min_tp)
            else:
                min_tp = entry_price * (1 - min_tp_pct)
                take_profit = min(take_profit, min_tp)
            
            # Round to appropriate precision
            tick_size = 0.5  # Default, should be fetched from exchange
            take_profit = round(take_profit / tick_size) * tick_size
            
            return take_profit
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            
            # Fall back to default take profit
            if side.lower() == "buy":
                return entry_price * (1 + self.default_take_profit_pct)
            else:
                return entry_price * (1 - self.default_take_profit_pct)
    
    def check_risk_limits(self, active_positions: List[TradePosition], account_balance: float) -> bool:
        """Check if current portfolio is within risk limits."""
        # Check number of positions
        if len(active_positions) >= self.max_positions:
            logger.info(f"Maximum positions ({self.max_positions}) reached")
            return False
        
        # Check daily loss limit
        if self.daily_stats["net_pnl"] < -self.max_daily_loss * self.daily_stats["starting_balance"]:
            logger.warning(f"Daily loss limit reached: {self.daily_stats['net_pnl']}")
            return False
        
        # Check drawdown limit
        if self.current_drawdown > self.max_drawdown:
            logger.warning(f"Maximum drawdown limit reached: {self.current_drawdown:.1%}")
            return False
        
        # Check position correlation if more than one position
        if len(active_positions) > 1:
            symbols = [pos.symbol for pos in active_positions]
            is_correlated = self._check_correlation(symbols)
            if is_correlated:
                logger.info(f"Positions are too correlated, limiting additional risk")
                return False
        
        return True
    
    def _check_correlation(self, symbols: List[str]) -> bool:
        """Check if symbols are highly correlated based on actual price data."""
        # If only one symbol, no correlation
        if len(symbols) <= 1:
            return False
            
        try:
            # Calculate correlation matrix for all symbols
            correlation_matrix = {}
            
            # Build combined DataFrame of prices
            combined_prices = pd.DataFrame()
            
            # Use the primary timeframe for correlation analysis
            primary_timeframe = self.config.get("timeframes", "primary")
            
            # Get up to 100 candles for each symbol
            for symbol in symbols:
                if symbol in self.trading_bot.price_data and primary_timeframe in self.trading_bot.price_data[symbol]:
                    df = self.trading_bot.price_data[symbol][primary_timeframe]
                    # Use last 100 candles for correlation
                    if len(df) > 20:  # Need enough data for meaningful correlation
                        price_series = df['close'].iloc[-100:]
                        combined_prices[symbol] = price_series
            
            # If we don't have enough price data, fall back to safer approach
            if combined_prices.empty or combined_prices.shape[1] < 2:
                logger.warning("Insufficient price data for correlation analysis, using fallback method")
                return self._check_correlation_fallback(symbols)
            
            # Calculate correlation matrix
            correlation_matrix = combined_prices.corr()
            
            # Check for high correlations between any pair of symbols
            high_correlation_count = 0
            total_pair_count = 0
            high_correlation_pairs = []
            
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    sym1, sym2 = symbols[i], symbols[j]
                    
                    # Skip if we don't have correlation data
                    if sym1 not in correlation_matrix.index or sym2 not in correlation_matrix.columns:
                        continue
                        
                    correlation = correlation_matrix.loc[sym1, sym2]
                    total_pair_count += 1
                    
                    # Check if correlation exceeds threshold
                    if abs(correlation) > self.position_correlation_limit:
                        high_correlation_count += 1
                        high_correlation_pairs.append((sym1, sym2, correlation))
            
            # If more than 30% of pairs are highly correlated, consider the portfolio correlated
            if total_pair_count > 0 and high_correlation_count / total_pair_count > 0.3:
                logger.info(f"Portfolio is highly correlated: {high_correlation_count}/{total_pair_count} pairs exceed limit")
                for pair in high_correlation_pairs:
                    logger.debug(f"High correlation: {pair[0]} and {pair[1]}: {pair[2]:.2f}")
                return True
                
            # Additional check for cluster correlation
            # If we have groups of assets that are all correlated with each other
            G = nx.Graph()
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    sym1, sym2 = symbols[i], symbols[j]
                    
                    # Skip if we don't have correlation data
                    if sym1 not in correlation_matrix.index or sym2 not in correlation_matrix.columns:
                        continue
                        
                    correlation = correlation_matrix.loc[sym1, sym2]
                    
                    # Add edge if correlation exceeds threshold
                    if abs(correlation) > self.position_correlation_limit:
                        G.add_edge(sym1, sym2, weight=correlation)
            
            # Find clusters/communities in the graph
            if nx.number_of_nodes(G) > 0:
                communities = list(nx.community.greedy_modularity_communities(G))
                
                # Check if any community is too large
                for community in communities:
                    if len(community) > self.max_positions * 0.5:  # More than 50% of max positions in one cluster
                        logger.info(f"Found correlated cluster with {len(community)} assets: {community}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            # Fall back to safer approach in case of error
            return self._check_correlation_fallback(symbols)

    def _check_correlation_fallback(self, symbols: List[str]) -> bool:
        """Fallback correlation check using asset categories when price data isn't available."""
        # Count unique base currencies (e.g., BTC, ETH, etc.)
        base_currencies = set([symbol.split('/')[0] for symbol in symbols])
        
        # If all positions are in the same base currency, consider it correlated
        if len(base_currencies) == 1 and len(symbols) >= 3:
            logger.info(f"All positions ({len(symbols)}) are in the same base currency: {list(base_currencies)[0]}")
            return True
        
        # Define cryptocurrency groups
        crypto_groups = {
            "major": ["BTC", "ETH"],
            "defi": ["UNI", "AAVE", "SNX", "COMP", "SUSHI", "YFI", "CAKE", "CRV"],
            "layer1": ["SOL", "ADA", "AVAX", "DOT", "ATOM", "NEAR", "FTM", "ONE", "ALGO"],
            "meme": ["DOGE", "SHIB", "ELON", "FLOKI"],
            "exchange": ["BNB", "FTT", "CRO", "OKB", "KCS"],
            "oracle": ["LINK", "BAND", "API3"],
            "gaming": ["AXS", "SAND", "MANA", "ENJ", "GALA", "ILV"],
            "storage": ["FIL", "AR", "STORJ"],
        }
        
        # Count positions in each group
        group_counts = {group: 0 for group in crypto_groups}
        
        for symbol in symbols:
            base = symbol.split('/')[0]
            for group, tokens in crypto_groups.items():
                if base in tokens:
                    group_counts[group] += 1
                    break
        
        # If any group has more than limit positions, consider it correlated
        correlation_threshold = max(2, len(symbols) * self.position_correlation_limit)
        
        for group, count in group_counts.items():
            if count > correlation_threshold:
                logger.info(f"Too many positions ({count}) in the {group} category")
                return True
        
        return False

class TradeManager:
    """Manages trade execution, monitoring, and updates."""
    
    def __init__(self, config, api, risk_manager):
        """Initialize trade manager with configuration."""
        self.config = config
        self.api = api
        self.risk_manager = risk_manager
        self.active_positions = []
        self.pending_orders = []
        self.order_history = {}
        
        # Position updates cache
        self.position_updates = {}
        
        # Lock for synchronizing position updates
        self._lock = threading.Lock()
        
        # Trade execution settings
        self.order_type = config.get("execution", "order_type")
        self.limit_order_distance = config.get("execution", "limit_order_distance")
        self.retry_attempts = config.get("execution", "retry_attempts")
        self.retry_delay = config.get("execution", "retry_delay")
        self.execution_cooldown = config.get("execution", "execution_cooldown")
        self.last_execution_time = datetime.now() - timedelta(seconds=self.execution_cooldown)
        
        # Load existing positions
        self._load_positions()
    
    def _load_positions(self):
        """Load existing positions from exchange."""
        try:
            positions = self.api.get_positions()
            
            for pos in positions:
                if float(pos['contracts']) > 0:
                    # Convert to our internal position format
                    symbol = pos['symbol']
                    side = "buy" if pos['side'] == 'long' else "sell"
                    entry_price = float(pos['entryPrice'])
                    amount = float(pos['contracts'])
                    
                    # Create position object
                    position = TradePosition(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        amount=amount,
                        stop_loss=0,  # Will be updated later
                        take_profit=0,  # Will be updated later
                        entry_time=datetime.now() - timedelta(days=1),  # Estimate
                        id=pos['id'] if 'id' in pos else str(uuid.uuid4()),
                        status="open"
                    )
                    
                    self.active_positions.append(position)
            
            logger.info(f"Loaded {len(self.active_positions)} existing positions")
        except Exception as e:
            logger.error(f"Error loading existing positions: {e}")
    
    def open_position(self, symbol: str, side: str, signal_strength: float, 
                     current_price: float, dataframe: pd.DataFrame) -> Optional[str]:
        """Open a new trading position."""
        try:
            # Get account balance
            balance_info = self.api.get_wallet_balance("USDT")
            account_balance = balance_info.get('total', 0)
            
            if account_balance <= 0:
                logger.error(f"Insufficient balance to open position")
                return None
            
            # Check risk limits
            if not self.risk_manager.check_risk_limits(self.active_positions, account_balance):
                logger.info(f"Risk limits prevent opening new position for {symbol}")
                return None
            
            # Calculate stop loss
            stop_loss = self.risk_manager.calculate_stop_loss(
                symbol=symbol,
                entry_price=current_price,
                side=side,
                df=dataframe
            )
            
            # Calculate take profit
            take_profit = self.risk_manager.calculate_take_profit(
                symbol=symbol,
                entry_price=current_price,
                side=side,
                stop_loss=stop_loss
            )
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=current_price,
                stop_loss=stop_loss,
                account_balance=account_balance,
                signal_strength=signal_strength
            )
            
            if position_size <= 0:
                logger.warning(f"Position size calculation resulted in zero or negative size")
                return None
            
            # Execute the order
            order_result = self._execute_order(
                symbol=symbol,
                side=side,
                quantity=position_size,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if not order_result or "id" not in order_result:
                logger.error(f"Failed to execute order: {order_result}")
                return None
            
            # Create position object
            position = TradePosition(
                symbol=symbol,
                side=side,
                entry_price=current_price,
                amount=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.now(),
                order_id=order_result["id"],
                status="open",
                tags=[f"signal_strength_{signal_strength:.2f}"]
            )
            
            # Add to active positions
            with self._lock:
                self.active_positions.append(position)
            
            logger.info(f"Opened new position: {symbol} {side} at {current_price}, size: {position_size}, stop: {stop_loss}, tp: {take_profit}")
            
            return position.id
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None
    
    def _execute_order(self, symbol: str, side: str, quantity: float, price: float,
                      stop_loss: float, take_profit: float) -> dict:
        """Execute an order with retry logic."""
        attempts = 0
        order_result = None
        
        while attempts < self.retry_attempts:
            try:
                if self.order_type.upper() == "LIMIT":
                    # Calculate limit price with defined distance
                    if side.lower() == "buy":
                        limit_price = price * (1 + self.limit_order_distance)
                    else:
                        limit_price = price * (1 - self.limit_order_distance)
                    
                    # Place limit order
                    order_result = self.api.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        order_type="LIMIT",
                        price=limit_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                else:
                    # Place market order
                    order_result = self.api.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        order_type="MARKET",
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                
                if order_result and "id" in order_result:
                    # Success
                    return order_result
                
                # Failed, retry
                logger.warning(f"Order attempt {attempts+1} failed: {order_result}")
                attempts += 1
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Error executing order (attempt {attempts+1}): {e}")
                attempts += 1
                time.sleep(self.retry_delay)
        
        logger.error(f"Failed to execute order after {self.retry_attempts} attempts")
        return None
    
    def close_position(self, position_id: str, reason: str = "manual") -> bool:
        """Close an existing position."""
        # Find the position
        position = None
        for pos in self.active_positions:
            if pos.id == position_id:
                position = pos
                break
        
        if not position:
            logger.error(f"Position {position_id} not found")
            return False
        
        try:
            # Place closing order
            close_side = "sell" if position.side == "buy" else "buy"
            
            order_result = self.api.place_order(
                symbol=position.symbol,
                side=close_side,
                quantity=position.amount,
                order_type="MARKET",
                reduce_only=True
            )
            
            if not order_result or "id" not in order_result:
                logger.error(f"Failed to close position {position_id}: {order_result}")
                return False
            
            # Get execution price
            if "price" in order_result:
                exit_price = float(order_result["price"])
            else:
                # Get from recent market data if not provided
                ticker = self.api.get_ticker(position.symbol)
                exit_price = ticker.get("last_price", position.entry_price)
            
            # Calculate P&L
            if position.side == "buy":
                pnl = (exit_price - position.entry_price) * position.amount
                pnl_percentage = (exit_price / position.entry_price - 1) * 100
            else:
                pnl = (position.entry_price - exit_price) * position.amount
                pnl_percentage = (position.entry_price / exit_price - 1) * 100
            
            # Update position
            position.exit_price = exit_price
            position.exit_time = datetime.now()
            position.status = "closed"
            position.pnl = pnl
            position.pnl_percentage = pnl_percentage
            position.tags.append(f"exit_reason_{reason}")
            
            # Record trade
            self.risk_manager.record_trade(position)
            
            # Remove from active positions
            with self._lock:
                self.active_positions = [pos for pos in self.active_positions if pos.id != position_id]
            
            logger.info(f"Closed position {position_id}: {position.symbol} {position.side} at {exit_price}, PnL: {pnl:.2f} ({pnl_percentage:.2f}%)")
            
            return True
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return False
    
    def update_position(self, position_id: str, stop_loss: float = None, 
                       take_profit: float = None) -> bool:
        """Update an existing position's parameters."""
        # Find the position
        position = None
        for pos in self.active_positions:
            if pos.id == position_id:
                position = pos
                break
        
        if not position:
            logger.error(f"Position {position_id} not found for update")
            return False
        
        try:
            updates = {}
            if stop_loss is not None:
                updates["stop_loss"] = stop_loss
            if take_profit is not None:
                updates["take_profit"] = take_profit
            
            if not updates:
                logger.warning(f"No updates provided for position {position_id}")
                return False
            
            # Update on exchange
            result = self.api.amend_order(
                order_id=position.order_id,
                symbol=position.symbol,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if not result or "success" in result and not result["success"]:
                logger.error(f"Failed to update position {position_id} on exchange: {result}")
                return False
            
            # Update local position object
            with self._lock:
                if stop_loss is not None:
                    position.stop_loss = stop_loss
                if take_profit is not None:
                    position.take_profit = take_profit
            
            logger.info(f"Updated position {position_id}: {updates}")
            return True
        except Exception as e:
            logger.error(f"Error updating position {position_id}: {e}")
            return False
    
    def manage_trailing_stops(self, current_prices: Dict[str, float]):
        """Update trailing stops for all active positions."""
        if not self.risk_manager.trailing_stop:
            return
        
        for position in self.active_positions:
            try:
                symbol = position.symbol
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                
                # Check if position is in profit enough to activate trailing stop
                if position.side == "buy":
                    profit_pct = (current_price / position.entry_price - 1)
                    
                    # If profit exceeds activation threshold
                    if profit_pct > self.risk_manager.trailing_stop_activation:
                        # Calculate new stop loss level
                        new_stop = current_price * (1 - self.risk_manager.trailing_stop_distance)
                        
                        # Only move stop loss up, never down
                        if new_stop > position.stop_loss:
                            self.update_position(position.id, stop_loss=new_stop)
                
                else:  # sell position
                    profit_pct = (position.entry_price / current_price - 1)
                    
                    # If profit exceeds activation threshold
                    if profit_pct > self.risk_manager.trailing_stop_activation:
                        # Calculate new stop loss level
                        new_stop = current_price * (1 + self.risk_manager.trailing_stop_distance)
                        
                        # Only move stop loss down, never up
                        if new_stop < position.stop_loss or position.stop_loss == 0:
                            self.update_position(position.id, stop_loss=new_stop)
            
            except Exception as e:
                logger.error(f"Error managing trailing stop for {position.id}: {e}")
    
    def check_stop_conditions(self, current_prices: Dict[str, float]) -> List[str]:
        """Check if any positions should be closed based on market conditions."""
        positions_to_close = []
        
        for position in self.active_positions:
            try:
                symbol = position.symbol
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                
                # Check stop loss hit
                if position.side == "buy" and position.stop_loss > 0 and current_price <= position.stop_loss:
                    positions_to_close.append((position.id, "stop_loss"))
                    continue
                
                elif position.side == "sell" and position.stop_loss > 0 and current_price >= position.stop_loss:
                    positions_to_close.append((position.id, "stop_loss"))
                    continue
                
                # Check take profit hit
                if position.side == "buy" and position.take_profit > 0 and current_price >= position.take_profit:
                    positions_to_close.append((position.id, "take_profit"))
                    continue
                
                elif position.side == "sell" and position.take_profit > 0 and current_price <= position.take_profit:
                    positions_to_close.append((position.id, "take_profit"))
                    continue
                
            except Exception as e:
                logger.error(f"Error checking stop conditions for {position.id}: {e}")
        
        # Close positions
        closed_positions = []
        for position_id, reason in positions_to_close:
            success = self.close_position(position_id, reason)
            if success:
                closed_positions.append(position_id)
        
        return closed_positions
    
    def get_position_summary(self) -> Dict:
        """Get a summary of all active positions."""
        summary = {
            "total_positions": len(self.active_positions),
            "long_positions": 0,
            "short_positions": 0,
            "total_exposure": 0,
            "positions": []
        }
        
        for position in self.active_positions:
            position_value = position.entry_price * position.amount
            
            if position.side == "buy":
                summary["long_positions"] += 1
            else:
                summary["short_positions"] += 1
            
            summary["total_exposure"] += position_value
            
            # Add to detailed positions list
            summary["positions"].append({
                "id": position.id,
                "symbol": position.symbol,
                "side": position.side,
                "entry_price": position.entry_price,
                "amount": position.amount,
                "value": position_value,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "entry_time": position.entry_time.isoformat()
            })
        
        return summary


class TradingBot:
    """Main trading bot class that coordinates all components."""
    
    def __init__(self, config_file="config.json"):
        """Initialize the trading bot with configuration."""
        # Load configuration
        self.config = Config(config_file)
        
        # Initialize API
        self.api = BybitAPI(self.config)
        
        # Initialize components
        self.features = FeatureEngineering(self.config)
        self.sentiment = SentimentAnalyzer(self.config)
        self.market_analyzer = MarketAnalysis(self.config)
        self.risk_manager = RiskManagement(self.config)
        self.ml_models = MLModels(self.config, self.features)
        self.trade_manager = TradeManager(self.config, self.api, self.risk_manager)
        
        # Data caches
        self.price_data = {}  # Symbol -> Timeframe -> DataFrame
        self.current_prices = {}  # Symbol -> Current Price
        self.trading_signals = {}  # Symbol -> Signal
        
        # Synchronization
        self._lock = threading.Lock()
        
        # State
        self.is_running = False
        self.last_model_train = {}
        self.market_hours = self._is_market_hours()
        
        # Scheduling
        self.scheduler = schedule.Scheduler()
        self._setup_schedules()
        
        logger.info("Trading bot initialized")
    
    def _setup_schedules(self):
        """Set up scheduled tasks."""
        # Update market data every minute
        self.scheduler.every(1).minutes.do(self.update_market_data)
        
        # Process trading signals every 5 minutes
        self.scheduler.every(5).minutes.do(self.process_trading_signals)
        
        # Manage positions every 2 minutes
        self.scheduler.every(2).minutes.do(self.manage_positions)
        
        # Check for stop conditions every 30 seconds
        self.scheduler.every(30).seconds.do(self.check_stop_conditions)
        
        # Update sentiments every 30 minutes
        self.scheduler.every(30).minutes.do(self.update_sentiments)
        
        # Calculate performance metrics every 6 hours
        metrics_interval = self.config.get("performance_monitoring", "metrics_interval_hours")
        self.scheduler.every(metrics_interval).hours.do(self.calculate_performance_metrics)
        
        # Reset daily stats at midnight
        self.scheduler.every().day.at("00:00").do(self._reset_daily_stats)
        
        # Training queues
        self.scheduler.every(30).seconds.do(self._process_training_queue)

    
    def start(self):
        """Start the trading bot."""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        try:
            logger.info("Starting trading bot")
            self.is_running = True
            
            # Initialize data
            self.update_market_data()
            self.update_sentiments()
            
            # Main loop
            while self.is_running:
                try:
                    self.scheduler.run_pending()
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
            self.stop()
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            self.stop()
    
    def stop(self):
        """Stop the trading bot."""
        logger.info("Stopping trading bot")
        self.is_running = False
        
        # Save data and state
        self._save_state()
    
    def update_market_data(self):
        """Update market data for all configured symbols and timeframes."""
        symbols = self.config.get("symbols")
        timeframes = self.config.get("timeframes")
        
        for symbol in symbols:
            try:
                # Update current price
                ticker = self.api.get_ticker(symbol)
                if ticker and "last_price" in ticker:
                    self.current_prices[symbol] = ticker["last_price"]
                
                # Update price data for each timeframe
                for tf_name, tf_value in timeframes.items():
                    if isinstance(tf_value, list):
                        # Handle multiple timeframes in a category
                        for tf in tf_value:
                            self._update_price_data(symbol, tf)
                    else:
                        # Single timeframe
                        self._update_price_data(symbol, tf_value)
            except Exception as e:
                logger.error(f"Error updating market data for {symbol}: {e}")
        
        # Update market hours status
        self.market_hours = self._is_market_hours()
        
        logger.debug(f"Market data updated for {len(symbols)} symbols")
    
    def _update_price_data(self, symbol: str, timeframe: str):
        """Update price data for a specific symbol and timeframe."""
        try:
            # Get most recent data
            ohlcv = self.api.get_klines(symbol, timeframe, limit=500)
            
            if ohlcv.empty:
                logger.warning(f"No data received for {symbol} on {timeframe} timeframe")
                return
            
            # Store in cache
            if symbol not in self.price_data:
                self.price_data[symbol] = {}
            
            self.price_data[symbol][timeframe] = ohlcv
            
            # Check if models need training
            model_key = f"{symbol}_{timeframe}"
            if model_key not in self.ml_models.last_trained or (
                        datetime.now() - self.ml_models.last_trained.get(model_key, datetime.min)).days >= self.config.get("ml_settings", "train_interval_days"):
                    
                    # Add to training queue instead of starting thread immediately
                    if not hasattr(self, 'training_queue'):
                        self.training_queue = []
                        self.is_training = False
                    
                    self.training_queue.append((symbol, timeframe, ohlcv.copy()))
                    logger.info(f"Added {symbol} {timeframe} to training queue (position {len(self.training_queue)})")
        
        except Exception as e:
            logger.error(f"Error updating price data for {symbol} on {timeframe}: {e}")

    def _train_models_for_symbol(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Train ML models for a symbol and timeframe."""
        try:
            logger.info(f"Training models for {symbol} on {timeframe}")
            
            success = self.ml_models.train_models(symbol, timeframe, df)
            
            if success:
                self.last_model_train[f"{symbol}_{timeframe}"] = datetime.now()
                logger.info(f"Model training completed for {symbol} on {timeframe}")
            else:
                logger.warning(f"Model training failed for {symbol} on {timeframe}")
        
        except Exception as e:
            logger.error(f"Error training models for {symbol} on {timeframe}: {e}")
    
    def _process_training_queue(self):
        """Process the training queue one item at a time to avoid memory issues."""
        if not hasattr(self, 'training_queue') or not self.training_queue or self.is_training:
            return
        
        self.is_training = True
        symbol, timeframe, ohlcv = self.training_queue.pop(0)
        logger.info(f"Processing training queue: {symbol} {timeframe} (remaining: {len(self.training_queue)})")
        
        try:
            self._train_models_for_symbol(symbol, timeframe, ohlcv)
        except Exception as e:
            logger.error(f"Error training from queue: {e}")
        finally:
            self.is_training = False

    def update_sentiments(self):
        """Update sentiment data for all symbols."""
        if not self.sentiment.enabled:
            return
        
        symbols = self.config.get("symbols")
        
        for symbol in symbols:
            try:
                self.sentiment.update_sentiment(symbol)
            except Exception as e:
                logger.error(f"Error updating sentiment for {symbol}: {e}")
        
        logger.debug(f"Sentiment data updated for {len(symbols)} symbols")
    
    def process_trading_signals(self):

        # Add near the top of process_trading_signals method

        """Process trading signals for all symbols."""
        if not self.market_hours:
            logger.info("Market is closed or low liquidity hours, skipping signal processing")
            return
        
        # Get relevant data
        symbols = self.config.get("symbols")
        primary_timeframe = self.config.get("timeframes", "primary")
        secondary_timeframes = self.config.get("timeframes", "secondary")
        
        for symbol in symbols:
            logger.info(f"Processing signals for {symbol}...")
            try:
                # Check if we have data for this symbol
                if symbol not in self.price_data or primary_timeframe not in self.price_data[symbol]:
                    logger.warning(f"No price data available for {symbol} on {primary_timeframe}")
                    continue


                
                # Get primary timeframe data
                primary_df = self.price_data[symbol][primary_timeframe]
                
                # Get ML model signals
                ml_signals = self.ml_models.get_trading_signals(symbol, primary_timeframe, primary_df)
                primary_signal = ml_signals.get("signal", 0)
                confidence = ml_signals.get("confidence", 0)
                
                # Skip low confidence signals
                if confidence < self.config.get("ml_settings", "confidence_threshold"):
                    logger.debug(f"Low confidence signal for {symbol}: {confidence:.2f}")
                    continue
                
                # Get market regime
                market_regime = self.market_analyzer.detect_market_regime(primary_df)
                
                # Adjust signal based on market regime
                regime_adjusted_signal = self.market_analyzer.get_regime_adjusted_signal(
                    symbol, primary_timeframe, primary_signal, market_regime
                )
                
                # Check for chart patterns
                patterns = self.market_analyzer.identify_patterns(primary_df)
                pattern_signal = self.market_analyzer.get_pattern_signal(patterns)
                
                # Get sentiment signal
                sentiment_signal = self.sentiment.get_sentiment_signal(symbol)
                
                # Combine signals with weights
                combined_signal = (
                    regime_adjusted_signal * 0.6 +  # ML models are the core
                    pattern_signal * 0.2 +          # Technical patterns
                    sentiment_signal * 0.2          # Sentiment analysis
                )
                
                # Check secondary timeframes for confirmation
                if len(secondary_timeframes) > 0:
                    secondary_signals = []
                    
                    for tf in secondary_timeframes:
                        if tf in self.price_data[symbol]:
                            secondary_df = self.price_data[symbol][tf]
                            if not secondary_df.empty:
                                sec_signal = self.ml_models.get_trading_signals(symbol, tf, secondary_df)
                                secondary_signals.append(sec_signal.get("signal", 0))
                    
                    if secondary_signals:
                        # Weigh longer timeframes more
                        weights = [1/(i+1) for i in range(len(secondary_signals))]
                        weight_sum = sum(weights)
                        
                        # Normalize weights
                        weights = [w/weight_sum for w in weights]
                        
                        # Calculate weighted average
                        avg_secondary_signal = sum(s*w for s, w in zip(secondary_signals, weights))
                        
                        # Use secondary timeframes as confirmation/filter
                        # If they contradict primary, reduce signal strength
                        if (combined_signal > 0 and avg_secondary_signal < 0) or \
                           (combined_signal < 0 and avg_secondary_signal > 0):
                            combined_signal *= 0.5
                
                # Store the combined signal
                self.trading_signals[symbol] = {
                    "signal": combined_signal,
                    "confidence": confidence,
                    "market_regime": market_regime.value,
                    "ml_signal": primary_signal,
                    "pattern_signal": pattern_signal,
                    "sentiment_signal": sentiment_signal,
                    "timestamp": datetime.now(),
                    "patterns": patterns
                }
                
                # Check if the signal is strong enough to act on
                self.execute_trading_signals(symbol)
                
                logger.info(f"Processed trading signal for {symbol}: {combined_signal:.2f} (confidence: {confidence:.2f})")

            
            except Exception as e:
                logger.error(f"Error processing trading signals for {symbol}: {e}")
    
    def execute_trading_signals(self, symbol: str):
        """Execute trading actions based on signals."""
        # Check execution cooldown
        time_since_last = (datetime.now() - self.trade_manager.last_execution_time).total_seconds()
        if time_since_last < self.trade_manager.execution_cooldown:
            logger.info(f"Execution cooldown active ({time_since_last:.0f}s < {self.trade_manager.execution_cooldown}s)")
            return
        
        # Check if we have a signal for this symbol
        if symbol not in self.trading_signals:
            return
        
        signal_data = self.trading_signals[symbol]
        signal = signal_data["signal"]
        confidence = signal_data["confidence"]
        
        # Check if signal is strong enough
        signal_threshold = 0.3  # Signal must be stronger than this to open a position
        if abs(signal) < signal_threshold:
            logger.info(f"Signal for {symbol} too weak to act on: {signal:.2f}")
            return
        
        # Get current price
        if symbol not in self.current_prices:
            logger.warning(f"No current price available for {symbol}")
            return
        
        current_price = self.current_prices[symbol]
        
        # Check existing positions for this symbol
        existing_positions = [p for p in self.trade_manager.active_positions if p.symbol == symbol]
        
        # Logic for opening positions
        if len(existing_positions) == 0:
            # No existing position, check if we should open one
            side = "buy" if signal > 0 else "sell"
            primary_timeframe = self.config.get("timeframes", "primary")
            
            # Open position
            position_id = self.trade_manager.open_position(
                symbol=symbol,
                side=side,
                signal_strength=abs(signal),
                current_price=current_price,
                dataframe=self.price_data[symbol][primary_timeframe] if symbol in self.price_data else None
            )
            
            if position_id:
                logger.info(f"Opened new {side} position for {symbol} with signal {signal:.2f}")
                # Update last execution time
                self.trade_manager.last_execution_time = datetime.now()
        
        else:
            # Position exists, check if we should close or reverse
            position = existing_positions[0]
            
            # Check for position reversal (strong opposite signal)
            signal_reversal_threshold = 0.6  # Need stronger signal to reverse
            
            if (position.side == "buy" and signal < -signal_reversal_threshold) or \
               (position.side == "sell" and signal > signal_reversal_threshold):
                # Close existing position
                closed = self.trade_manager.close_position(position.id, "signal_reversal")
                
                if closed:
                    logger.info(f"Closed {position.side} position for {symbol} due to signal reversal: {signal:.2f}")
                    
                    # Open new position in opposite direction (with slight delay)
                    time.sleep(1)
                    new_side = "sell" if position.side == "buy" else "buy"
                    primary_timeframe = self.config.get("timeframes", "primary")
                    
                    position_id = self.trade_manager.open_position(
                        symbol=symbol,
                        side=new_side,
                        signal_strength=abs(signal),
                        current_price=current_price,
                        dataframe=self.price_data[symbol][primary_timeframe] if symbol in self.price_data else None
                    )
                    
                    if position_id:
                        logger.info(f"Opened new {new_side} position for {symbol} with signal {signal:.2f}")
                    
                    # Update last execution time
                    self.trade_manager.last_execution_time = datetime.now()
    
    def manage_positions(self):
        """Manage existing positions (update stops, trailing stops, etc.)."""
        if not self.trade_manager.active_positions:
            return
        
        # Update trailing stops
        self.trade_manager.manage_trailing_stops(self.current_prices)
        
        # Check for other position management logic
        for position in self.trade_manager.active_positions:
            try:
                symbol = position.symbol
                
                # Skip if no current price
                if symbol not in self.current_prices:
                    continue
                
                current_price = self.current_prices[symbol]
                
                # Check time-based exit (close positions held too long)
                max_hold_time = timedelta(days=7)  # Maximum hold time
                if datetime.now() - position.entry_time > max_hold_time:
                    self.trade_manager.close_position(position.id, "time_exit")
                    logger.info(f"Closed position {position.id} due to max hold time")
                    continue
                
                # Additional position management logic can be added here
                
            except Exception as e:
                logger.error(f"Error managing position {position.id}: {e}")
    
    def check_stop_conditions(self):
        """Check if any positions should be closed based on stop conditions."""
        if not self.trade_manager.active_positions:
            return
        
        closed_positions = self.trade_manager.check_stop_conditions(self.current_prices)
        
        if closed_positions:
            logger.info(f"Closed {len(closed_positions)} positions due to stop conditions")
    
    def calculate_performance_metrics(self):
        """Calculate and log performance metrics."""
        try:
            # Get account balance
            balance_info = self.api.get_wallet_balance("USDT")
            current_balance = balance_info.get('total', 0)
            
            # Update equity curve
            self.risk_manager.update_equity_curve(current_balance)
            
            # Update daily stats
            self.risk_manager.daily_stats["current_balance"] = current_balance
            
            # Calculate metrics
            position_summary = self.trade_manager.get_position_summary()
            
            # Benchmark comparison
            benchmark = self.config.get("performance_monitoring", "benchmark")
            benchmark_return = 0
            
            if benchmark == "HODL" and "BTC/USDT" in self.price_data:
                btc_df = self.price_data["BTC/USDT"].get("1d")
                if btc_df is not None and len(btc_df) > 30:
                    start_price = btc_df.iloc[-30]['close']
                    end_price = btc_df.iloc[-1]['close']
                    benchmark_return = (end_price / start_price - 1) * 100
            
            # Trade statistics
            win_rate = 0
            if self.risk_manager.daily_stats["trades"] > 0:
                win_rate = self.risk_manager.daily_stats["wins"] / self.risk_manager.daily_stats["trades"] * 100
            
            # Log metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "account_balance": current_balance,
                "active_positions": position_summary["total_positions"],
                "total_exposure": position_summary["total_exposure"],
                "daily_trades": self.risk_manager.daily_stats["trades"],
                "daily_pnl": self.risk_manager.daily_stats["net_pnl"],
                "win_rate": win_rate,
                "current_drawdown": self.risk_manager.current_drawdown * 100,
                "benchmark_return": benchmark_return
            }
            
            logger.info(f"Performance metrics: {metrics}")
            
            # Save metrics
            self._save_metrics(metrics)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _save_metrics(self, metrics):
        """Save performance metrics to file."""
        try:
            metrics_file = "performance_metrics.json"
            
            # Load existing metrics
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_history = json.load(f)
            else:
                metrics_history = []
            
            # Add current metrics
            metrics_history.append(metrics)
            
            # Save updated history (keep last 1000 entries)
            with open(metrics_file, 'w') as f:
                json.dump(metrics_history[-1000:], f, indent=4)
            
            logger.debug(f"Performance metrics saved")
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
    
    def _reset_daily_stats(self):
        """Reset daily statistics."""
        try:
            # Get account balance
            balance_info = self.api.get_wallet_balance("USDT")
            current_balance = balance_info.get('total', 0)
            
            # Reset stats
            self.risk_manager.reset_daily_stats(current_balance)
            
            logger.info(f"Daily stats reset")
        except Exception as e:
            logger.error(f"Error resetting daily stats: {e}")
    
    def _save_state(self):
        """Save the current state of the trading bot."""
        try:
            # Save feature engineering scalers
            self.features.save_scalers()
            
            # Save trade history
            self.risk_manager._save_trade_history()
            
            # Save daily stats
            self.risk_manager._save_daily_stats()
            
            logger.info("Trading bot state saved")
        except Exception as e:
            logger.error(f"Error saving trading bot state: {e}")
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within optimal trading hours."""
        
        # For crypto, we'll generally return True as markets are 24/7
        return True


class WebSocketManager:
    """Manages WebSocket connections for real-time data."""
    
    def __init__(self, api, trading_bot):
        """Initialize WebSocket manager."""
        self.api = api
        self.trading_bot = trading_bot
        self.ws = None
        self.is_connected = False
        self.subscriptions = []
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Thread for WebSocket
        self.ws_thread = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def start(self):
        """Start WebSocket connection."""
        self.ws_thread = threading.Thread(target=self._run_websocket)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def _run_websocket(self):
        """Run WebSocket connection."""
        # Define connection URL
        if self.api.testnet:
            ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            ws_url = "wss://stream.bybit.com/v5/public/linear"
        
        # Define callback functions
        def on_message(ws, message):
            try:
                msg = json.loads(message)
                
                # Process tick data
                if 'topic' in msg and 'data' in msg and 'tick' in msg['topic']:
                    self._process_tick_data(msg)
                
                # Process kline data
                elif 'topic' in msg and 'data' in msg and 'kline' in msg['topic']:
                    self._process_kline_data(msg)
                
                # Process trade data
                elif 'topic' in msg and 'data' in msg and 'trade' in msg['topic']:
                    self._process_trade_data(msg)
                
                # Process order book data
                elif 'topic' in msg and 'data' in msg and 'orderbook' in msg['topic']:
                    self._process_orderbook_data(msg)
                
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            self.is_connected = False
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
            self.is_connected = False
            self._attempt_reconnect()
        
        def on_open(ws):
            logger.info("WebSocket connection established")
            self.is_connected = True
            self.reconnect_attempts = 0
            
            # Subscribe to channels
            self._subscribe_to_all_channels()
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start WebSocket connection
        try:
            self.ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket run error: {e}")
            self.is_connected = False
            self._attempt_reconnect()
    
    def _attempt_reconnect(self):
        """Attempt to reconnect WebSocket."""
        if not self.is_connected and self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            
            # Exponential backoff
            wait_time = min(30, 2 ** self.reconnect_attempts)
            logger.info(f"Attempting to reconnect WebSocket in {wait_time} seconds (attempt {self.reconnect_attempts})")
            
            time.sleep(wait_time)
            
            # Start a new WebSocket thread
            self.ws_thread = threading.Thread(target=self._run_websocket)
            self.ws_thread.daemon = True
            self.ws_thread.start()
        else:
            logger.warning(f"WebSocket reconnection attempts exhausted ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
    
    def _subscribe_to_all_channels(self):
        """Subscribe to all configured channels."""
        symbols = self.trading_bot.config.get("symbols")
        
        # Clear existing subscriptions
        self.subscriptions = []
        
        # Prepare subscription messages
        subscriptions = []
        
        # Subscribe to ticker data
        for symbol in symbols:
            # Convert to Bybit format
            formatted_symbol = symbol.replace('/', '')
            subscriptions.append(f"tickers.{formatted_symbol}")
            self.subscriptions.append(f"tickers.{formatted_symbol}")
        
        # Subscribe to kline data for the primary timeframe
        primary_timeframe = self.trading_bot.config.get("timeframes", "primary")
        for symbol in symbols:
            formatted_symbol = symbol.replace('/', '')
            timeframe_mapping = {
                "1m": "1",
                "5m": "5",
                "15m": "15",
                "30m": "30",
                "1h": "60",
                "4h": "240",
                "1d": "D"
            }
            tf = timeframe_mapping.get(primary_timeframe, "15")
            subscriptions.append(f"kline.{tf}.{formatted_symbol}")
            self.subscriptions.append(f"kline.{tf}.{formatted_symbol}")
        
        # Subscribe to order book (depth)
        for symbol in symbols:
            formatted_symbol = symbol.replace('/', '')
            subscriptions.append(f"orderbook.1.{formatted_symbol}")
            self.subscriptions.append(f"orderbook.1.{formatted_symbol}")
        
        # Send subscription message
        if self.is_connected and self.ws:
            subscribe_msg = {
                "op": "subscribe",
                "args": subscriptions
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {len(subscriptions)} WebSocket channels")
    
    def _process_tick_data(self, message):
        """Process tick data from WebSocket."""
        try:
            if 'data' in message and 'symbol' in message['data']:
                symbol = message['data']['symbol']
                
                # Convert from Bybit format to our format
                formatted_symbol = f"{symbol[:-4]}/{symbol[-4:]}" if symbol.endswith('USDT') else symbol
                
                # Extract price
                if 'lastPrice' in message['data']:
                    price = float(message['data']['lastPrice'])
                    
                    # Update current price
                    with self._lock:
                        self.trading_bot.current_prices[formatted_symbol] = price
                    
                    # Check stop conditions if price changed significantly
                    if self.trading_bot.trade_manager.active_positions:
                        self.trading_bot.check_stop_conditions()
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")
    
    def _process_kline_data(self, message):
        """Process kline (candle) data from WebSocket."""
        try:
            if 'data' in message and 'topic' in message:
                # Extract symbol and timeframe from topic
                topic_parts = message['topic'].split('.')
                if len(topic_parts) >= 3:
                    timeframe_code = topic_parts[1]
                    bybit_symbol = topic_parts[2]
                    
                    # Map timeframe code back to our format
                    timeframe_mapping = {
                        "1": "1m",
                        "5": "5m",
                        "15": "15m",
                        "30": "30m",
                        "60": "1h",
                        "240": "4h",
                        "D": "1d"
                    }
                    timeframe = timeframe_mapping.get(timeframe_code, "15m")
                    
                    # Convert symbol format
                    symbol = f"{bybit_symbol[:-4]}/{bybit_symbol[-4:]}" if bybit_symbol.endswith('USDT') else bybit_symbol
                    
                    # Process candle data
                    candle_data = message['data']
                    if isinstance(candle_data, list) and len(candle_data) > 0:
                        # Update in-memory price data
                        if symbol in self.trading_bot.price_data and timeframe in self.trading_bot.price_data[symbol]:
                            # Get existing dataframe
                            df = self.trading_bot.price_data[symbol][timeframe]
                            
                            # Process each new candle
                            for candle in candle_data:
                                # Convert timestamp to datetime
                                if 'timestamp' in candle and candle['timestamp']:
                                    timestamp = pd.to_datetime(candle['timestamp'], unit='ms')
                                    
                                    # Create new row
                                    new_row = pd.DataFrame({
                                        'open': [float(candle['open'])],
                                        'high': [float(candle['high'])],
                                        'low': [float(candle['low'])],
                                        'close': [float(candle['close'])],
                                        'volume': [float(candle['volume'])]
                                    }, index=[timestamp])
                                    
                                    # Update existing dataframe
                                    # If timestamp exists, replace data
                                    if timestamp in df.index:
                                        df.loc[timestamp] = new_row.loc[timestamp]
                                    else:
                                        # Append new data
                                        df = pd.concat([df, new_row])
                                    
                                    # Sort by timestamp
                                    df = df.sort_index()
                                    
                                    # Update in cache
                                    self.trading_bot.price_data[symbol][timeframe] = df
                                    
                                    logger.debug(f"Updated kline data for {symbol} on {timeframe}")
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")
    
    def _process_trade_data(self, message):
        """Process trade data from WebSocket."""
        try:
            if 'data' in message and isinstance(message['data'], list) and len(message['data']) > 0:
                # Extract symbol from topic
                topic_parts = message['topic'].split('.')
                if len(topic_parts) >= 2:
                    bybit_symbol = topic_parts[1]
                    
                    # Convert symbol format
                    symbol = f"{bybit_symbol[:-4]}/{bybit_symbol[-4:]}" if bybit_symbol.endswith('USDT') else bybit_symbol
                    
                    # Process the most recent trade
                    latest_trade = message['data'][0]
                    
                    # Update current price with the latest trade price
                    if 'price' in latest_trade:
                        price = float(latest_trade['price'])
                        side = latest_trade.get('side', '').lower()
                        
                        with self._lock:
                            self.trading_bot.current_prices[symbol] = price
                        
                        # Additional trade analysis can be added here
                        # For example, detecting large trades, etc.
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    def _process_orderbook_data(self, message):
        """Process order book data from WebSocket."""
        try:
            if 'data' in message and 'topic' in message:
                # Extract symbol from topic
                topic_parts = message['topic'].split('.')
                if len(topic_parts) >= 3:
                    bybit_symbol = topic_parts[2]
                    
                    # Convert symbol format
                    symbol = f"{bybit_symbol[:-4]}/{bybit_symbol[-4:]}" if bybit_symbol.endswith('USDT') else bybit_symbol
                    
                    # Process order book data
                    # This could be used for advanced order execution algorithms
                    # For example, detecting liquidity, etc.
                    
                    # For now, we're just logging the update
                    if 'type' in message['data']:
                        update_type = message['data']['type']
                        logger.debug(f"Order book {update_type} for {symbol}")
        except Exception as e:
            logger.error(f"Error processing orderbook data: {e}")
    
    def stop(self):
        """Stop WebSocket connection."""
        if self.ws:
            self.ws.close()
            logger.info("WebSocket connection closed")
        
        self.is_connected = False


def main():
    """Main entry point for the trading bot."""
    parser = argparse.ArgumentParser(description="AI-Powered Crypto Trading Bot")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--logfile", type=str, default="trading_bot.log", help="Path to log file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no real trades)")
    parser.add_argument("--backtest", type=str, help="Backtest using historical data from date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.logfile),
            logging.StreamHandler(sys.stdout)
        ]
    )

    
    
    # Initialize trading bot
    bot = TradingBot(args.config)

    # Update config for test mode
    if args.test:
        bot.config.update({"testnet": True})
        logger.info("Running in test mode (testnet)")
    
    # Handle backtest mode
    if args.backtest:
        logger.info(f"Backtest mode not implemented yet")
        return
    
    # Initialize WebSocket
    ws_manager = WebSocketManager(bot.api, bot)
    
    try:
        # Start WebSocket
        ws_manager.start()
        
        # Start trading bot
        bot.start()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        ws_manager.stop()
        bot.stop()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        ws_manager.stop()
        bot.stop()

    try:
        from bot_dashboard_adapter import attach_dashboard
        # Attach dashboard interface to the bot with improved command handling
        dashboard = attach_dashboard(bot)
        
        # Set up dashboard command processor to run in a separate thread
        # This ensures dashboard commands are processed promptly and don't time out
        def run_dashboard_processor():
            """Run dashboard command processor in a separate thread."""
            logger.info("Starting dashboard command processor thread")
            while bot.is_running:
                try:
                    # Process any pending dashboard commands (non-blocking)
                    if hasattr(dashboard, "process_commands") and callable(dashboard.process_commands):
                        dashboard.process_commands()
                    # Brief pause to prevent CPU spinning
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error in dashboard command processor: {e}")
                    time.sleep(1)  # Longer pause on error
        
        # Start dashboard processor thread
        dashboard_thread = threading.Thread(target=run_dashboard_processor, daemon=True)
        dashboard_thread.start()
        
        logger.info("Dashboard interface attached with dedicated command processor")
    except Exception as e:
        logger.error(f"Error attaching dashboard interface: {e}")
        dashboard = None



if __name__ == "__main__":
    main()