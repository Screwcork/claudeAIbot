#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading Bot Dashboard - HMI Interface
-------------------------------------
Provides real-time monitoring, visualization, and control for the Trading Bot.
"""

import os
import sys
import json
import time
import logging
import threading
import datetime
import pandas as pd
import numpy as np
import socket
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Web server and socketio
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, emit
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Shared memory and IPC
import mmap
import pickle
import tempfile
from filelock import FileLock
import zmq

# Dashboard logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TradingDashboard")

# Constants
DATA_DIR = "dashboard_data"
CONFIG_FILE = "config.json"
SHARED_DATA_FILE = os.path.join(DATA_DIR, "shared_data.pkl")
COMMAND_FILE = os.path.join(DATA_DIR, "commands.json")
DEFAULT_PORT = 5000
DEFAULT_HOST = "127.0.0.1"
ZMQ_PORT = 5555  # ZeroMQ port for bot communication

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Flask app and SocketIO
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'tradingbot_dashboard_secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# State variables
bot_status = {
    "status": "disconnected",  # "running", "paused", "stopped", "disconnected"
    "last_update": None,
    "uptime": 0,
    "version": "1.0.0"
}

# Communication with bot

class BotCommunication:
    """Handles communication with the trading bot."""
    
    def __init__(self, host=DEFAULT_HOST, port=ZMQ_PORT):
        """Initialize communication."""
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        # Set more aggressive socket options
        self.socket.setsockopt(zmq.LINGER, 0)      # Don't wait on close
        self.socket.setsockopt(zmq.RCVTIMEO, 5000) # 5 second timeout (reduced from 8s)
        self.socket.setsockopt(zmq.SNDTIMEO, 3000) # 3 second send timeout (reduced from 5s)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)   # Don't queue messages if no connection
        self.socket.setsockopt(zmq.RECONNECT_IVL, 500)   # Reconnect interval 0.5 seconds (reduced from 1s)
        self.socket.setsockopt(zmq.RECONNECT_IVL_MAX, 3000)  # Max reconnect interval 3 seconds (reduced from 5s)
        
        self.connected = False
        self.lock = threading.Lock()
        self.reconnect_attempt = 0
        self.last_successful_communication = 0
    
    def connect(self):
        """Connect to the bot's command socket."""
        try:
            # Close any existing connection first
            if self.connected:
                try:
                    self.socket.disconnect(f"tcp://{self.host}:{self.port}")
                except:
                    pass
                
            self.socket.connect(f"tcp://{self.host}:{self.port}")
            self.connected = True
            self.reconnect_attempt = 0
            self.last_successful_communication = time.time()
            logger.info(f"Connected to bot at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to bot: {e}")
            self.connected = False
            return False
        
    def check_connection(self):
        """Check if the connection is still valid with a simple ping."""
        try:
            # Use a very simple ping with minimal data
            message = {
                "command": "ping",  # Use "ping" which is handled by the bot adapter
                "params": {},
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Create a dedicated socket for this check to avoid interfering with main socket
            check_context = zmq.Context()
            check_socket = check_context.socket(zmq.REQ)
            check_socket.setsockopt(zmq.LINGER, 0)
            check_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout for ping
            check_socket.setsockopt(zmq.SNDTIMEO, 2000)
            check_socket.connect(f"tcp://{self.host}:{self.port}")
            
            # Send ping and wait for response with timeout handling
            try:
                check_socket.send_json(message)
                response = check_socket.recv_json()
                
                # Update last successful communication time
                if response.get("success", False):
                    self.last_successful_communication = time.time()
                
                # Clean up socket
                check_socket.disconnect(f"tcp://{self.host}:{self.port}")
                check_socket.close()
                check_context.term()
                
                return True
            except:
                # Clean up socket on failure
                try:
                    check_socket.disconnect(f"tcp://{self.host}:{self.port}")
                    check_socket.close()
                    check_context.term()
                except:
                    pass
                return False
        except:
            return False

    def send_command(self, command, params=None):
        """Send a command to the bot and get response."""
        # Check for stuck connections (no successful communication for 60 seconds)
        current_time = time.time()
        if self.last_successful_communication > 0 and current_time - self.last_successful_communication > 60:
            logger.warning("No successful communication for 60 seconds, forcing socket reset")
            self.reset_connection()
        
        if not self.connected and not self.reconnect():
            return {"success": False, "error": "Not connected to bot"}
        
        with self.lock:
            try:
                # Prepare message
                message = {
                    "command": command,
                    "params": params or {},
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                logger.info(f"Sending command via ZMQ: {command}")
                
                # Use a dedicated socket for this request
                request_context = zmq.Context()
                request_socket = request_context.socket(zmq.REQ)
                
                # Set aggressive socket options
                request_socket.setsockopt(zmq.LINGER, 0)
                request_socket.setsockopt(zmq.RCVTIMEO, 4000)  # 4 second timeout (reduced from 6s)
                request_socket.setsockopt(zmq.SNDTIMEO, 2000)  # 2 second send timeout (reduced from 3s)
                request_socket.setsockopt(zmq.IMMEDIATE, 1)
                
                # Connect socket
                request_socket.connect(f"tcp://{self.host}:{self.port}")
                
                # Send message with timeout handling
                send_success = False
                try:
                    request_socket.send_json(message)
                    send_success = True
                except zmq.error.Again:
                    logger.error(f"Send timeout for command {command}")
                    # Fall back to file-based communication
                    success = save_command(command, params)
                    if success:
                        return {"success": True, "message": "Command sent via file-based communication (send timeout)"}
                    else:
                        return {"success": False, "error": "Send timeout and file-based fallback failed"}
                except Exception as e:
                    logger.error(f"Error sending command {command}: {e}")
                    send_success = False
                
                # If send was successful, wait for response
                if send_success:
                    try:
                        response = request_socket.recv_json()
                        logger.info(f"Received ZMQ response for {command}")
                        self.last_successful_communication = time.time()
                        
                        # Clean up socket
                        try:
                            request_socket.disconnect(f"tcp://{self.host}:{self.port}")
                            request_socket.close()
                            request_context.term()
                        except:
                            pass
                            
                        return response
                    except zmq.error.Again:
                        logger.error(f"Command {command} timed out waiting for response")
                        # Fall back to file-based communication
                        success = save_command(command, params)
                        if success:
                            return {"success": True, "message": "Command sent via file-based communication (receive timeout)"}
                        else:
                            return {"success": False, "error": "Command timed out and file-based fallback failed"}
                
                # Clean up socket if still here
                try:
                    request_socket.disconnect(f"tcp://{self.host}:{self.port}")
                    request_socket.close()
                    request_context.term()
                except:
                    pass
                    
                # If we got here, there was an error
                self.connected = False
                
                # Try file-based communication as fallback
                success = save_command(command, params)
                if success:
                    return {"success": True, "message": "Command sent via file-based communication after error"}
                else:
                    return {"success": False, "error": "Communication error and file-based fallback failed"}
                
            except Exception as e:
                logger.error(f"General error in send_command for {command}: {e}")
                self.connected = False
                
                # Try file-based communication as fallback
                success = save_command(command, params)
                if success:
                    return {"success": True, "message": "Command sent via file-based communication after error"}
                else:
                    return {"success": False, "error": str(e)}
                
    def reset_connection(self):
        """Reset the ZMQ connection completely."""
        logger.info("Resetting ZMQ connection")
        
        with self.lock:
            try:
                # Close existing socket
                try:
                    self.socket.disconnect(f"tcp://{self.host}:{self.port}")
                    self.socket.close()
                except Exception as e:
                    logger.warning(f"Error closing socket during reset: {e}")
                
                # Wait a moment
                time.sleep(0.5)
                
                # Terminate old context
                try:
                    self.context.term()
                except Exception as e:
                    logger.warning(f"Error terminating context during reset: {e}")
                
                # Small delay to ensure resources are released
                time.sleep(0.5)
                
                # Force garbage collection to clean up any lingering references
                import gc
                gc.collect()
                
                # Create a new context and socket
                self.context = zmq.Context()
                self.socket = self.context.socket(zmq.REQ)
                
                # Set aggressive socket options
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.setsockopt(zmq.RCVTIMEO, 8000)  # 8 second receive timeout
                self.socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second send timeout
                self.socket.setsockopt(zmq.IMMEDIATE, 1)    # Don't queue messages 
                self.socket.setsockopt(zmq.RECONNECT_IVL, 1000)  # Reconnect interval 1 second
                self.socket.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)  # Max reconnect interval 5 seconds
                
                # Connect to bot
                self.socket.connect(f"tcp://{self.host}:{self.port}")
                self.connected = True
                self.reconnect_attempt = 0
                
                logger.info(f"ZMQ connection reset successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to reset ZMQ connection: {e}")
                self.connected = False
                return False

    def disconnect(self):
        """Disconnect from the bot."""
        if self.connected:
            try:
                self.socket.disconnect(f"tcp://{self.host}:{self.port}")
                self.socket.close()
                self.context.term()
                self.connected = False
                logger.info("Disconnected from bot")
            except Exception as e:
                logger.error(f"Error disconnecting from bot: {e}")
    
    def reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        if self.connected:
            return True
            
        self.reconnect_attempt += 1
        # More aggressive backoff starting with 1s and capping at 20s
        backoff = min(20, 1 * self.reconnect_attempt)
        logger.info(f"Attempting to reconnect to bot (attempt {self.reconnect_attempt}, waiting {backoff}s)")
        
        try:
            time.sleep(backoff)
            success = self.reset_connection()
            
            # If successful, try a quick ping to verify
            if success:
                if self.check_connection():
                    logger.info("Reconnection verified with ping")
                    return True
                else:
                    logger.warning("Reconnection succeeded but ping failed")
                    # Try one more reset immediately
                    time.sleep(0.5)
                    return self.reset_connection()
            return success
        except Exception as e:
            logger.error(f"Error during reconnection: {e}")
            return False


# Create bot communication instance
bot_comm = BotCommunication()

# Load configs
def load_config():
    """Load configuration from file."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Config file {CONFIG_FILE} not found")
            return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

# Save configs
def save_config(config):
    """Save configuration to file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Config saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False

# Load shared data
def load_shared_data():
    """Load shared data from file."""
    try:
        if os.path.exists(SHARED_DATA_FILE):
            with FileLock(f"{SHARED_DATA_FILE}.lock"):
                with open(SHARED_DATA_FILE, 'rb') as f:
                    return pickle.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading shared data: {e}")
        return {}

# Save command
def save_command(command, params=None):
    """Save command to file for bot to pick up."""
    try:
        data = {
            "command": command,
            "params": params or {},
            "timestamp": datetime.datetime.now().isoformat(),
            "processed": False
        }
        
        # Ensure the file exists
        if not os.path.exists(COMMAND_FILE):
            with open(COMMAND_FILE, 'w') as f:
                json.dump({"commands": []}, f)
        
        # Add command to list
        with FileLock(f"{COMMAND_FILE}.lock"):
            with open(COMMAND_FILE, 'r') as f:
                commands = json.load(f)
            
            commands["commands"].append(data)
            
            with open(COMMAND_FILE, 'w') as f:
                json.dump(commands, f, indent=4)
        
        logger.info(f"Command saved: {command}")
        return True
    except Exception as e:
        logger.error(f"Error saving command: {e}")
        return False

# Check bot status
def check_bot_status():
    """Check if the bot is running and update status."""
    global bot_status
    
    # Try to communicate with bot through ZMQ
    response = bot_comm.send_command("ping")
    
    if response.get("success"):
        bot_status["status"] = response.get("status", "running")
        bot_status["last_update"] = datetime.datetime.now().isoformat()
        bot_status["uptime"] = response.get("uptime", 0)
        bot_status["version"] = response.get("version", "1.0.0")
        return True
    
    # Fall back to shared file method
    try:
        shared_data = load_shared_data()
        if shared_data and "status" in shared_data:
            last_update = shared_data.get("last_update")
            if last_update:
                # Check if update is within last minute
                last_datetime = datetime.datetime.fromisoformat(last_update)
                if (datetime.datetime.now() - last_datetime).total_seconds() < 60:
                    bot_status["status"] = shared_data["status"]
                    bot_status["last_update"] = last_update
                    bot_status["uptime"] = shared_data.get("uptime", 0)
                    bot_status["version"] = shared_data.get("version", "1.0.0")
                    return True
        
        # If we got here, bot is not responsive
        bot_status["status"] = "disconnected"
        return False
    except Exception as e:
        logger.error(f"Error checking bot status: {e}")
        bot_status["status"] = "disconnected"
        return False



# Background thread for updating data
def background_updater():
    """Background thread that updates data from the bot."""
    reconnect_delay = 2  # Start with 2 seconds (reduced from 5)
    max_reconnect_delay = 30  # Maximum 30 seconds (reduced from 60)
    last_successful_update = time.time()
    connection_check_interval = 30  # Check connection every 30 seconds
    last_connection_check = 0
    connection_reset_count = 0
        
    while True:
        try:
            # Periodic full connection check and reset
            current_time = time.time()
            global bot_comm
            
            # Check if we're stuck in a deadlock
            if current_time - last_successful_update > 30:  # No updates for 30 seconds (reduced from 45)
                logger.warning("Possible ZMQ deadlock detected. Resetting connection.")
                bot_comm.reset_connection()
                connection_reset_count += 1
                
                # If we've reset multiple times with no success, try more aggressive approach
                if connection_reset_count >= 3:
                    logger.warning("Multiple connection resets failed. Trying complete recreation.")
                    # Create a completely new connection instance
                    
                    old_bot_comm = bot_comm
                    bot_comm = BotCommunication()
                    
                    # Try to clean up old instance
                    try:
                        old_bot_comm.disconnect()
                    except:
                        pass
                    
                    # Reset counter
                    connection_reset_count = 0
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
            
            # Periodic connection check - more frequent checks
            if current_time - last_connection_check > 15:  # Check every 15 seconds (reduced from 30)
                if not bot_comm.check_connection():
                    logger.warning("Connection check failed. Attempting to reconnect.")
                    bot_comm.reconnect()
                last_connection_check = current_time
            
            # Check bot status
            status_ok = check_bot_status()
            
            # If status check succeeded, update the timestamp
            if status_ok:
                last_successful_update = time.time()
            
            # Send status update to clients
            socketio.emit('status_update', {
                'status': bot_status["status"],
                'last_update': bot_status["last_update"],
                'uptime': bot_status["uptime"],
                'version': bot_status["version"]
            })
            
            if status_ok:
                # Reset reconnect delay on success
                reconnect_delay = 5
                
                # Get latest data from bot
                try:
                    # Try simpler commands first to ensure connection is stable
                    if bot_comm.check_connection():
                        # Get portfolio summary
                        portfolio = bot_comm.send_command("get_portfolio")
                        if portfolio.get("success"):
                            socketio.emit('portfolio_update', portfolio.get("data", {}))
                        
                        # Get active positions
                        positions = bot_comm.send_command("get_positions")
                        if positions.get("success"):
                            socketio.emit('positions_update', positions.get("data", []))
                    
                        # Get recent signals
                        signals = bot_comm.send_command("get_signals")
                        if signals.get("success"):
                            socketio.emit('signals_update', signals.get("data", {}))
                        
                        # Get market regimes
                        regimes = bot_comm.send_command("get_market_regimes")
                        if regimes.get("success"):
                            socketio.emit('regimes_update', regimes.get("data", {}))
                        
                        # Get performance metrics
                        metrics = bot_comm.send_command("get_performance")
                        if metrics.get("success"):
                            socketio.emit('metrics_update', metrics.get("data", {}))
                        
                        # Get recent trades
                        trades = bot_comm.send_command("get_trades", {"limit": 10})
                        if trades.get("success"):
                            socketio.emit('trades_update', trades.get("data", []))
                        
                    else:
                        logger.warning("Bot connection check failed, skipping data updates")
                except Exception as e:
                    logger.error(f"Error fetching data from bot: {e}")
            else:
                # Increase reconnect delay on failure (exponential backoff)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                logger.warning(f"Bot not responsive, will retry in {reconnect_delay} seconds")
            
            # Sleep for a few seconds
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in background updater: {e}")
            time.sleep(reconnect_delay)  # Use backoff delay on error

# Generate sample data for testing
def generate_sample_data():
    """Generate sample data for testing the dashboard."""
    # Sample positions
    positions = [
        {
            "id": "pos1",
            "symbol": "BTC/USDT",
            "side": "buy",
            "entry_price": 65432.10,
            "amount": 0.15,
            "value": 9814.82,
            "stop_loss": 63500.00,
            "take_profit": 68000.00,
            "unrealized_pnl": 214.50,
            "pnl_percentage": 2.18,
            "entry_time": "2025-03-19T15:30:00",
            "duration": "1d 2h 15m"
        },
        {
            "id": "pos2",
            "symbol": "ETH/USDT",
            "side": "sell",
            "entry_price": 3250.75,
            "amount": 0.5,
            "value": 1625.38,
            "stop_loss": 3350.00,
            "take_profit": 3050.00,
            "unrealized_pnl": -45.25,
            "pnl_percentage": -1.39,
            "entry_time": "2025-03-20T09:45:00",
            "duration": "5h 30m"
        }
    ]
    
    # Sample portfolio
    portfolio = {
        "total_balance": 25463.92,
        "available_balance": 14023.72,
        "total_positions_value": 11440.20,
        "daily_pnl": 342.87,
        "daily_pnl_percentage": 1.36,
        "total_positions": 2,
        "long_positions": 1,
        "short_positions": 1,
        "win_rate": 65.2,
        "current_drawdown": 2.8
    }
    
    # Sample signals
    signals = {
        "BTC/USDT": {
            "signal": 0.75,
            "confidence": 0.82,
            "market_regime": "trending_up",
            "sentiment": 0.63,
            "timestamp": "2025-03-20T14:28:00",
            "patterns": ["double_bottom", "bullish_engulfing"]
        },
        "ETH/USDT": {
            "signal": -0.42,
            "confidence": 0.65,
            "market_regime": "ranging",
            "sentiment": -0.28,
            "timestamp": "2025-03-20T14:25:00",
            "patterns": ["head_and_shoulders"]
        },
        "SOL/USDT": {
            "signal": 0.38,
            "confidence": 0.58,
            "market_regime": "consolidating",
            "sentiment": 0.45,
            "timestamp": "2025-03-20T14:20:00",
            "patterns": ["ascending_triangle"]
        }
    }
    
    # Sample market regimes
    regimes = {
        "BTC/USDT": {
            "regime": "trending_up",
            "trend_score": 68.5,
            "volatility": 32.7,
            "since": "2025-03-18T10:15:00"
        },
        "ETH/USDT": {
            "regime": "ranging",
            "trend_score": 12.3,
            "volatility": 28.4,
            "since": "2025-03-19T14:30:00"
        },
        "SOL/USDT": {
            "regime": "consolidating",
            "trend_score": 8.7,
            "volatility": 18.2,
            "since": "2025-03-19T22:45:00"
        },
        "BNB/USDT": {
            "regime": "volatile",
            "trend_score": -5.2,
            "volatility": 76.9,
            "since": "2025-03-20T08:30:00"
        }
    }
    
    # Sample performance metrics
    metrics = {
        "total_trades": 124,
        "winning_trades": 81,
        "losing_trades": 43,
        "win_rate": 65.32,
        "avg_win": 2.87,
        "avg_loss": -1.56,
        "profit_factor": 3.24,
        "max_drawdown": 7.92,
        "sharpe_ratio": 1.87,
        "equity_data": [
            {"date": "2025-03-01", "equity": 20000.00},
            {"date": "2025-03-02", "equity": 20120.50},
            {"date": "2025-03-03", "equity": 20345.75},
            {"date": "2025-03-04", "equity": 20289.30},
            {"date": "2025-03-05", "equity": 20402.80},
            {"date": "2025-03-06", "equity": 20750.25},
            {"date": "2025-03-07", "equity": 21126.75},
            {"date": "2025-03-08", "equity": 21092.40},
            {"date": "2025-03-09", "equity": 21263.80},
            {"date": "2025-03-10", "equity": 21587.25},
            {"date": "2025-03-11", "equity": 21490.60},
            {"date": "2025-03-12", "equity": 21378.90},
            {"date": "2025-03-13", "equity": 21682.35},
            {"date": "2025-03-14", "equity": 22104.70},
            {"date": "2025-03-15", "equity": 22315.85},
            {"date": "2025-03-16", "equity": 22493.20},
            {"date": "2025-03-17", "equity": 22782.55},
            {"date": "2025-03-18", "equity": 23140.90},
            {"date": "2025-03-19", "equity": 24986.35},
            {"date": "2025-03-20", "equity": 25463.92}
        ]
    }
    
    # Sample trades
    trades = [
        {
            "id": "trade1",
            "symbol": "BTC/USDT",
            "side": "buy",
            "entry_price": 63250.75,
            "exit_price": 65480.20,
            "amount": 0.2,
            "pnl": 445.89,
            "pnl_percentage": 3.53,
            "entry_time": "2025-03-18T14:30:00",
            "exit_time": "2025-03-19T10:45:00",
            "duration": "20h 15m",
            "exit_reason": "take_profit"
        },
        {
            "id": "trade2",
            "symbol": "ETH/USDT",
            "side": "buy",
            "entry_price": 3180.50,
            "exit_price": 3350.25,
            "amount": 0.8,
            "pnl": 135.80,
            "pnl_percentage": 5.34,
            "entry_time": "2025-03-17T09:20:00",
            "exit_time": "2025-03-18T16:35:00",
            "duration": "1d 7h 15m",
            "exit_reason": "take_profit"
        },
        {
            "id": "trade3",
            "symbol": "SOL/USDT",
            "side": "sell",
            "entry_price": 142.75,
            "exit_price": 138.20,
            "amount": 10,
            "pnl": 45.50,
            "pnl_percentage": 3.19,
            "entry_time": "2025-03-18T11:05:00",
            "exit_time": "2025-03-19T08:50:00",
            "duration": "21h 45m",
            "exit_reason": "take_profit"
        },
        {
            "id": "trade4",
            "symbol": "BNB/USDT",
            "side": "buy",
            "entry_price": 590.25,
            "exit_price": 572.80,
            "amount": 1.5,
            "pnl": -26.18,
            "pnl_percentage": -2.95,
            "entry_time": "2025-03-19T13:40:00",
            "exit_time": "2025-03-19T19:20:00",
            "duration": "5h 40m",
            "exit_reason": "stop_loss"
        },
        {
            "id": "trade5",
            "symbol": "XRP/USDT",
            "side": "buy",
            "entry_price": 0.6580,
            "exit_price": 0.6740,
            "amount": 2000,
            "pnl": 32.00,
            "pnl_percentage": 2.43,
            "entry_time": "2025-03-19T16:15:00",
            "exit_time": "2025-03-20T11:30:00",
            "duration": "19h 15m",
            "exit_reason": "manual"
        }
    ]
    
    return {
        "positions": positions,
        "portfolio": portfolio,
        "signals": signals,
        "regimes": regimes,
        "metrics": metrics,
        "trades": trades
    }

# Flask routes
@app.route('/')
def index():
    """Main dashboard page."""
    # Check if we're connected to the bot
    check_bot_status()
    
    # If config doesn't exist, create sample config
    if not os.path.exists(CONFIG_FILE):
        create_sample_config()
    
    return render_template('index.html', bot_status=bot_status)

@app.route('/dashboard')
def dashboard():
    """Main dashboard overview."""
    sample_data = generate_sample_data()
    return render_template(
        'dashboard.html',
        bot_status=bot_status,
        portfolio=sample_data["portfolio"],
        positions=sample_data["positions"],
        signals=sample_data["signals"]
    )

@app.route('/positions')
def positions():
    """Positions page."""
    sample_data = generate_sample_data()
    return render_template(
        'positions.html',
        bot_status=bot_status,
        positions=sample_data["positions"]
    )

@app.route('/performance')
def performance():
    """Performance and analytics page."""
    sample_data = generate_sample_data()
    return render_template(
        'performance.html',
        bot_status=bot_status,
        metrics=sample_data["metrics"]
    )

@app.route('/markets')
def markets():
    """Market analysis page."""
    sample_data = generate_sample_data()
    return render_template(
        'markets.html',
        bot_status=bot_status,
        regimes=sample_data["regimes"],
        signals=sample_data["signals"]
    )

@app.route('/config')
def config():
    """Configuration page."""
    # Load config
    config_data = load_config()
    return render_template('config.html', bot_status=bot_status, config=config_data)

@app.route('/logs')
def logs():
    """Logs and history page."""
    sample_data = generate_sample_data()
    return render_template(
        'logs.html',
        bot_status=bot_status,
        trades=sample_data["trades"]
    )

# API routes
@app.route('/api/status')
def api_status():
    """Get bot status."""
    check_bot_status()
    return jsonify(bot_status)

@app.route('/api/portfolio')
def api_portfolio():
    """Get portfolio data."""
    # First try to get from bot
    response = bot_comm.send_command("get_portfolio")
    if response.get("success"):
        return jsonify(response.get("data", {}))
        
    # Fall back to sample data
    sample_data = generate_sample_data()
    return jsonify(sample_data["portfolio"])

@app.route('/api/positions')
def api_positions():
    """Get positions data."""
    # First try to get from bot
    response = bot_comm.send_command("get_positions")
    if response.get("success"):
        return jsonify(response.get("data", []))
        
    # Fall back to sample data
    sample_data = generate_sample_data()
    return jsonify(sample_data["positions"])

@app.route('/api/signals')
def api_signals():
    """Get trading signals."""
    # First try to get from bot
    response = bot_comm.send_command("get_signals")
    if response.get("success"):
        return jsonify(response.get("data", {}))
        
    # Fall back to sample data
    sample_data = generate_sample_data()
    return jsonify(sample_data["signals"])

@app.route('/api/regimes')
def api_regimes():
    """Get market regimes."""
    # First try to get from bot
    response = bot_comm.send_command("get_market_regimes")
    if response.get("success"):
        return jsonify(response.get("data", {}))
        
    # Fall back to sample data
    sample_data = generate_sample_data()
    return jsonify(sample_data["regimes"])

@app.route('/api/metrics')
def api_metrics():
    """Get performance metrics."""
    # First try to get from bot
    response = bot_comm.send_command("get_performance")
    if response.get("success"):
        return jsonify(response.get("data", {}))
        
    # Fall back to sample data
    sample_data = generate_sample_data()
    return jsonify(sample_data["metrics"])

@app.route('/api/trades')
def api_trades():
    """Get recent trades."""
    # First try to get from bot
    limit = request.args.get('limit', 10, type=int)
    response = bot_comm.send_command("get_trades", {"limit": limit})
    if response.get("success"):
        return jsonify(response.get("data", []))
        
    # Fall back to sample data
    sample_data = generate_sample_data()
    return jsonify(sample_data["trades"])

@app.route('/api/command', methods=['POST'])
def api_command():
    """Send command to bot."""
    data = request.json
    if not data or 'command' not in data:
        return jsonify({"success": False, "error": "Invalid command format"})
    
    command = data['command']
    params = data.get('params', {})
    
    # First try direct communication
    response = bot_comm.send_command(command, params)
    if response.get("success"):
        return jsonify(response)
    
    # Fall back to file-based communication
    success = save_command(command, params)
    if success:
        return jsonify({"success": True, "message": "Command sent"})
    else:
        return jsonify({"success": False, "error": "Failed to send command"})

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Get or update configuration."""
    if request.method == 'GET':
        config_data = load_config()
        return jsonify(config_data)
    else:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "Invalid config data"})
        
        # First try direct communication
        response = bot_comm.send_command("update_config", {"config": data})
        if response.get("success"):
            # Also update local config
            save_config(data)
            return jsonify(response)
        
        # Fall back to file-based communication
        success = save_config(data)
        if success:
            # Also send command to bot
            save_command("update_config", {"config": data})
            return jsonify({"success": True, "message": "Configuration updated"})
        else:
            return jsonify({"success": False, "error": "Failed to update configuration"})

@app.route('/api/chart/price/<symbol>/<timeframe>')
def api_chart_price(symbol, timeframe):
    """Get price chart data for a symbol and timeframe."""
    # Try to get from bot
    response = bot_comm.send_command("get_price_data", {"symbol": symbol, "timeframe": timeframe})
    if response.get("success"):
        # Create chart
        chart_data = create_price_chart(response.get("data", {}))
        return jsonify(chart_data)
    
    # Generate sample chart
    chart_data = create_sample_price_chart(symbol, timeframe)
    return jsonify(chart_data)

@app.route('/api/chart/equity')
def api_chart_equity():
    """Get equity chart data."""
    # Try to get from bot
    response = bot_comm.send_command("get_performance")
    if response.get("success") and "equity_data" in response.get("data", {}):
        # Create chart
        chart_data = create_equity_chart(response["data"]["equity_data"])
        return jsonify(chart_data)
    
    # Generate sample chart
    sample_data = generate_sample_data()
    chart_data = create_equity_chart(sample_data["metrics"]["equity_data"])
    return jsonify(chart_data)

@app.route('/api/bot/start')
def api_bot_start():
    """Start the trading bot."""
    # Try direct communication
    response = bot_comm.send_command("start")
    if response.get("success"):
        return jsonify(response)
    
    # Fall back to file-based command
    success = save_command("start")
    if success:
        return jsonify({"success": True, "message": "Start command sent"})
    else:
        return jsonify({"success": False, "error": "Failed to send start command"})

@app.route('/api/bot/stop')
def api_bot_stop():
    """Stop the trading bot."""
    # Try direct communication
    response = bot_comm.send_command("stop")
    if response.get("success"):
        return jsonify(response)
    
    # Fall back to file-based command
    success = save_command("stop")
    if success:
        return jsonify({"success": True, "message": "Stop command sent"})
    else:
        return jsonify({"success": False, "error": "Failed to send stop command"})

@app.route('/api/bot/pause')
def api_bot_pause():
    """Pause the trading bot."""
    # Try direct communication
    response = bot_comm.send_command("pause")
    if response.get("success"):
        return jsonify(response)
    
    # Fall back to file-based command
    success = save_command("pause")
    if success:
        return jsonify({"success": True, "message": "Pause command sent"})
    else:
        return jsonify({"success": False, "error": "Failed to send pause command"})

@app.route('/api/position/close/<position_id>')
def api_position_close(position_id):
    """Close a position."""
    # Try direct communication
    response = bot_comm.send_command("close_position", {"position_id": position_id})
    if response.get("success"):
        return jsonify(response)
    
    # Fall back to file-based command
    success = save_command("close_position", {"position_id": position_id})
    if success:
        return jsonify({"success": True, "message": f"Close position command sent for {position_id}"})
    else:
        return jsonify({"success": False, "error": f"Failed to send close position command for {position_id}"})

@app.route('/api/position/update', methods=['POST'])
def api_position_update():
    """Update a position."""
    data = request.json
    if not data or 'position_id' not in data:
        return jsonify({"success": False, "error": "Invalid position data"})
    
    # Try direct communication
    response = bot_comm.send_command("update_position", data)
    if response.get("success"):
        return jsonify(response)
    
    # Fall back to file-based command
    success = save_command("update_position", data)
    if success:
        return jsonify({"success": True, "message": "Update position command sent"})
    else:
        return jsonify({"success": False, "error": "Failed to send update position command"})

@app.route('/api/position/create', methods=['POST'])
def api_position_create():
    """Create a new position."""
    data = request.json
    if not data or 'symbol' not in data or 'side' not in data or 'amount' not in data:
        return jsonify({"success": False, "error": "Invalid position data"})
    
    # Try direct communication
    response = bot_comm.send_command("create_position", data)
    if response.get("success"):
        return jsonify(response)
    
    # Fall back to file-based command
    success = save_command("create_position", data)
    if success:
        return jsonify({"success": True, "message": "Create position command sent"})
    else:
        return jsonify({"success": False, "error": "Failed to send create position command"})

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected")
    
    # Check bot status and send to client
    check_bot_status()
    emit('status_update', {
        'status': bot_status["status"],
        'last_update': bot_status["last_update"],
        'uptime': bot_status["uptime"],
        'version': bot_status["version"]
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")

@socketio.on('command')
def handle_command(data):
    """Handle command from client."""
    if not data or 'command' not in data:
        emit('command_response', {"success": False, "error": "Invalid command format"})
        return
    
    command = data['command']
    params = data.get('params', {})
    
    logger.info(f"Received command: {command} with params: {params}")
    
    # Try direct communication
    response = bot_comm.send_command(command, params)
    if response.get("success"):
        emit('command_response', response)
        return
    
    # Fall back to file-based command
    success = save_command(command, params)
    if success:
        emit('command_response', {"success": True, "message": "Command sent"})
    else:
        emit('command_response', {"success": False, "error": "Failed to send command"})

# Helper functions
def create_sample_price_chart(symbol, timeframe):
    """Create a sample price chart for testing."""
    # Generate sample price data
    now = datetime.datetime.now()
    dates = [now - datetime.timedelta(minutes=i * 15) for i in range(100)]
    dates.reverse()
    
    # Generate realistic price movement
    base_price = 65000 if 'BTC' in symbol else 3500 if 'ETH' in symbol else 150 if 'SOL' in symbol else 500
    price_data = []
    last_price = base_price * (0.95 + 0.1 * np.random.random())
    
    for i in range(len(dates)):
        # Random walk with some trend and volatility
        change = last_price * (0.015 * np.random.randn() + 0.002)
        
        # Add some mean reversion
        mean_reversion = (base_price - last_price) * 0.02
        change += mean_reversion
        
        high = last_price + abs(change) * (0.5 + 0.5 * np.random.random())
        low = last_price - abs(change) * (0.5 + 0.5 * np.random.random())
        
        if change > 0:
            open_price = low * (0.2 + 0.8 * np.random.random()) + high * (0.8 * np.random.random())
            close = last_price + change
        else:
            open_price = high * (0.2 + 0.8 * np.random.random()) + low * (0.8 * np.random.random())
            close = last_price + change
        
        price_data.append({
            'timestamp': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': round(base_price * 10 * (0.5 + np.random.random()), 2)
        })
        
        last_price = close
    
    # Create chart data
    chart_data = {
        'symbol': symbol,
        'timeframe': timeframe,
        'data': price_data,
        'layout': {
            'title': f'{symbol} - {timeframe} Chart',
            'xaxis': {
                'title': 'Time',
                'rangeslider': {'visible': False}
            },
            'yaxis': {
                'title': 'Price',
                'autorange': True
            }
        }
    }
    
    return chart_data

def create_price_chart(price_data):
    """Create a price chart from real data."""
    if not price_data or 'data' not in price_data:
        return {}
    
    # Create chart data
    chart_data = {
        'symbol': price_data.get('symbol', ''),
        'timeframe': price_data.get('timeframe', ''),
        'data': price_data['data'],
        'layout': {
            'title': f"{price_data.get('symbol', '')} - {price_data.get('timeframe', '')} Chart",
            'xaxis': {
                'title': 'Time',
                'rangeslider': {'visible': False}
            },
            'yaxis': {
                'title': 'Price',
                'autorange': True
            }
        }
    }
    
    return chart_data

def create_equity_chart(equity_data):
    """Create an equity chart from performance data."""
    if not equity_data:
        return {}
    
    # Parse dates and values
    dates = [item['date'] for item in equity_data]
    values = [item['equity'] for item in equity_data]
    
    # Create chart data
    chart_data = {
        'data': [
            {
                'x': dates,
                'y': values,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Equity',
                'line': {'color': '#17A2B8', 'width': 2}
            }
        ],
        'layout': {
            'title': 'Account Equity Curve',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Equity Value'},
            'showlegend': True
        }
    }
    
    return chart_data

def create_sample_config():
    """Create a sample configuration file for testing."""
    config_data = {
        "api_key": "your_api_key_here",
        "api_secret": "your_api_secret_here",
        "base_url": "https://api.bybit.com",
        "testnet": False,
        "symbols": [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "XRP/USDT",
            "SOL/USDT", 
            "DOGE/USDT"
        ],
        "timeframes": {
            "primary": "15m",
            "secondary": [
                "1h",
                "4h",
                "1d"
            ]
        },
        "trade_settings": {
            "max_positions": 3,
            "account_risk_per_trade": 0.02,
            "default_stop_loss_pct": 0.02,
            "default_take_profit_pct": 0.06,
            "trailing_stop": True,
            "trailing_stop_activation": 0.02,
            "trailing_stop_distance": 0.01
        },
        "ml_settings": {
            "feature_window": 50,
            "prediction_window": 12,
            "train_interval_days": 7,
            "confidence_threshold": 0.50,
            "ensemble_method": "weighted",
            "model_weights": {
                "xgboost": 0.25,
                "lstm": 0.35,
                "catboost": 0.25,
                "transformer": 0.15
            }
        },
        "risk_management": {
            "max_daily_loss": 0.05,
            "max_drawdown": 0.15,
            "position_correlation_limit": 0.7
        },
        "sentiment_analysis": {
            "enabled": True,
            "sources": [
                "twitter",
                "reddit",
                "news"
            ],
            "refresh_interval_minutes": 30,
            "impact_weight": 0.2
        },
        "execution": {
            "order_type": "limit",
            "limit_order_distance": 0.001,
            "retry_attempts": 3,
            "retry_delay": 1,
            "execution_cooldown": 60
        },
        "logging": {
            "level": "INFO",
            "save_predictions": True,
            "detailed_trade_logs": True
        }
    }
    
    save_config(config_data)

# Main entry point
if __name__ == "__main__":
    # Create templates and static directories if they don't exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Create sample data directories
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Try to connect to the bot
    bot_comm.connect()
    
    # Start background thread for updates
    global update_thread
    update_thread = threading.Thread(target=background_updater)
    update_thread.daemon = True
    update_thread.start()
    
    
    # Add watchdog thread to monitor and restart the update thread if it fails
    def watchdog():
        """Monitor the update thread and restart it if it fails."""
        while True:
            if not update_thread.is_alive():
                logger.warning("Update thread died, restarting...")
                
                
                # Start a new update thread
                update_thread = threading.Thread(target=background_updater)
                update_thread.daemon = True
                update_thread.start()
                logger.info("Update thread restarted")
                
            # Check every 30 seconds
            time.sleep(30)
    
    # Start watchdog thread
    watchdog_thread = threading.Thread(target=watchdog)
    watchdog_thread.daemon = True
    watchdog_thread.start()
    

    # Start the server
    # Start the server
    try:
        logger.info(f"Starting dashboard on {DEFAULT_HOST}:{DEFAULT_PORT}")
        socketio.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")