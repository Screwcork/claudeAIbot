#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading Bot Dashboard Interface Adapter
--------------------------------------
Add this module to your trading bot to enable communication with the dashboard.
"""

import os
import sys
import json
import time
import pickle
import threading
import datetime
import logging
from pathlib import Path
from filelock import FileLock
import zmq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_interface.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BotInterface")

# Constants
DATA_DIR = "dashboard_data"
SHARED_DATA_FILE = os.path.join(DATA_DIR, "shared_data.pkl")
COMMAND_FILE = os.path.join(DATA_DIR, "commands.json")
ZMQ_PORT = 5555  # ZeroMQ port for dashboard communication
ZMQ_URL = "tcp://127.0.0.1:5555"
SOCKET_TIMEOUT_MS = 3000  # 3 seconds socket timeout (reduced from 5)
COMMAND_TIMEOUT_S = 10     # 10 seconds command timeout (reduced from 15)
HEALTH_CHECK_INTERVAL_S = 10  # Health check every 10 seconds (reduced from 15)
SOCKET_RESET_INTERVAL_S = 1800  # Reset socket every 30 minutes (reduced from 1 hour)
MAX_CONSECUTIVE_ERRORS = 2  # Reset socket after this many consecutive errors (reduced from 3)
# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

class DashboardInterface:
    """Interface for communication between the trading bot and dashboard."""
    
    def __init__(self, trading_bot):
        """Initialize interface with reference to main trading bot."""
        self.trading_bot = trading_bot
        self.running = False
        self.context = None
        self.socket = None
        self.last_socket_reset = time.time()
        self.consecutive_errors = 0
        self.start_time = datetime.datetime.now()
        
        # Initialize ZeroMQ server
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        
        # Initialize command processing
        self.command_threads = {}  # Track command execution threads
        self.command_events = {}   # Events for command completion signaling
        self.command_results = {}  # Store command results
    
    def _create_socket(self):
        """Create and initialize the ZMQ socket."""
        try:
            # Clean up any existing socket first
            if hasattr(self, 'socket') and self.socket:
                try:
                    self.socket.close()
                except Exception as e:
                    logger.warning(f"Error closing existing socket: {e}")
            
            # Create a new context and socket
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
            
            # Set socket options
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.RCVTIMEO, SOCKET_TIMEOUT_MS)
            self.socket.setsockopt(zmq.SNDTIMEO, SOCKET_TIMEOUT_MS)
            
            # Bind to address
            self.socket.bind(ZMQ_URL)
            self.last_socket_reset = time.time()
            self.consecutive_errors = 0
            
            logger.info(f"ZMQ socket created and bound to {ZMQ_URL}")
            return True
        except Exception as e:
            logger.error(f"Error creating socket: {e}")
            return False
    
    def _cleanup_socket(self):
        """Clean up socket resources."""
        try:
            if hasattr(self, 'socket') and self.socket:
                self.socket.close()
                
            if hasattr(self, 'context') and self.context:
                self.context.term()
                
            logger.info("Socket resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up socket: {e}")
    
    def start(self):
        """Start the dashboard interface."""
        if self.running:
            logger.warning("Dashboard already running")
            return
        
        try:
            logger.info(f"Starting ZMQ server on {ZMQ_URL}")
            self._create_socket()
            
            # Start command thread
            self.command_thread = threading.Thread(target=self._process_commands)
            self.command_thread.daemon = True
            self.command_thread.start()
            logger.info("Command thread started")
            
            # Start file processor
            self.file_thread = threading.Thread(target=self._process_file_commands)
            self.file_thread.daemon = True
            self.file_thread.start()
            logger.info("File processing thread started")
            
            # Start health check
            self.health_thread = threading.Thread(target=self._health_check)
            self.health_thread.daemon = True
            self.health_thread.start()
            logger.info("Health check thread started")
            
            self.running = True
            logger.info("Dashboard interface running")
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            self._cleanup_socket()
           

    def _health_check(self):
        """Periodic health check for socket and connection health."""
        while self.running:
            try:
                # Verify socket is working
                if self.socket is None:
                    logger.warning("Socket is None, recreating")
                    self._create_socket()
                    continue
                
                # Test socket with a ping command
                try:
                    # Set a shorter timeout for test messages
                    original_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
                    self.socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout for health check
                    
                    # Create test message
                    test_msg = {"command": "health_check", "timestamp": time.time()}
                    test_bytes = json.dumps(test_msg).encode('utf-8')
                    
                    # Only do test send/recv in debug mode to avoid potential socket issues
                    if logger.level <= logging.DEBUG:
                        logger.debug("Testing socket with health check message")
                        self.socket.send(test_bytes)
                        _ = self.socket.recv()
                        logger.debug("Socket health check passed")
                    
                    # Restore original timeout
                    self.socket.setsockopt(zmq.RCVTIMEO, original_timeout)
                    
                except Exception as e:
                    logger.warning(f"Socket health check failed: {e}")
                    self.consecutive_errors += 1
                    
                    # If multiple health checks fail, recreate the socket
                    if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.warning("Health check failures exceeded threshold, recreating socket")
                        self._create_socket()
                    
                # Update shared data file with current state
                self._update_shared_data()
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                self.consecutive_errors += 1
                
                # On serious health check errors, force socket recreation
                if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.warning("Multiple health check errors, performing full reset")
                    self._create_socket()
            
            # Run every HEALTH_CHECK_INTERVAL_S seconds (15)
            time.sleep(HEALTH_CHECK_INTERVAL_S)

    
    def stop(self):
        """Stop the dashboard interface."""
        if not self.running:
            return
        
        try:
            logger.info("Stopping dashboard interface...")
            self.running = False
            
            # Wait for threads to finish (max 5 seconds)
            time.sleep(0.5)
            
            # Close socket and context
            self._cleanup_socket()
            
            # Cancel all pending commands
            for cmd_id, event in self.command_events.items():
                event.set()
            
            logger.info("Dashboard interface stopped")
        except Exception as e:
            logger.error(f"Error stopping dashboard: {e}")

    def _update_shared_data(self):
        """Update shared data file with current state."""
        try:
            # Basic state data
            data = {
                "status": "running" if self.running else "stopped",
                "last_update": datetime.datetime.now().isoformat(),
                "last_socket_reset": datetime.datetime.fromtimestamp(self.last_socket_reset).isoformat(),
                "consecutive_errors": self.consecutive_errors,
                "version": "1.0.1"  # Increment version to track file updates
            }
            
            # Add trading bot data if available
            if hasattr(self.trading_bot, 'trading_signals'):
                data["trading_signals"] = self.trading_bot.trading_signals
            
            if hasattr(self.trading_bot, 'current_prices'):
                data["current_prices"] = self.trading_bot.current_prices
            
            if hasattr(self.trading_bot, 'risk_manager'):
                data["risk_stats"] = {
                    "daily_stats": self.trading_bot.risk_manager.daily_stats,
                    "current_drawdown": getattr(self.trading_bot.risk_manager, 'current_drawdown', 0)
                }
            
            # Write to file using pickle for better compatibility with complex objects
            with open(SHARED_DATA_FILE, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Error updating shared data: {e}")

    def _process_commands(self):
        """Process commands from ZMQ socket."""
        logger.info("Starting command processing loop")
        
        while self.running:
            try:
                # Check if socket needs scheduled reset
                if time.time() - self.last_socket_reset > SOCKET_RESET_INTERVAL_S:
                    logger.info("Performing scheduled socket reset")
                    self._create_socket()
                
                # Try to receive a message
                try:
                    if not self.socket:
                        logger.warning("Socket is None, recreating...")
                        if not self._create_socket():
                            time.sleep(1)
                            continue
                    
                    message_bytes = self.socket.recv()
                    if not message_bytes:
                        continue
                    
                    message = json.loads(message_bytes.decode('utf-8'))
                    command = message.get("command", "unknown")
                    params = message.get("params", {})
                    command_id = message.get("id", str(time.time()))
                    
                    logger.info(f"Received command: {command} (ID: {command_id})")
                    
                    # Create an event for this command
                    self.command_events[command_id] = threading.Event()
                    
                    # Execute command in a separate thread with timeout
                    cmd_thread = threading.Thread(
                        target=self._execute_command_with_timeout,
                        args=(command, params, command_id)
                    )
                    cmd_thread.daemon = True
                    cmd_thread.start()
                    self.command_threads[command_id] = cmd_thread
                    
                    # Wait for command completion or timeout
                    if not self.command_events[command_id].wait(COMMAND_TIMEOUT_S):
                        logger.warning(f"Command {command} (ID: {command_id}) timed out")
                        result = {"success": False, "error": f"Command timed out after {COMMAND_TIMEOUT_S} seconds"}
                    else:
                        # Get result (or default error if somehow missing)
                        result = self.command_results.get(
                            command_id, 
                            {"success": False, "error": "Command execution failed"}
                        )
                    
                    # Clean up command resources
                    if command_id in self.command_events:
                        del self.command_events[command_id]
                    if command_id in self.command_threads:
                        del self.command_threads[command_id]
                    if command_id in self.command_results:
                        del self.command_results[command_id]
                    
                    # Send response
                    try:
                        response = json.dumps(result).encode('utf-8')
                        self.socket.send(response)
                        logger.info(f"Sent response for {command} (ID: {command_id})")
                        self.consecutive_errors = 0  # Reset error counter on success
                    except Exception as send_err:
                        logger.error(f"Error sending response: {send_err}")
                        self.consecutive_errors += 1
                        if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            logger.warning(f"Too many consecutive errors ({self.consecutive_errors}), recreating socket")
                            self._create_socket()
                    
                except zmq.Again:
                    # This is normal - just no commands received
                    pass
                except Exception as e:
                    logger.error(f"Error in command processing: {e}")
                    self.consecutive_errors += 1
                    
                    # More aggressive socket reset on errors
                    if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.warning(f"Too many consecutive errors ({self.consecutive_errors}), recreating socket")
                        self._create_socket()
                    else:
                        # Brief pause to avoid rapid failure cycles
                        time.sleep(0.1)
            except Exception as e:
                logger.error(f"Serious error in command loop: {e}")
                self.consecutive_errors += 1
                time.sleep(1)
                
                # Complete socket and context recreation for serious errors
                if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.warning("Multiple serious errors, performing full ZMQ reset")
                    self._create_socket()
        
        logger.info("Command processing loop ended")
    

    def _process_commands_from_file(self):
        """Process commands from the command file."""
        while self.running:
            try:
                if os.path.exists(COMMAND_FILE):
                    with FileLock(f"{COMMAND_FILE}.lock"):
                        with open(COMMAND_FILE, 'r') as f:
                            commands = json.load(f)
                        
                        # Process unprocessed commands
                        for i, cmd in enumerate(commands.get("commands", [])):
                            if not cmd.get("processed", False):
                                self._execute_command(cmd["command"], cmd.get("params", {}))
                                commands["commands"][i]["processed"] = True
                        
                        # Write updated commands back to file
                        with open(COMMAND_FILE, 'w') as f:
                            json.dump(commands, f, indent=4)
            except Exception as e:
                logger.error(f"Error processing commands from file: {e}")
            
            # Update shared data and sleep
            self._update_shared_data()
            time.sleep(1)
    
    def _process_zmq_commands(self):
        """Process commands from ZeroMQ socket."""
        last_reset_time = time.time()
        reset_interval = 3600  # Reset socket every hour even if no errors
        
        while self.running:
            try:
                # Periodically reset socket to prevent stale states
                current_time = time.time()
                if current_time - last_reset_time > reset_interval:
                    logger.info("Performing scheduled ZMQ socket reset")
                    self._reset_zmq_socket()
                    last_reset_time = current_time
                
                # Wait for message with a short timeout
                if self.socket.poll(500) == 0:  # 500ms poll timeout - check more frequently
                    continue
                
                # Use short timeouts for socket operations to prevent hanging
                message = None
                message_received = False
                command = "unknown"
                params = {}
                
                # Receive message with explicit timeout handling
                try:
                    # Set shorter timeout for receiving
                    self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout for receiving
                    message = self.socket.recv_json()
                    message_received = True
                    command = message.get("command", "unknown")
                    params = message.get("params", {})
                    logger.info(f"ZMQ request received: {command}")
                except zmq.error.Again:
                    logger.warning("ZMQ receive timed out, resetting socket")
                    self._reset_zmq_socket()
                    last_reset_time = time.time()
                    continue
                except Exception as e:
                    logger.error(f"Error receiving ZMQ message: {e}")
                    self._reset_zmq_socket()
                    last_reset_time = time.time()
                    continue
                
                # Prepare a response, which we'll always try to send
                result = {"success": False, "error": "Unknown error in command processing"}
                
                # Execute command and prepare response if we received a message
                if message_received:
                    try:
                        # Set a timeout for command execution by running in a thread
                        self._command_result = None
                        command_thread = threading.Thread(
                            target=self._execute_command_with_timeout,
                            args=(command, params)
                        )
                        command_thread.daemon = True
                        command_thread.start()
                        
                        # Wait for command to complete with timeout
                        command_thread.join(timeout=10)  # 10 second timeout for command execution
                        
                        # Check if command completed in time
                        if hasattr(self, '_command_result'):
                            result = self._command_result
                            delattr(self, '_command_result')
                        else:
                            logger.warning(f"Command {command} timed out during execution")
                            result = {"success": False, "error": f"Command {command} timed out"}
                    except Exception as e:
                        logger.error(f"Error executing command {command}: {e}")
                        result = {"success": False, "error": str(e)}
                
                # Always try to send a response
                try:
                    # Set shorter timeout for sending
                    self.socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout for sending
                    self.socket.send_json(result)
                    logger.info(f"ZMQ response sent for {command}")
                except zmq.error.Again:
                    logger.warning(f"ZMQ send timed out for {command}, resetting socket")
                    self._reset_zmq_socket()
                    last_reset_time = time.time()
                except Exception as e:
                    logger.error(f"Failed to send ZMQ response: {e}")
                    self._reset_zmq_socket()
                    last_reset_time = time.time()
            
            except zmq.error.ZMQError as e:
                logger.error(f"ZeroMQ error in command processing: {e}")
                self._reset_zmq_socket()
                last_reset_time = time.time()
            except Exception as e:
                logger.error(f"General error in ZMQ command loop: {e}")
                self._reset_zmq_socket()
                last_reset_time = time.time()
    
    def _execute_command_with_timeout(self, command, params):
        """Execute command and store the result for the main thread to access."""
        try:
            self._command_result = self._execute_command(command, params)
        except Exception as e:
            logger.error(f"Error in threaded command execution for {command}: {e}")
            self._command_result = {"success": False, "error": str(e)}
    
    def _reset_zmq_socket(self):
        """Reset the ZMQ socket to recover from errors."""
        try:
            # Close the existing socket
            try:
                self.socket.close()
            except:
                pass
            
            # Create a new socket
            try:
                self.socket = self.context.socket(zmq.REP)
                # Set socket options for better reliability
                self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
                self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5s receive timeout
                self.socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5s send timeout
                self.socket.bind(f"tcp://*:{ZMQ_PORT}")
                logger.info("ZMQ socket reset successfully")
            except Exception as e:
                logger.error(f"Failed to create new ZMQ socket: {e}")
                # Add small delay to avoid CPU spinning on repeated failures
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error resetting ZMQ socket: {e}")
            time.sleep(1)

    def _check_health(self):
        """Check if the ZMQ server is healthy and restart if needed."""
        if not self.running:
            return
        
        try:
            # Check if ZMQ thread is alive
            if not self.zmq_thread or not self.zmq_thread.is_alive():
                logger.warning("ZMQ thread is not running, restarting...")
                
                # Restart ZMQ thread
                self.zmq_thread = threading.Thread(target=self._process_zmq_commands)
                self.zmq_thread.daemon = True
                self.zmq_thread.start()
                
                logger.info("ZMQ thread restarted")
            
            # Check if command thread is alive
            if not self.command_thread or not self.command_thread.is_alive():
                logger.warning("Command thread is not running, restarting...")
                
                # Restart command thread
                self.command_thread = threading.Thread(target=self._process_commands_from_file)
                self.command_thread.daemon = True
                self.command_thread.start()
                
                logger.info("Command thread restarted")
                
            # Test socket responsiveness with a lightweight ping
            try:
                # Create a test context and socket
                test_context = zmq.Context()
                test_socket = test_context.socket(zmq.REQ)
                test_socket.setsockopt(zmq.LINGER, 0)
                test_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2s timeout
                test_socket.setsockopt(zmq.SNDTIMEO, 2000)  # 2s timeout
                test_socket.connect(f"tcp://localhost:{ZMQ_PORT}")
                
                # Send a simple ping message
                test_socket.send_json({"command": "protocol_version", "params": {}})
                
                # Wait for response
                response = test_socket.recv_json()
                
                # Close test socket
                test_socket.close()
                test_context.term()
                
                logger.debug("ZMQ socket health check passed")
            except Exception as socket_e:
                logger.warning(f"ZMQ socket health check failed: {socket_e}, resetting socket")
                self._reset_zmq_socket()
                
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            
    def _execute_command(self, command, params):
        """Execute a command and return result."""
        logger.info(f"Executing: {command} with params: {params}")
        
        try:
            # Commands related to bot control
            if command == "ping":
                return {"success": True, "message": "pong", "timestamp": time.time()}
            
            elif command == "health_check":
                # Internal health check command
                return {"success": True, "status": "healthy", "timestamp": time.time()}
            
            elif command == "get_status":
                status = {
                    "running": getattr(self.trading_bot, 'is_running', False),
                    "uptime": (time.time() - getattr(self.trading_bot, 'start_time', time.time())),
                    "market_hours": getattr(self.trading_bot, 'market_hours', False),
                    "positions": len(getattr(getattr(self.trading_bot, 'trade_manager', {}), 'active_positions', [])),
                    "adapter_uptime": (time.time() - getattr(self, 'last_socket_reset', time.time())),
                    "consecutive_errors": self.consecutive_errors
                }
                return {"success": True, "status": status}
            
            elif command == "start_bot":
                if not getattr(self.trading_bot, 'is_running', False):
                    # Start the bot in a separate thread
                    threading.Thread(target=self.trading_bot.start).start()
                    return {"success": True, "message": "Bot started"}
                else:
                    return {"success": False, "error": "Bot is already running"}
            
            elif command == "stop_bot":
                if getattr(self.trading_bot, 'is_running', False):
                    self.trading_bot.stop()
                    return {"success": True, "message": "Bot stopped"}
                else:
                    return {"success": False, "error": "Bot is not running"}
            
            # Commands related to data
            elif command == "get_positions":
                if hasattr(self.trading_bot, 'trade_manager'):
                    positions = self.trading_bot.trade_manager.get_position_summary()
                    return {"success": True, "positions": positions}
                else:
                    return {"success": False, "error": "Trade manager not available"}
            
            elif command == "get_signals":
                if hasattr(self.trading_bot, 'trading_signals'):
                    return {"success": True, "signals": self.trading_bot.trading_signals}
                else:
                    return {"success": False, "error": "Trading signals not available"}
            
            elif command == "get_performance":
                try:
                    # Directly calculate performance metrics
                    if hasattr(self.trading_bot, 'calculate_performance_metrics'):
                        metrics = self.trading_bot.calculate_performance_metrics()
                        return {"success": True, "performance": metrics}
                    else:
                        return {"success": False, "error": "Performance metrics not available"}
                except Exception as e:
                    logger.error(f"Error getting performance metrics: {e}")
                    return {"success": False, "error": f"Error getting performance: {str(e)}"}
            
            elif command == "get_prices":
                if hasattr(self.trading_bot, 'current_prices'):
                    return {"success": True, "prices": self.trading_bot.current_prices}
                else:
                    return {"success": False, "error": "Price data not available"}
            
            elif command == "get_history":
                if hasattr(self.trading_bot, 'risk_manager') and hasattr(self.trading_bot.risk_manager, 'trade_history'):
                    # Get limit param or default to 20
                    limit = params.get("limit", 20)
                    history = self.trading_bot.risk_manager.trade_history[-limit:] if limit > 0 else []
                    return {"success": True, "history": history}
                else:
                    return {"success": False, "error": "Trade history not available"}
            
            elif command == "get_config":
                if hasattr(self.trading_bot, 'config'):
                    if isinstance(self.trading_bot.config, dict):
                        return {"success": True, "config": self.trading_bot.config}
                    else:
                        # Handle Config object if it's not a dict
                        return {"success": True, "config": self.trading_bot.config.config}
                else:
                    return {"success": False, "error": "Config not available"}
            
            # Commands related to trade actions
            elif command == "open_position":
                if not hasattr(self.trading_bot, 'trade_manager'):
                    return {"success": False, "error": "Trade manager not available"}
                
                # Get parameters
                symbol = params.get("symbol")
                side = params.get("side")
                if not symbol or not side:
                    return {"success": False, "error": "Symbol and side are required"}
                
                # Optional parameters
                signal_strength = params.get("signal_strength", 0.5)
                
                # Get current price
                current_price = None
                if hasattr(self.trading_bot, 'current_prices') and symbol in self.trading_bot.current_prices:
                    current_price = self.trading_bot.current_prices[symbol]
                else:
                    return {"success": False, "error": "Current price not available"}
                
                # Get price data
                dataframe = None
                primary_timeframe = self.trading_bot.config.get("timeframes", "primary")
                if (hasattr(self.trading_bot, 'price_data') and 
                    symbol in self.trading_bot.price_data and
                    primary_timeframe in self.trading_bot.price_data[symbol]):
                    dataframe = self.trading_bot.price_data[symbol][primary_timeframe]
                
                # Open position
                position_id = self.trading_bot.trade_manager.open_position(
                    symbol=symbol,
                    side=side,
                    signal_strength=signal_strength,
                    current_price=current_price,
                    dataframe=dataframe
                )
                
                if position_id:
                    return {"success": True, "position_id": position_id}
                else:
                    return {"success": False, "error": "Failed to open position"}
            
            elif command == "close_position":
                if not hasattr(self.trading_bot, 'trade_manager'):
                    return {"success": False, "error": "Trade manager not available"}
                
                position_id = params.get("position_id")
                if not position_id:
                    return {"success": False, "error": "Position ID is required"}
                
                reason = params.get("reason", "manual")
                
                success = self.trading_bot.trade_manager.close_position(position_id, reason)
                
                if success:
                    return {"success": True, "message": f"Position {position_id} closed"}
                else:
                    return {"success": False, "error": f"Failed to close position {position_id}"}
            
            elif command == "update_position":
                if not hasattr(self.trading_bot, 'trade_manager'):
                    return {"success": False, "error": "Trade manager not available"}
                
                position_id = params.get("position_id")
                if not position_id:
                    return {"success": False, "error": "Position ID is required"}
                
                # Get update parameters
                stop_loss = params.get("stop_loss")
                take_profit = params.get("take_profit")
                
                success = self.trading_bot.trade_manager.update_position(
                    position_id=position_id,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if success:
                    return {"success": True, "message": f"Position {position_id} updated"}
                else:
                    return {"success": False, "error": f"Failed to update position {position_id}"}
            
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
        
        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}


    def _process_file_commands(self):
        """Process commands from file as fallback."""
        while self.running:
            try:
                if os.path.exists(COMMAND_FILE):
                    try:
                        with open(COMMAND_FILE, 'r') as f:
                            commands = json.load(f)
                        
                        # Process unprocessed commands
                        for i, cmd in enumerate(commands.get("commands", [])):
                            if not cmd.get("processed", False):
                                logger.info(f"Processing file command: {cmd['command']}")
                                self._execute_command(cmd["command"], cmd.get("params", {}))
                                commands["commands"][i]["processed"] = True
                        
                        # Write back
                        with open(COMMAND_FILE, 'w') as f:
                            json.dump(commands, f, indent=4)
                    except Exception as e:
                        logger.error(f"Error with command file: {e}")
                        # If file is corrupted, rename it and create a new one
                        try:
                            os.rename(COMMAND_FILE, f"{COMMAND_FILE}.bak.{int(time.time())}")
                            with open(COMMAND_FILE, 'w') as f:
                                json.dump({"commands": []}, f)
                        except Exception as backup_err:
                            logger.error(f"Error creating backup command file: {backup_err}")
            except Exception as e:
                logger.error(f"Error in file processing: {e}")
            
            # Update shared data file with current state
            self._update_shared_data()
            
            # Sleep between file checks (2 seconds)
            time.sleep(2)

    # Data retrieval methods - customize these to match your trading bot's structure
    
    def _get_portfolio_summary(self):
        """Get portfolio summary from trading bot."""
        try:
            # Get account balance
            balance_info = self.trading_bot.api.get_wallet_balance("USDT")
            account_balance = balance_info.get('total', 0)
            available_balance = balance_info.get('free', 0)
            
            # Get positions summary
            position_summary = self.trading_bot.trade_manager.get_position_summary()
            total_positions = position_summary.get("total_positions", 0)
            long_positions = position_summary.get("long_positions", 0)
            short_positions = position_summary.get("short_positions", 0)
            total_exposure = position_summary.get("total_exposure", 0)
            
            # Get daily P&L
            daily_pnl = self.trading_bot.risk_manager.daily_stats.get("net_pnl", 0)
            daily_pnl_percentage = 0
            if self.trading_bot.risk_manager.daily_stats.get("starting_balance", 0) > 0:
                daily_pnl_percentage = (daily_pnl / self.trading_bot.risk_manager.daily_stats["starting_balance"]) * 100
            
            # Get performance metrics
            win_rate = 0
            if self.trading_bot.risk_manager.daily_stats.get("trades", 0) > 0:
                win_rate = (self.trading_bot.risk_manager.daily_stats.get("wins", 0) / 
                           self.trading_bot.risk_manager.daily_stats["trades"]) * 100
            
            # Get drawdown
            current_drawdown = self.trading_bot.risk_manager.current_drawdown * 100
            
            return {
                "total_balance": account_balance,
                "available_balance": available_balance,
                "total_positions_value": total_exposure,
                "daily_pnl": daily_pnl,
                "daily_pnl_percentage": daily_pnl_percentage,
                "total_positions": total_positions,
                "long_positions": long_positions,
                "short_positions": short_positions,
                "win_rate": win_rate,
                "current_drawdown": current_drawdown
            }
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def _get_positions(self):
        """Get active positions from trading bot."""
        try:
            positions = []
            
            for pos in self.trading_bot.trade_manager.active_positions:
                # Calculate unrealized P&L
                unrealized_pnl = 0
                pnl_percentage = 0
                
                if pos.symbol in self.trading_bot.current_prices:
                    current_price = self.trading_bot.current_prices[pos.symbol]
                    
                    if pos.side == "buy":
                        unrealized_pnl = (current_price - pos.entry_price) * pos.amount
                        pnl_percentage = ((current_price / pos.entry_price) - 1) * 100
                    else:
                        unrealized_pnl = (pos.entry_price - current_price) * pos.amount
                        pnl_percentage = ((pos.entry_price / current_price) - 1) * 100
                
                # Calculate duration
                duration = datetime.datetime.now() - pos.entry_time
                duration_str = self._format_duration(duration)
                
                # Calculate position value
                position_value = pos.entry_price * pos.amount
                
                positions.append({
                    "id": pos.id,
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "amount": pos.amount,
                    "value": position_value,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "unrealized_pnl": unrealized_pnl,
                    "pnl_percentage": pnl_percentage,
                    "entry_time": pos.entry_time.isoformat(),
                    "duration": duration_str
                })
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def _get_signals(self):
        """Get trading signals from trading bot."""
        try:
            return self.trading_bot.trading_signals
        except Exception as e:
            logger.error(f"Error getting trading signals: {e}")
            return {}
    
    def _get_market_regimes(self):
        """Get market regimes from trading bot."""
        try:
            regimes = {}
            
            for symbol in self.trading_bot.config.get("symbols"):
                if symbol in self.trading_bot.price_data:
                    primary_timeframe = self.trading_bot.config.get("timeframes", "primary")
                    if primary_timeframe in self.trading_bot.price_data[symbol]:
                        df = self.trading_bot.price_data[symbol][primary_timeframe]
                        
                        # Detect market regime
                        regime = self.trading_bot.market_analyzer.detect_market_regime(df)
                        
                        # Get regime history
                        key = f"{symbol}_{primary_timeframe}"
                        since = datetime.datetime.now() - datetime.timedelta(days=1)
                        if key in self.trading_bot.market_analyzer.regime_history:
                            history = self.trading_bot.market_analyzer.regime_history[key]
                            if history:
                                since = history[-1]["timestamp"]
                        
                        # Additional metrics
                        trend_score = 0
                        volatility = 0
                        
                        # Try to extract metrics from the data (these are simulated values)
                        if len(df) > 20:
                            # Simulate trend score - use RSI as proxy
                            rsi = df['close'].diff().rolling(window=14).apply(
                                lambda x: 100 - (100 / (1 + (x[x > 0].sum() / -x[x < 0].sum())))
                            )
                            trend_score = rsi.iloc[-1] - 50  # Center around 0
                            
                            # Simulate volatility - use standard deviation of returns
                            volatility = df['close'].pct_change().rolling(window=20).std().iloc[-1] * 100
                        
                        regimes[symbol] = {
                            "regime": regime.value,
                            "trend_score": trend_score,
                            "volatility": volatility,
                            "since": since.isoformat()
                        }
            
            return regimes
        except Exception as e:
            logger.error(f"Error getting market regimes: {e}")
            return {}
    
    def _get_performance_metrics(self):
        """Get performance metrics from trading bot."""
        try:
            # Try to get metrics from risk manager
            metrics = {}
            
            # Total trades
            metrics["total_trades"] = len(self.trading_bot.risk_manager.trade_history)
            
            # Win/loss count
            winning_trades = [t for t in self.trading_bot.risk_manager.trade_history if t.get("pnl", 0) > 0]
            losing_trades = [t for t in self.trading_bot.risk_manager.trade_history if t.get("pnl", 0) <= 0]
            metrics["winning_trades"] = len(winning_trades)
            metrics["losing_trades"] = len(losing_trades)
            
            # Win rate
            if metrics["total_trades"] > 0:
                metrics["win_rate"] = (metrics["winning_trades"] / metrics["total_trades"]) * 100
            else:
                metrics["win_rate"] = 0
            
            # Average win/loss
            if winning_trades:
                metrics["avg_win"] = sum(t.get("pnl_percentage", 0) for t in winning_trades) / len(winning_trades)
            else:
                metrics["avg_win"] = 0
                
            if losing_trades:
                metrics["avg_loss"] = sum(t.get("pnl_percentage", 0) for t in losing_trades) / len(losing_trades)
            else:
                metrics["avg_loss"] = 0
            
            # Profit factor
            total_profit = sum(t.get("pnl", 0) for t in winning_trades)
            total_loss = abs(sum(t.get("pnl", 0) for t in losing_trades))
            if total_loss > 0:
                metrics["profit_factor"] = total_profit / total_loss
            else:
                metrics["profit_factor"] = total_profit if total_profit > 0 else 0
            
            # Max drawdown
            metrics["max_drawdown"] = self.trading_bot.risk_manager.current_drawdown * 100
            
            # Sharpe ratio (simplified)
            metrics["sharpe_ratio"] = 0
            if metrics["avg_win"] > 0 and metrics["avg_loss"] < 0:
                metrics["sharpe_ratio"] = metrics["avg_win"] / abs(metrics["avg_loss"])
            
            # Equity curve data
            metrics["equity_data"] = []
            for point in self.trading_bot.risk_manager.equity_curve:
                metrics["equity_data"].append({
                    "date": point.get("timestamp", datetime.datetime.now()).strftime("%Y-%m-%d"),
                    "equity": point.get("balance", 0)
                })
            
            return metrics
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _get_trade_history(self, limit=10):
        """Get trade history from trading bot."""
        try:
            # Get trades from risk manager
            trades = self.trading_bot.risk_manager.trade_history[-limit:]
            
            # Format trades
            formatted_trades = []
            for trade in trades:
                # Calculate duration
                entry_time = datetime.datetime.fromisoformat(trade.get("entry_time"))
                exit_time = datetime.datetime.fromisoformat(trade.get("exit_time"))
                duration = exit_time - entry_time
                duration_str = self._format_duration(duration)
                
                formatted_trades.append({
                    "id": trade.get("id", ""),
                    "symbol": trade.get("symbol", ""),
                    "side": trade.get("side", ""),
                    "entry_price": trade.get("entry_price", 0),
                    "exit_price": trade.get("exit_price", 0),
                    "amount": trade.get("amount", 0),
                    "pnl": trade.get("pnl", 0),
                    "pnl_percentage": trade.get("pnl_percentage", 0),
                    "entry_time": trade.get("entry_time", ""),
                    "exit_time": trade.get("exit_time", ""),
                    "duration": duration_str,
                    "exit_reason": trade.get("exit_reason", "unknown")
                })
            
            return formatted_trades
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    def _get_price_data(self, symbol, timeframe):
        """Get price data for a symbol and timeframe."""
        try:
            if symbol in self.trading_bot.price_data and timeframe in self.trading_bot.price_data[symbol]:
                df = self.trading_bot.price_data[symbol][timeframe]
                
                # Convert DataFrame to list of dictionaries
                data = []
                for index, row in df.iterrows():
                    data.append({
                        "timestamp": index.isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"])
                    })
                
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data": data
                }
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting price data for {symbol} {timeframe}: {e}")
            return {}
    
    def _format_duration(self, duration):
        """Format a timedelta into a human-readable string."""
        days = duration.days
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 and not parts:
            parts.append(f"{seconds}s")
        
        return " ".join(parts) if parts else "0s"

# Function to attach the dashboard interface to an existing trading bot
def attach_dashboard(trading_bot):
    """Attach dashboard interface to an existing trading bot."""
    interface = DashboardInterface(trading_bot)
    trading_bot.dashboard = interface
    interface.start()
    
    # Add a shutdown hook to stop the interface when the bot stops
    original_stop = trading_bot.stop
    
    def new_stop():
        original_stop()
        interface.stop()
    
    trading_bot.stop = new_stop
    
    return interface