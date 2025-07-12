#!/usr/bin/env python3
"""
Claude-OpenAI API Bridge
Version: 2.1.0 (2025-01-11)

Allows Claude to query the OpenAI API through function calling

Features:
- Interactive model selection for both Claude and OpenAI
- Twin window display mode with separate OpenAI/Claude windows
- Continuing conversation with memory
- Live statistics display per model
- Session logging
- Conversation save/load
- OpenAI interaction visibility toggle

API Key Priority:
1. System environment variables (highest priority)
2. .env file in current directory
3. Interactive setup if no keys found

Changelog:
- v2.1.0: Added twin window mode with separate OpenAI/Claude displays
- v2.0.0: Fixed global variable syntax error, added OpenAI interaction display
- v1.5.0: Added terminal UI with live stats
- v1.0.0: Initial release
"""

import os
import sys
import json
from typing import Dict, List, Optional, Any, Tuple
from anthropic import Anthropic
from openai import OpenAI
import logging
from pathlib import Path
from datetime import datetime
import time
import shutil
import subprocess
import platform
import signal
import textwrap

# Version
VERSION = "2.1.0"
VERSION_DATE = "2025-01-11"

# Configuration
CLAUDE_MODELS = [
    ("claude-opus-4-20250514", "Claude Opus 4 - Most capable model"),
    ("claude-sonnet-4-20250514", "Claude Sonnet 4 - Balanced performance"),
    ("claude-3-opus-20240229", "Claude 3 Opus (Legacy) - Deprecated"),
    ("claude-3-sonnet-20240229", "Claude 3 Sonnet (Legacy) - Deprecated"),
    ("claude-3-haiku-20240307", "Claude 3 Haiku (Legacy) - Fast, lightweight")
]

# Selected models (will be set interactively)
CLAUDE_MODEL = None
OPENAI_MODEL = None

# Display options - using a mutable object to avoid global declaration issues
class Config:
    SHOW_OPENAI_INTERACTIONS = True
    USE_TWIN_WINDOWS = False

config = Config()

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Bold
    BOLD = '\033[1m'

# Set up logging
def setup_logging():
    """Set up logging with both console and file handlers"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"claude_openai_bridge_{timestamp}.log"
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler (simplified output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler (detailed output)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger, log_file

# Initialize logging
logger, current_log_file = setup_logging()
logger.info(f"Claude-OpenAI Bridge v{VERSION} starting...")
logger.info(f"Session started. Logging to: {current_log_file}")

# Check for environment variables first
anthropic_from_env = os.getenv("ANTHROPIC_API_KEY")
openai_from_env = os.getenv("OPENAI_API_KEY")

if anthropic_from_env and openai_from_env:
    logger.info("✅ Found API keys in environment variables")
else:
    # Only load .env if environment variables are not set
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            logger.info("📄 Loaded API keys from .env file (environment variables not found)")
    except ImportError:
        logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")
        if not (anthropic_from_env and openai_from_env):
            logger.error("❌ No API keys found in environment variables and can't load .env file")


class TwinTerminalUI:
    """Handle twin window terminal UI with separate OpenAI and Claude displays"""
    
    def __init__(self):
        self.openai_lines = []
        self.claude_lines = []
        self.max_lines_per_window = 250
        self.last_terminal_size = None
        self.resize_handler_installed = False
        self.status_height = 3  # Height of status bar for each window
        self.divider_height = 1  # Height of divider between windows
        self.input_height = 2  # Height for input area
        self.install_resize_handler()
        
        # Separate stats for each model
        self.openai_stats = {
            'message_count': 0,
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'estimated_cost': 0.0,
            'last_model': OPENAI_MODEL or 'gpt-4'
        }
        
        self.claude_stats = {
            'message_count': 0,
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'estimated_cost': 0.0,
            'context_size': 0
        }
        
    def install_resize_handler(self):
        """Install handler for terminal resize events"""
        try:
            if hasattr(signal, 'SIGWINCH'):
                signal.signal(signal.SIGWINCH, self._handle_resize)
                self.resize_handler_installed = True
        except:
            pass
    
    def _handle_resize(self, signum, frame):
        """Handle terminal resize signal"""
        self.last_terminal_size = None
        
    def clear_screen(self):
        """Clear the terminal screen"""
        if os.name == 'nt':
            os.system('cls')
        else:
            print('\033[2J\033[H', end='')
            sys.stdout.flush()
    
    def move_cursor(self, row, col):
        """Move cursor to specific position"""
        print(f"\033[{row};{col}H", end='')
        sys.stdout.flush()
    
    def get_terminal_size(self):
        """Get terminal dimensions"""
        try:
            size = os.get_terminal_size()
            return size
        except:
            try:
                cols, lines = shutil.get_terminal_size((80, 24))
                return os.terminal_size((cols, lines))
            except:
                return os.terminal_size((80, 24))
    
    def draw_status_bar(self, row, width, title, stats, color):
        """Draw a colored status bar with statistics"""
        self.move_cursor(row, 1)
        
        # Create status line content
        if 'openai' in title.lower():
            status = f" {title} | Model: {stats['last_model']} | Messages: {stats['message_count']} | "
            status += f"Tokens: {stats['total_tokens']:,} (I:{stats['input_tokens']:,}/O:{stats['output_tokens']:,}) | "
            status += f"Cost: ${stats['estimated_cost']:.4f} "
        else:  # Claude
            status = f" {title} | Model: {CLAUDE_MODEL} | Messages: {stats['message_count']} | "
            status += f"Tokens: {stats['total_tokens']:,} (I:{stats['input_tokens']:,}/O:{stats['output_tokens']:,}) | "
            status += f"Context: {stats['context_size']:,} | Cost: ${stats['estimated_cost']:.4f} "
        
        # Truncate if too long
        if len(status) > width - 2:
            status = status[:width-5] + "... "
        
        # Pad to full width
        status = status.ljust(width)
        
        # Print with color
        print(f"{color}{Colors.BOLD}{status}{Colors.RESET}")
    
    def draw_divider(self, row, width):
        """Draw a divider line between windows"""
        self.move_cursor(row, 1)
        print(f"{Colors.BRIGHT_BLACK}{'═' * width}{Colors.RESET}")
    
    def add_openai_line(self, line: str):
        """Add a line to OpenAI window"""
        if '\n' in line:
            for sub_line in line.split('\n'):
                self.add_openai_line(sub_line)
            return
        
        self.openai_lines.append(line)
        while len(self.openai_lines) > self.max_lines_per_window * 2:
            self.openai_lines.pop(0)
    
    def add_claude_line(self, line: str):
        """Add a line to Claude window"""
        if '\n' in line:
            for sub_line in line.split('\n'):
                self.add_claude_line(sub_line)
            return
        
        self.claude_lines.append(line)
        while len(self.claude_lines) > self.max_lines_per_window * 2:
            self.claude_lines.pop(0)
    
    def update_stats(self, model_type, **kwargs):
        """Update statistics for a specific model"""
        if model_type == 'openai':
            stats = self.openai_stats
        else:
            stats = self.claude_stats
        
        for key, value in kwargs.items():
            if key in stats:
                if key in ['message_count', 'total_tokens', 'input_tokens', 'output_tokens']:
                    stats[key] += value
                elif key == 'estimated_cost':
                    stats[key] += value
                else:
                    stats[key] = value
    
    def calculate_window_sizes(self, term_height):
        """Calculate the size of each window"""
        # Reserve space for input area
        available_height = term_height - self.input_height
        
        # Calculate window sizes
        openai_total = (available_height - self.divider_height) // 2
        claude_total = available_height - self.divider_height - openai_total
        
        openai_content = openai_total - self.status_height
        claude_content = claude_total - self.status_height
        
        return {
            'openai_start': 1,
            'openai_status': 1,
            'openai_content_start': 1 + self.status_height,
            'openai_content_height': max(1, openai_content),
            'divider_row': 1 + openai_total,
            'claude_start': 1 + openai_total + self.divider_height,
            'claude_status': 1 + openai_total + self.divider_height,
            'claude_content_start': 1 + openai_total + self.divider_height + self.status_height,
            'claude_content_height': max(1, claude_content),
            'input_row': term_height - 1
        }
    
    def refresh_display(self):
        """Refresh the entire twin window display"""
        term_size = self.get_terminal_size()
        term_height = term_size.lines
        term_width = term_size.columns
        
        # Check if terminal was resized
        if self.last_terminal_size != term_size:
            self.clear_screen()
            self.last_terminal_size = term_size
        
        # Calculate window positions
        layout = self.calculate_window_sizes(term_height)
        
        # Draw OpenAI window
        self.draw_status_bar(
            layout['openai_status'], 
            term_width, 
            "🤖 OpenAI", 
            self.openai_stats,
            Colors.BG_BLUE + Colors.WHITE
        )
        
        # Draw OpenAI content
        for i in range(layout['openai_content_height']):
            self.move_cursor(layout['openai_content_start'] + i, 1)
            print(" " * term_width, end='\r')
        
        display_lines = self.openai_lines[-layout['openai_content_height']:] if self.openai_lines else []
        for i, line in enumerate(display_lines):
            if i < layout['openai_content_height']:
                self.move_cursor(layout['openai_content_start'] + i, 1)
                print(line[:term_width-1] if len(line) >= term_width else line)
        
        # Draw divider
        self.draw_divider(layout['divider_row'], term_width)
        
        # Draw Claude window
        self.draw_status_bar(
            layout['claude_status'], 
            term_width, 
            "🧠 Claude", 
            self.claude_stats,
            Colors.BG_MAGENTA + Colors.WHITE
        )
        
        # Draw Claude content
        for i in range(layout['claude_content_height']):
            self.move_cursor(layout['claude_content_start'] + i, 1)
            print(" " * term_width, end='\r')
        
        display_lines = self.claude_lines[-layout['claude_content_height']:] if self.claude_lines else []
        for i, line in enumerate(display_lines):
            if i < layout['claude_content_height']:
                self.move_cursor(layout['claude_content_start'] + i, 1)
                print(line[:term_width-1] if len(line) >= term_width else line)
        
        # Position cursor for input
        self.move_cursor(layout['input_row'], 1)
        print(" " * term_width, end='\r')
        sys.stdout.flush()
    
    def check_terminal_size(self):
        """Check if terminal is large enough for twin window UI"""
        term_size = self.get_terminal_size()
        min_width = 80
        min_height = 30
        
        if term_size.columns < min_width or term_size.lines < min_height:
            print(f"\n⚠️  Terminal size ({term_size.columns}x{term_size.lines}) is smaller than recommended ({min_width}x{min_height})")
            print("Twin window mode requires a larger terminal.")
            print("Consider resizing your terminal or using standard mode.")
            input("\nPress Enter to continue anyway...")
            return False
        return True


class TerminalUI:
    """Handle terminal UI with statistics at the top (original single window mode)"""
    
    def __init__(self):
        self.conversation_lines = []
        self.max_conversation_lines = 500
        self.header_height = 11
        self.last_terminal_size = None
        self.resize_handler_installed = False
        self.install_resize_handler()
        
    def install_resize_handler(self):
        """Install handler for terminal resize events"""
        try:
            if hasattr(signal, 'SIGWINCH'):
                signal.signal(signal.SIGWINCH, self._handle_resize)
                self.resize_handler_installed = True
        except:
            pass
    
    def _handle_resize(self, signum, frame):
        """Handle terminal resize signal"""
        self.last_terminal_size = None
        
    def clear_screen(self):
        """Clear the terminal screen"""
        if os.name == 'nt':
            os.system('cls')
        else:
            print('\033[2J\033[H', end='')
            sys.stdout.flush()
    
    def move_cursor(self, row, col):
        """Move cursor to specific position"""
        print(f"\033[{row};{col}H", end='')
        sys.stdout.flush()
    
    def save_cursor(self):
        """Save current cursor position"""
        print("\033[s", end='')
        sys.stdout.flush()
    
    def restore_cursor(self):
        """Restore saved cursor position"""
        print("\033[u", end='')
        sys.stdout.flush()
    
    def get_terminal_size(self):
        """Get terminal dimensions with WSL compatibility"""
        try:
            size = os.get_terminal_size()
            return size
        except:
            try:
                cols, lines = shutil.get_terminal_size((80, 24))
                return os.terminal_size((cols, lines))
            except:
                return os.terminal_size((80, 24))
    
    def detect_environment(self):
        """Detect if running in WSL or other special environments"""
        info = {
            'is_wsl': False,
            'is_windows': platform.system() == 'Windows',
            'is_unix': platform.system() in ['Linux', 'Darwin'],
            'terminal': os.environ.get('TERM', 'unknown')
        }
        
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    info['is_wsl'] = True
        except:
            pass
        
        if 'WSL_DISTRO_NAME' in os.environ or 'WSL_INTEROP' in os.environ:
            info['is_wsl'] = True
            
        return info
    
    def draw_header(self, stats: Dict[str, Any]):
        """Draw the statistics header"""
        self.save_cursor()
        self.move_cursor(1, 1)
        
        term_size = self.get_terminal_size()
        cols = term_size.columns
        
        if self.last_terminal_size != term_size:
            self.clear_screen()
            self.last_terminal_size = term_size
            self.refresh_display(stats)
            return
        
        for i in range(self.header_height):
            self.move_cursor(i + 1, 1)
            print(" " * cols)
        
        self.move_cursor(1, 1)
        
        print("📊 Context Usage Statistics (Live)")
        print("=" * min(60, cols - 1))
        print(f"Messages: {stats['message_count']} | Conversation length: {stats['conversation_length']} messages")
        print(f"Total tokens: {stats['total_tokens']:,} (Input: {stats['total_input_tokens']:,} | Output: {stats['total_output_tokens']:,})")
        print(f"Avg tokens/msg: {stats['avg_tokens_per_message']:.0f} | Session: {stats['session_duration_minutes']:.1f} min")
        print(f"Estimated cost: ${stats['estimated_cost_usd']:.4f}")
        openai_status = "ON" if config.SHOW_OPENAI_INTERACTIONS else "OFF"
        print(f"OpenAI interactions: {openai_status}")
        print("=" * min(60, cols - 1))
        print("Commands: exit, clear, history, stats, save, load, resize, last, toggle, help")
        print("-" * min(60, cols - 1))
        
        self.restore_cursor()
        sys.stdout.flush()
    
    def add_conversation_line(self, line: str):
        """Add a line to the conversation display"""
        term_size = self.get_terminal_size()
        max_width = term_size.columns - 2
        
        if '\n' in line:
            for sub_line in line.split('\n'):
                self.add_conversation_line(sub_line)
            return
        
        if len(line) > max_width and max_width > 20:
            wrapped_lines = textwrap.wrap(line, width=max_width, 
                                        break_long_words=False, 
                                        break_on_hyphens=False)
            for wrapped_line in wrapped_lines:
                self.conversation_lines.append(wrapped_line)
        else:
            self.conversation_lines.append(line)
            
        while len(self.conversation_lines) > self.max_conversation_lines * 2:
            self.conversation_lines.pop(0)
    
    def refresh_display(self, stats: Dict[str, Any]):
        """Refresh the entire display"""
        self.draw_header(stats)
        
        self.move_cursor(self.header_height + 1, 1)
        
        term_size = self.get_terminal_size()
        term_height = term_size.lines
        term_width = term_size.columns
        
        conversation_height = term_height - self.header_height - 2
        
        for i in range(conversation_height):
            self.move_cursor(self.header_height + 1 + i, 1)
            print(" " * term_width, end='\r')
        
        self.move_cursor(self.header_height + 1, 1)
        if conversation_height > 0:
            display_lines = self.conversation_lines[-conversation_height:] if self.conversation_lines else []
            for i, line in enumerate(display_lines):
                if i < conversation_height:
                    self.move_cursor(self.header_height + 1 + i, 1)
                    print(line[:term_width-1] if len(line) >= term_width else line)
        
        self.move_cursor(term_height - 1, 1)
        print(" " * term_width, end='\r')
        sys.stdout.flush()
    
    def check_terminal_size(self):
        """Check if terminal is large enough for UI"""
        term_size = self.get_terminal_size()
        min_width = 60
        min_height = 20
        
        if term_size.columns < min_width or term_size.lines < min_height:
            print(f"\n⚠️  Terminal size ({term_size.columns}x{term_size.lines}) is smaller than recommended ({min_width}x{min_height})")
            print("The enhanced UI may not display correctly.")
            print("Consider resizing your terminal or using standard mode.")
            input("\nPress Enter to continue anyway...")
            return False
        return True
    
    def print_environment_info(self):
        """Print detected environment information"""
        env = self.detect_environment()
        term_size = self.get_terminal_size()
        logger.info(f"Environment: WSL={env['is_wsl']}, Terminal={env['terminal']}, Size={term_size.columns}x{term_size.lines}")


class OpenAITool:
    """Wrapper for OpenAI API calls that Claude can use"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Make a chat completion request to OpenAI
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: OpenAI model to use (defaults to globally selected model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with the response
        """
        if model is None:
            model = OPENAI_MODEL or "gpt-4"
            
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_models(self) -> Dict[str, Any]:
        """List available OpenAI models"""
        try:
            models = self.client.models.list()
            model_ids = [model.id for model in models.data]
            return {
                "success": True,
                "models": model_ids
            }
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for OpenAI usage"""
        # OpenAI pricing (verify current rates)
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},  # per 1k tokens
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            'o1-preview': {'input': 0.015, 'output': 0.06},
            'o1-mini': {'input': 0.003, 'output': 0.012}
        }
        
        # Default pricing if model not found
        default_pricing = {'input': 0.01, 'output': 0.03}
        
        # Find matching pricing
        model_pricing = default_pricing
        for key in pricing:
            if key in model.lower():
                model_pricing = pricing[key]
                break
        
        input_cost = (input_tokens / 1000) * model_pricing['input']
        output_cost = (output_tokens / 1000) * model_pricing['output']
        
        return round(input_cost + output_cost, 6)


class ClaudeOpenAIBridge:
    """Main bridge class that handles Claude-OpenAI interaction"""
    
    def __init__(self, anthropic_api_key: str, openai_api_key: str):
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        self.openai_tool = OpenAITool(openai_api_key)
        self.conversation_history = []
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.message_count = 0
        self.session_start = datetime.now()
        
        # Separate OpenAI tracking
        self.openai_total_input_tokens = 0
        self.openai_total_output_tokens = 0
        self.openai_message_count = 0
        self.openai_total_cost = 0.0
        
        # Define the tools available to Claude
        self.tools = [
            {
                "name": "query_openai",
                "description": "Query the OpenAI API for chat completions",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                    "content": {"type": "string"}
                                },
                                "required": ["role", "content"]
                            },
                            "description": "Messages to send to OpenAI"
                        },
                        "model": {
                            "type": "string",
                            "description": "OpenAI model to use",
                            "default": "gpt-4"
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature (0-2)",
                            "default": 0.7
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens in response",
                            "default": None
                        }
                    },
                    "required": ["messages"]
                }
            },
            {
                "name": "list_openai_models",
                "description": "List available OpenAI models",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        
        return {
            "message_count": self.message_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": total_tokens,
            "avg_tokens_per_message": total_tokens / max(self.message_count, 1),
            "conversation_length": len(self.conversation_history),
            "session_duration_minutes": round(session_duration, 2),
            "estimated_cost_usd": self._estimate_cost(self.total_input_tokens, self.total_output_tokens),
            "openai_stats": {
                "message_count": self.openai_message_count,
                "total_input_tokens": self.openai_total_input_tokens,
                "total_output_tokens": self.openai_total_output_tokens,
                "total_cost": self.openai_total_cost
            }
        }
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on Claude Opus 4 pricing"""
        input_cost_per_1k = 0.015
        output_cost_per_1k = 0.075
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return round(input_cost + output_cost, 4)
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        print("\n📊 Context Usage Statistics:")
        print("=" * 50)
        print(f"Messages exchanged: {stats['message_count']}")
        print(f"Conversation length: {stats['conversation_length']} messages")
        print(f"Total tokens used: {stats['total_tokens']:,}")
        print(f"  - Input tokens: {stats['total_input_tokens']:,}")
        print(f"  - Output tokens: {stats['total_output_tokens']:,}")
        print(f"Average tokens/message: {stats['avg_tokens_per_message']:.0f}")
        print(f"Session duration: {stats['session_duration_minutes']:.1f} minutes")
        print(f"Estimated Claude cost: ${stats['estimated_cost_usd']:.4f}")
        
        if stats['openai_stats']['message_count'] > 0:
            print("\n🤖 OpenAI Usage:")
            print(f"Messages: {stats['openai_stats']['message_count']}")
            print(f"Total tokens: {stats['openai_stats']['total_input_tokens'] + stats['openai_stats']['total_output_tokens']:,}")
            print(f"Estimated OpenAI cost: ${stats['openai_stats']['total_cost']:.4f}")
        print("=" * 50)
    
    def clear_conversation(self):
        """Clear the conversation history and reset statistics"""
        self.conversation_history = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.message_count = 0
        self.openai_total_input_tokens = 0
        self.openai_total_output_tokens = 0
        self.openai_message_count = 0
        self.openai_total_cost = 0.0
        self.session_start = datetime.now()
        logger.info("Conversation history and statistics cleared")
    
    def save_conversation(self, filename: str = "conversation_history.json"):
        """Save conversation history to a JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            logger.info(f"Conversation saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self, filename: str = "conversation_history.json"):
        """Load conversation history from a JSON file"""
        try:
            with open(filename, 'r') as f:
                self.conversation_history = json.load(f)
            logger.info(f"Loaded {len(self.conversation_history)} messages from {filename}")
            return True
        except FileNotFoundError:
            logger.warning(f"No saved conversation found at {filename}")
            return False
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return False
    
    def handle_tool_use(self, tool_name: str, tool_input: Dict[str, Any], ui=None, twin_ui=None) -> Dict[str, Any]:
        """Handle tool calls from Claude"""
        if tool_name == "query_openai":
            logger.info(f"OpenAI Query - Model: {tool_input.get('model', 'default')}")
            logger.info(f"OpenAI Query - Messages: {json.dumps(tool_input.get('messages', []), indent=2)}")
            
            # Display in appropriate UI
            if twin_ui:
                # Add to OpenAI window
                twin_ui.add_openai_line("\n🔧 SENDING TO OPENAI:")
                twin_ui.add_openai_line("="*40)
                model = tool_input.get('model', OPENAI_MODEL or 'gpt-4')
                twin_ui.add_openai_line(f"Model: {model}")
                twin_ui.add_openai_line(f"Temperature: {tool_input.get('temperature', 0.7)}")
                twin_ui.add_openai_line("\nMessages:")
                for msg in tool_input.get('messages', []):
                    msg_preview = msg['content'][:150] + '...' if len(msg['content']) > 150 else msg['content']
                    twin_ui.add_openai_line(f"  {msg['role'].upper()}: {msg_preview}")
                twin_ui.add_openai_line("="*40)
                twin_ui.refresh_display()
            elif config.SHOW_OPENAI_INTERACTIONS:
                if ui:
                    ui.add_conversation_line("\n🔧 Claude is using tool: " + tool_name)
                    ui.add_conversation_line("="*40)
                    ui.add_conversation_line("🤖 SENDING TO OPENAI:")
                    model = tool_input.get('model', OPENAI_MODEL or 'gpt-4')
                    ui.add_conversation_line(f"Model: {model}")
                    ui.add_conversation_line("Messages:")
                    for msg in tool_input.get('messages', []):
                        msg_preview = msg['content'][:150] + '...' if len(msg['content']) > 150 else msg['content']
                        ui.add_conversation_line(f"  {msg['role'].upper()}: {msg_preview}")
                    ui.add_conversation_line("="*40)
                    ui.refresh_display(self.get_statistics())
                else:
                    print("\n" + "="*60)
                    print("🤖 SENDING TO OPENAI:")
                    print("="*60)
                    print(f"Model: {tool_input.get('model', OPENAI_MODEL or 'gpt-4')}")
                    print(f"Temperature: {tool_input.get('temperature', 0.7)}")
                    print("\nMessages:")
                    for msg in tool_input.get('messages', []):
                        print(f"  {msg['role'].upper()}: {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")
                    print("="*60)
            
            # Make the API call
            result = self.openai_tool.chat_completion(**tool_input)
            
            # Update OpenAI statistics if successful
            if result['success']:
                self.openai_message_count += 1
                self.openai_total_input_tokens += result['usage']['prompt_tokens']
                self.openai_total_output_tokens += result['usage']['completion_tokens']
                
                # Calculate cost
                model_used = result.get('model', tool_input.get('model', OPENAI_MODEL or 'gpt-4'))
                cost = self.openai_tool.estimate_cost(
                    model_used,
                    result['usage']['prompt_tokens'],
                    result['usage']['completion_tokens']
                )
                self.openai_total_cost += cost
                
                # Update twin UI stats if available
                if twin_ui:
                    twin_ui.update_stats('openai', 
                        message_count=1,
                        input_tokens=result['usage']['prompt_tokens'],
                        output_tokens=result['usage']['completion_tokens'],
                        total_tokens=result['usage']['total_tokens'],
                        estimated_cost=cost,
                        last_model=model_used
                    )
            
            # Display the response
            if twin_ui:
                if result['success']:
                    twin_ui.add_openai_line("\n🤖 OPENAI RESPONSE:")
                    twin_ui.add_openai_line("="*40)
                    for line in result['content'].split('\n'):
                        twin_ui.add_openai_line(line)
                    twin_ui.add_openai_line("="*40)
                    twin_ui.add_openai_line(f"Tokens used: {result['usage']['total_tokens']}")
                    twin_ui.add_openai_line("")
                else:
                    twin_ui.add_openai_line(f"\n❌ OpenAI Error: {result['error']}\n")
                twin_ui.refresh_display()
            elif config.SHOW_OPENAI_INTERACTIONS:
                if ui:
                    if result['success']:
                        ui.add_conversation_line("\n🤖 OPENAI RESPONSE:")
                        ui.add_conversation_line("="*40)
                        for line in result['content'].split('\n'):
                            ui.add_conversation_line(line)
                        ui.add_conversation_line("="*40)
                        ui.add_conversation_line(f"OpenAI tokens used: {result['usage']['total_tokens']}")
                        ui.add_conversation_line("")
                        ui.refresh_display(self.get_statistics())
                    else:
                        ui.add_conversation_line(f"\n❌ OpenAI Error: {result['error']}\n")
                        ui.refresh_display(self.get_statistics())
                else:
                    if result['success']:
                        print("\n🤖 OPENAI RESPONSE:")
                        print("="*60)
                        print(result['content'])
                        print("="*60)
                        print(f"Tokens used: {result['usage']['total_tokens']}")
                        print("="*60 + "\n")
                    else:
                        print(f"\n❌ OpenAI Error: {result['error']}\n")
            
            return result
        elif tool_name == "list_openai_models":
            return self.openai_tool.list_models()
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    def chat_with_claude(self, user_message: str, system_prompt: Optional[str] = None, 
                        maintain_history: bool = True, ui=None, twin_ui=None) -> str:
        """
        Send a message to Claude with OpenAI tool access
        
        Args:
            user_message: The user's message to Claude
            system_prompt: Optional system prompt for Claude
            maintain_history: Whether to maintain conversation history
            ui: Optional TerminalUI instance for enhanced display
            twin_ui: Optional TwinTerminalUI instance for twin window display
            
        Returns:
            Claude's response as a string
        """
        if not CLAUDE_MODEL:
            logger.error("CLAUDE_MODEL is not set! Model selection must happen first.")
            return "Error: Claude model not selected. Please restart the application."
        
        if not system_prompt:
            system_prompt = f"""You are Claude (using model {CLAUDE_MODEL}), an AI assistant with access to the OpenAI API. 
You can query OpenAI models when needed to compare responses, get additional perspectives, 
or leverage OpenAI-specific capabilities. The default OpenAI model is {OPENAI_MODEL}.
Use the provided tools to interact with OpenAI's API."""
        
        if maintain_history:
            self.conversation_history.append({"role": "user", "content": user_message})
            messages = self.conversation_history.copy()
        else:
            messages = [{"role": "user", "content": user_message}]
        
        try:
            logger.info(f"Sending message to Claude using model: {CLAUDE_MODEL}. History length: {len(messages)}")
            
            response = self.anthropic.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                tools=self.tools
            )
            
            # Track token usage
            if hasattr(response, 'usage'):
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
                self.message_count += 1
                logger.info(f"Token usage - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")
                
                # Update twin UI stats if available
                if twin_ui:
                    cost = self._estimate_cost(response.usage.input_tokens, response.usage.output_tokens)
                    twin_ui.update_stats('claude',
                        message_count=1,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                        estimated_cost=cost,
                        context_size=len(str(messages))
                    )
            
            # Handle tool use if Claude wants to call OpenAI
            final_response = ""
            tool_used = False
            
            for content in response.content:
                if content.type == "text":
                    final_response += content.text
                elif content.type == "tool_use":
                    tool_used = True
                    logger.info(f"Claude is calling tool: {content.name}")
                    
                    # Show tool use in appropriate UI
                    if twin_ui:
                        twin_ui.add_claude_line("\n🔧 Claude is calling tool: " + content.name)
                        twin_ui.refresh_display()
                    elif ui and config.SHOW_OPENAI_INTERACTIONS:
                        ui.add_conversation_line("\n🔧 Claude is using tool: " + content.name)
                        if content.name == "query_openai":
                            ui.add_conversation_line("="*40)
                            ui.add_conversation_line("🤖 SENDING TO OPENAI:")
                            model = content.input.get('model', OPENAI_MODEL or 'gpt-4')
                            ui.add_conversation_line(f"Model: {model}")
                            ui.add_conversation_line("Messages:")
                            for msg in content.input.get('messages', []):
                                msg_preview = msg['content'][:150] + '...' if len(msg['content']) > 150 else msg['content']
                                ui.add_conversation_line(f"  {msg['role'].upper()}: {msg_preview}")
                            ui.add_conversation_line("="*40)
                            ui.refresh_display(self.get_statistics())
                    
                    tool_result = self.handle_tool_use(content.name, content.input, ui, twin_ui)
                    
                    # For tool use, we need to track the full exchange
                    if maintain_history:
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": response.content
                        })
                        self.conversation_history.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": json.dumps(tool_result)
                            }]
                        })
                        messages = self.conversation_history.copy()
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": response.content
                        })
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": json.dumps(tool_result)
                            }]
                        })
                    
                    # Get Claude's response after tool use
                    follow_up = self.anthropic.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=messages,
                        tools=self.tools
                    )
                    
                    # Track follow-up token usage
                    if hasattr(follow_up, 'usage'):
                        self.total_input_tokens += follow_up.usage.input_tokens
                        self.total_output_tokens += follow_up.usage.output_tokens
                        logger.info(f"Follow-up token usage - Input: {follow_up.usage.input_tokens}, Output: {follow_up.usage.output_tokens}")
                        
                        # Update twin UI stats if available
                        if twin_ui:
                            cost = self._estimate_cost(follow_up.usage.input_tokens, follow_up.usage.output_tokens)
                            twin_ui.update_stats('claude',
                                input_tokens=follow_up.usage.input_tokens,
                                output_tokens=follow_up.usage.output_tokens,
                                total_tokens=follow_up.usage.input_tokens + follow_up.usage.output_tokens,
                                estimated_cost=cost,
                                context_size=len(str(messages))
                            )
                    
                    for follow_up_content in follow_up.content:
                        if follow_up_content.type == "text":
                            final_response += "\n" + follow_up_content.text
            
            # Add Claude's final response to history
            if maintain_history and final_response:
                if not tool_used:
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_response
                    })
            
            logger.info(f"Response generated. Length: {len(final_response)} chars")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in Claude API call: {e}")
            return f"Error: {str(e)}"


def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    try:
        import anthropic
    except ImportError:
        missing.append("anthropic")
    
    try:
        import openai
    except ImportError:
        missing.append("openai")
    
    try:
        import dotenv
    except ImportError:
        missing.append("python-dotenv")
    
    if missing:
        print("\n⚠️  Missing required packages:")
        print(f"   pip install {' '.join(missing)}")
        print("\nOr install all dependencies:")
        print("   pip install anthropic openai python-dotenv")
        return False
    
    return True


def setup_env_file():
    """Interactive setup for .env file"""
    env_path = Path(".env")
    
    if env_path.exists():
        logger.info(".env file already exists")
        overwrite = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if overwrite != 'y':
            return True
    
    print("\n🔧 Setting up .env file for API keys...")
    print("=" * 50)
    print("Note: Environment variables will take precedence over .env file")
    print()
    
    anthropic_key = input("Enter your Anthropic API key (or press Enter to skip): ").strip()
    if not anthropic_key:
        anthropic_key = "your-anthropic-api-key-here"
        print("⚠️  You'll need to add your Anthropic API key to .env later")
    
    openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    if not openai_key:
        openai_key = "your-openai-api-key-here"
        print("⚠️  You'll need to add your OpenAI API key to .env later")
    
    env_content = f"""# API Keys for Claude-OpenAI Bridge
# Note: Environment variables take precedence over these values
ANTHROPIC_API_KEY={anthropic_key}
OPENAI_API_KEY={openai_key}
"""
    
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print(f"\n✅ Created .env file at: {env_path.absolute()}")
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, "w") as f:
            f.write(""".env
__pycache__/
*.pyc
conversation_history.json
*.json
logs/
*.log
claude-venv/
.DS_Store
Thumbs.db
""")
        print("✅ Created .gitignore to protect your API keys and logs")
    
    return anthropic_key != "your-anthropic-api-key-here" and openai_key != "your-openai-api-key-here"


def select_claude_model():
    """Interactive Claude model selection"""
    print("\n🤖 Available Claude Models:")
    print("=" * 60)
    for i, (model_id, description) in enumerate(CLAUDE_MODELS, 1):
        print(f"{i}. {model_id}")
        print(f"   {description}")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nSelect Claude model (1-{}): ".format(len(CLAUDE_MODELS))).strip()
            idx = int(choice) - 1
            if 0 <= idx < len(CLAUDE_MODELS):
                selected_model = CLAUDE_MODELS[idx][0]
                print(f"✅ Selected: {selected_model}")
                return selected_model
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Please enter a number.")


def select_openai_model(openai_tool):
    """Interactive OpenAI model selection"""
    print("\n🔍 Fetching available OpenAI models...")
    
    result = openai_tool.list_models()
    if not result["success"]:
        print(f"❌ Error fetching models: {result['error']}")
        print("Using default model: gpt-4")
        return "gpt-4"
    
    all_models = result["models"]
    chat_models = [m for m in all_models if any(prefix in m for prefix in 
                   ["gpt-4", "gpt-3.5", "o1", "o3", "chatgpt"])]
    
    model_groups = {
        "GPT-4 Series": [m for m in chat_models if m.startswith("gpt-4") and "o" not in m],
        "GPT-4o Series (Multimodal)": [m for m in chat_models if "gpt-4o" in m],
        "GPT-3.5 Series": [m for m in chat_models if "gpt-3.5" in m],
        "O1 Series (Reasoning)": [m for m in chat_models if m.startswith("o1")],
        "O3 Series": [m for m in chat_models if m.startswith("o3")],
        "ChatGPT": [m for m in chat_models if m.startswith("chatgpt")]
    }
    
    numbered_models = []
    print("\n🤖 Available OpenAI Chat Models:")
    print("=" * 60)
    
    for group_name, models in model_groups.items():
        if models:
            print(f"\n{group_name}:")
            for model in sorted(models):
                numbered_models.append(model)
                print(f"{len(numbered_models)}. {model}")
    
    print("=" * 60)
    
    print(f"\n{len(numbered_models) + 1}. Enter custom model name")
    
    while True:
        try:
            choice = input(f"\nSelect OpenAI model (1-{len(numbered_models) + 1}): ").strip()
            idx = int(choice) - 1
            
            if idx == len(numbered_models):
                custom_model = input("Enter custom model name: ").strip()
                if custom_model:
                    print(f"✅ Selected: {custom_model}")
                    return custom_model
                else:
                    print("❌ Model name cannot be empty.")
            elif 0 <= idx < len(numbered_models):
                selected_model = numbered_models[idx]
                print(f"✅ Selected: {selected_model}")
                return selected_model
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Please enter a number.")


def main():
    """Main function with model selection"""
    global CLAUDE_MODEL, OPENAI_MODEL
    
    print(f"\n🚀 Claude-OpenAI Bridge v{VERSION} ({VERSION_DATE})")
    print("=" * 60)
    
    logger.info("Starting main() function")
    
    if not check_dependencies():
        return
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    key_source = "environment variables" if (anthropic_from_env and openai_from_env) else ".env file"
    
    if not anthropic_key or not openai_key:
        print("\n❌ No API keys found!")
        print("You can either:")
        print("1. Set environment variables:")
        print("   export ANTHROPIC_API_KEY='your-key'")
        print("   export OPENAI_API_KEY='your-key'")
        print("\n2. Create a .env file with your keys")
        
        create_env = input("\nWould you like to create a .env file now? (y/n): ").strip().lower()
        if create_env == 'y':
            if not setup_env_file():
                print("\n⚠️  Please add your API keys to the .env file and run again.")
                return
            from dotenv import load_dotenv
            load_dotenv()
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            key_source = ".env file"
        else:
            return
    
    if not anthropic_key or anthropic_key == "your-anthropic-api-key-here":
        logger.error(f"❌ Valid Anthropic API key not found in {key_source}")
        if key_source == ".env file":
            logger.info("Please edit .env and add your Anthropic API key")
        else:
            logger.info("Please set the ANTHROPIC_API_KEY environment variable")
        return
    
    if not openai_key or openai_key == "your-openai-api-key-here":
        logger.error(f"❌ Valid OpenAI API key not found in {key_source}")
        if key_source == ".env file":
            logger.info("Please edit .env and add your OpenAI API key")
        else:
            logger.info("Please set the OPENAI_API_KEY environment variable")
        return
    
    logger.info(f"✅ API keys loaded successfully from {key_source}")
    
    print("\n🚀 Claude-OpenAI Bridge Setup")
    print("=" * 60)
    
    CLAUDE_MODEL = select_claude_model()
    if not CLAUDE_MODEL:
        logger.error("No Claude model selected")
        return
    
    temp_openai_tool = OpenAITool(openai_key)
    OPENAI_MODEL = select_openai_model(temp_openai_tool)
    if not OPENAI_MODEL:
        logger.error("No OpenAI model selected")
        return
    
    logger.info(f"Creating bridge with Claude model: {CLAUDE_MODEL}")
    logger.info(f"Creating bridge with OpenAI model: {OPENAI_MODEL}")
    bridge = ClaudeOpenAIBridge(anthropic_key, openai_key)
    
    try:
        print("\n🔍 OpenAI Interaction Display")
        print("=" * 60)
        print("Would you like to see OpenAI prompts and responses?")
        print("This shows what Claude sends to OpenAI and what comes back.")
        print("(You can toggle this later with the 'toggle' command)")
        show_openai = input("\nShow OpenAI interactions? (y/n): ").strip().lower() == 'y'
        config.SHOW_OPENAI_INTERACTIONS = show_openai
        logger.info(f"OpenAI interaction display: {'ON' if show_openai else 'OFF'}")
        
        print("\n💬 Starting Interactive Chat Mode")
        print("=" * 60)
        
        print("\nDisplay Options:")
        print("1. Standard mode (simple, reliable)")
        print("2. Enhanced terminal UI (live stats at top)")
        print("3. Twin window mode (separate OpenAI/Claude windows)")
        
        ui_choice = input("\nSelect display mode (1-3): ").strip()
        use_terminal_ui = ui_choice == "2"
        use_twin_windows = ui_choice == "3"
        
        if use_twin_windows:
            print("\n🖥️  Twin Window Mode")
            print("- OpenAI interactions display in top window")
            print("- Claude responses display in bottom window")
            print("- Live statistics for each model")
            print("- Type 'resize' if display looks wrong")
            time.sleep(2)
            
            twin_ui = TwinTerminalUI()
            twin_ui.check_terminal_size()
            twin_ui.clear_screen()
            twin_ui.refresh_display()
            
            while True:
                term_size = twin_ui.get_terminal_size()
                layout = twin_ui.calculate_window_sizes(term_size.lines)
                twin_ui.move_cursor(layout['input_row'], 1)
                print(" " * term_size.columns, end='\r')
                
                try:
                    user_input = input("You: ").strip()
                except KeyboardInterrupt:
                    user_input = "exit"
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    twin_ui.clear_screen()
                    bridge.print_statistics()
                    break
                elif user_input.lower() == 'clear':
                    bridge.clear_conversation()
                    twin_ui.openai_lines = []
                    twin_ui.claude_lines = []
                    twin_ui.openai_stats = {
                        'message_count': 0,
                        'total_tokens': 0,
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'estimated_cost': 0.0,
                        'last_model': OPENAI_MODEL or 'gpt-4'
                    }
                    twin_ui.claude_stats = {
                        'message_count': 0,
                        'total_tokens': 0,
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'estimated_cost': 0.0,
                        'context_size': 0
                    }
                    twin_ui.add_claude_line("🗑️  Conversation history cleared. Starting fresh!")
                    twin_ui.refresh_display()
                    continue
                elif user_input.lower() == 'toggle':
                    config.SHOW_OPENAI_INTERACTIONS = not config.SHOW_OPENAI_INTERACTIONS
                    status = "ON" if config.SHOW_OPENAI_INTERACTIONS else "OFF"
                    twin_ui.add_claude_line(f"\n🔄 OpenAI interaction display toggled: {status}")
                    twin_ui.refresh_display()
                    continue
                elif user_input.lower() == 'help':
                    twin_ui.add_claude_line("\n📚 Available Commands:")
                    twin_ui.add_claude_line("  'exit' or 'quit' - Exit the conversation")
                    twin_ui.add_claude_line("  'clear' - Clear conversation history")
                    twin_ui.add_claude_line("  'history' - Show conversation history")
                    twin_ui.add_claude_line("  'stats' - Show detailed statistics")
                    twin_ui.add_claude_line("  'save' - Save conversation to file")
                    twin_ui.add_claude_line("  'load' - Load previous conversation")
                    twin_ui.add_claude_line("  'resize' - Force refresh display")
                    twin_ui.add_claude_line("  'toggle' - Toggle OpenAI interaction display")
                    twin_ui.add_claude_line("  'help' - Show this help message")
                    twin_ui.refresh_display()
                    continue
                elif user_input.lower() == 'resize':
                    twin_ui.last_terminal_size = None
                    twin_ui.clear_screen()
                    twin_ui.refresh_display()
                    twin_ui.add_claude_line("🔄 Display refreshed")
                    continue
                elif user_input.lower() == 'stats':
                    bridge.print_statistics()
                    continue
                elif user_input.lower() == 'save':
                    twin_ui.move_cursor(layout['input_row'], 1)
                    print(" " * term_size.columns, end='\r')
                    filename = input("Enter filename (or press Enter for default): ").strip()
                    if not filename:
                        filename = "conversation_history.json"
                    if bridge.save_conversation(filename):
                        twin_ui.add_claude_line(f"💾 Conversation saved to {filename}")
                    else:
                        twin_ui.add_claude_line("❌ Failed to save conversation")
                    twin_ui.refresh_display()
                    continue
                elif user_input.lower() == 'load':
                    twin_ui.move_cursor(layout['input_row'], 1)
                    print(" " * term_size.columns, end='\r')
                    filename = input("Enter filename to load (or press Enter for default): ").strip()
                    if not filename:
                        filename = "conversation_history.json"
                    if bridge.load_conversation(filename):
                        twin_ui.add_claude_line(f"📂 Loaded conversation from {filename}")
                        twin_ui.add_claude_line("Note: Token statistics were reset")
                    else:
                        twin_ui.add_claude_line("❌ Failed to load conversation")
                    twin_ui.refresh_display()
                    continue
                elif user_input.lower() == 'history':
                    if not bridge.conversation_history:
                        twin_ui.add_claude_line("📜 No conversation history yet.")
                    else:
                        twin_ui.add_claude_line("\n📜 Conversation History:")
                        twin_ui.add_claude_line("-" * 40)
                        for i, msg in enumerate(bridge.conversation_history):
                            role = msg['role'].upper()
                            if isinstance(msg['content'], str):
                                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                            else:
                                content = "[Complex message with tool use]"
                            twin_ui.add_claude_line(f"{i+1}. {role}: {content}")
                        twin_ui.add_claude_line("-" * 40)
                    twin_ui.refresh_display()
                    continue
                
                # Add user message to Claude window
                twin_ui.add_claude_line(f"\nYou: {user_input}")
                twin_ui.refresh_display()
                
                # Get Claude's response
                twin_ui.add_claude_line("Claude: [thinking...]")
                twin_ui.refresh_display()
                
                response = bridge.chat_with_claude(user_input, twin_ui=twin_ui)
                
                # Remove the [thinking...] line
                if twin_ui.claude_lines and twin_ui.claude_lines[-1] == "Claude: [thinking...]":
                    twin_ui.claude_lines.pop()
                
                # Add the actual response
                twin_ui.add_claude_line(f"Claude: {response}")
                twin_ui.refresh_display()
                
        elif use_terminal_ui:
            print("\n🖥️  Enhanced Terminal UI Mode")
            print("- Live statistics stay at the top")
            print("- Conversation scrolls below")
            print("- Terminal resizing is supported")
            print("- Type 'resize' if display looks wrong")
            time.sleep(2)
            
            ui = TerminalUI()
            ui.print_environment_info()
            ui.check_terminal_size()
            ui.clear_screen()
            
            ui.refresh_display(bridge.get_statistics())
            
            while True:
                term_size = ui.get_terminal_size()
                ui.move_cursor(term_size.lines - 1, 1)
                print(" " * term_size.columns, end='\r')
                
                try:
                    user_input = input("You: ").strip()
                except KeyboardInterrupt:
                    user_input = "exit"
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    ui.clear_screen()
                    bridge.print_statistics()
                    break
                elif user_input.lower() == 'clear':
                    bridge.clear_conversation()
                    ui.conversation_lines = []
                    ui.add_conversation_line("🗑️  Conversation history cleared. Starting fresh!")
                    ui.refresh_display(bridge.get_statistics())
                    continue
                elif user_input.lower() == 'last':
                    last_assistant_msg = None
                    for msg in reversed(bridge.conversation_history):
                        if msg['role'] == 'assistant' and isinstance(msg['content'], str):
                            last_assistant_msg = msg['content']
                            break
                    
                    if last_assistant_msg:
                        ui.add_conversation_line("\n📜 Last Claude Response (Full):")
                        ui.add_conversation_line("=" * 40)
                        for line in last_assistant_msg.split('\n'):
                            ui.add_conversation_line(line)
                        ui.add_conversation_line("=" * 40)
                    else:
                        ui.add_conversation_line("No assistant response found yet.")
                    ui.refresh_display(bridge.get_statistics())
                    continue
                elif user_input.lower() == 'toggle':
                    config.SHOW_OPENAI_INTERACTIONS = not config.SHOW_OPENAI_INTERACTIONS
                    status = "ON" if config.SHOW_OPENAI_INTERACTIONS else "OFF"
                    ui.add_conversation_line(f"\n🔄 OpenAI interaction display toggled: {status}")
                    ui.refresh_display(bridge.get_statistics())
                    continue
                elif user_input.lower() == 'help':
                    ui.add_conversation_line("\n📚 Available Commands:")
                    ui.add_conversation_line("  'exit' or 'quit' - Exit the conversation")
                    ui.add_conversation_line("  'clear' - Clear conversation history")
                    ui.add_conversation_line("  'history' - Show conversation history")
                    ui.add_conversation_line("  'stats' - Refresh statistics display")
                    ui.add_conversation_line("  'save' - Save conversation to file")
                    ui.add_conversation_line("  'load' - Load previous conversation")
                    ui.add_conversation_line("  'resize' - Force refresh display")
                    ui.add_conversation_line("  'last' - Show full last response")
                    ui.add_conversation_line("  'toggle' - Toggle OpenAI interaction display")
                    ui.add_conversation_line("  'help' - Show this help message")
                    ui.refresh_display(bridge.get_statistics())
                    continue
                elif user_input.lower() == 'resize':
                    ui.last_terminal_size = None
                    ui.refresh_display(bridge.get_statistics())
                    ui.add_conversation_line("🔄 Display refreshed")
                    continue
                elif user_input.lower() == 'history':
                    if not bridge.conversation_history:
                        ui.add_conversation_line("📜 No conversation history yet.")
                    else:
                        ui.add_conversation_line("\n📜 Conversation History:")
                        ui.add_conversation_line("-" * 40)
                        for i, msg in enumerate(bridge.conversation_history):
                            role = msg['role'].upper()
                            if isinstance(msg['content'], str):
                                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                            else:
                                content = "[Complex message with tool use]"
                            ui.add_conversation_line(f"{i+1}. {role}: {content}")
                        ui.add_conversation_line("-" * 40)
                    ui.refresh_display(bridge.get_statistics())
                    continue
                elif user_input.lower() == 'stats':
                    ui.refresh_display(bridge.get_statistics())
                    continue
                elif user_input.lower() == 'save':
                    ui.move_cursor(term_size.lines - 1, 1)
                    print(" " * term_size.columns, end='\r')
                    filename = input("Enter filename (or press Enter for default): ").strip()
                    if not filename:
                        filename = "conversation_history.json"
                    if bridge.save_conversation(filename):
                        ui.add_conversation_line(f"💾 Conversation saved to {filename}")
                    else:
                        ui.add_conversation_line("❌ Failed to save conversation")
                    ui.refresh_display(bridge.get_statistics())
                    continue
                elif user_input.lower() == 'load':
                    ui.move_cursor(term_size.lines - 1, 1)
                    print(" " * term_size.columns, end='\r')
                    filename = input("Enter filename to load (or press Enter for default): ").strip()
                    if not filename:
                        filename = "conversation_history.json"
                    if bridge.load_conversation(filename):
                        ui.add_conversation_line(f"📂 Loaded conversation from {filename}")
                        ui.add_conversation_line("Note: Token statistics were reset")
                    else:
                        ui.add_conversation_line("❌ Failed to load conversation")
                    ui.refresh_display(bridge.get_statistics())
                    continue
                
                ui.add_conversation_line(f"\nYou: {user_input}")
                ui.refresh_display(bridge.get_statistics())
                
                ui.add_conversation_line("Claude: [thinking...]")
                ui.refresh_display(bridge.get_statistics())
                
                response = bridge.chat_with_claude(user_input, ui=ui)
                
                if ui.conversation_lines and ui.conversation_lines[-1] == "Claude: [thinking...]":
                    ui.conversation_lines.pop()
                
                if config.SHOW_OPENAI_INTERACTIONS:
                    ui.add_conversation_line(f"Claude: {response}")
                else:
                    ui.add_conversation_line(f"Claude: {response}")
                ui.refresh_display(bridge.get_statistics())
        
        else:
            # Original interactive mode without terminal UI
            print("\n💬 Interactive Chat Mode")
            print("=" * 50)
            openai_status = "ON" if config.SHOW_OPENAI_INTERACTIONS else "OFF"
            print(f"Commands: exit, clear, history, stats, save, load, help, last, toggle")
            print(f"(OpenAI interactions: {openai_status})")
            print("=" * 50)
            
            while True:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    bridge.print_statistics()
                    break
                elif user_input.lower() == 'clear':
                    bridge.clear_conversation()
                    print("🗑️  Conversation history cleared. Starting fresh!")
                    continue
                elif user_input.lower() == 'history':
                    if not bridge.conversation_history:
                        print("📜 No conversation history yet.")
                    else:
                        print("\n📜 Conversation History:")
                        print("-" * 40)
                        for i, msg in enumerate(bridge.conversation_history):
                            role = msg['role'].upper()
                            if isinstance(msg['content'], str):
                                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                            else:
                                content = "[Complex message with tool use]"
                            print(f"{i+1}. {role}: {content}")
                        print("-" * 40)
                    continue
                elif user_input.lower() == 'stats':
                    bridge.print_statistics()
                    continue
                elif user_input.lower() == 'save':
                    filename = input("Enter filename (or press Enter for default): ").strip()
                    if not filename:
                        filename = "conversation_history.json"
                    if bridge.save_conversation(filename):
                        print(f"💾 Conversation saved to {filename}")
                    else:
                        print("❌ Failed to save conversation")
                    continue
                elif user_input.lower() == 'load':
                    filename = input("Enter filename to load (or press Enter for default): ").strip()
                    if not filename:
                        filename = "conversation_history.json"
                    if bridge.load_conversation(filename):
                        print(f"📂 Loaded conversation from {filename}")
                        print("Note: Token statistics were reset (historical stats not restored)")
                    else:
                        print("❌ Failed to load conversation")
                    continue
                elif user_input.lower() == 'last':
                    last_assistant_msg = None
                    for msg in reversed(bridge.conversation_history):
                        if msg['role'] == 'assistant' and isinstance(msg['content'], str):
                            last_assistant_msg = msg['content']
                            break
                    
                    if last_assistant_msg:
                        print("\n📜 Last Claude Response (Full):")
                        print("=" * 50)
                        print(last_assistant_msg)
                        print("=" * 50)
                    else:
                        print("No assistant response found yet.")
                    continue
                elif user_input.lower() == 'toggle':
                    config.SHOW_OPENAI_INTERACTIONS = not config.SHOW_OPENAI_INTERACTIONS
                    status = "ON" if config.SHOW_OPENAI_INTERACTIONS else "OFF"
                    print(f"\n🔄 OpenAI interaction display toggled: {status}")
                    continue
                elif user_input.lower() == 'help':
                    print("\n📚 Available Commands:")
                    print("  'exit' or 'quit' - Exit the conversation")
                    print("  'clear' - Clear conversation history")
                    print("  'history' - Show conversation history")
                    print("  'stats' - Show token usage statistics")
                    print("  'save' - Save conversation to file")
                    print("  'load' - Load previous conversation")
                    print("  'last' - Show full last response")
                    print("  'toggle' - Toggle OpenAI interaction display")
                    print("  'help' - Show this help message")
                    continue
                
                print("\nClaude: ", end="")
                response = bridge.chat_with_claude(user_input)
                print(response)
                
                if bridge.message_count % 5 == 0:
                    stats = bridge.get_statistics()
                    print(f"\n[Tokens used: {stats['total_tokens']:,} | Messages: {stats['message_count']}]")
    
    finally:
        if 'bridge' in locals() and hasattr(bridge, 'get_statistics'):
            stats = bridge.get_statistics()
            logger.info(f"Session ended. Total messages: {stats['message_count']}, Total tokens: {stats['total_tokens']}")
        logger.info(f"Log saved to: {current_log_file}")


if __name__ == "__main__":
    main()
