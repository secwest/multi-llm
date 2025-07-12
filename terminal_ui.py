#!/usr/bin/env python3
"""
Terminal UI Module for Multi-LLM Bridge
Version: 5.1.0

This module contains all terminal UI components including multi-window and multi-pane layouts.
Updated: Compact display with statistics integrated into title bars
"""

import os
import sys
import time
import shutil
import signal
import textwrap
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

# Get logger from main module
logger = logging.getLogger("LLM_Bridge")

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
    
    # Bright background colors
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_RED = '\033[101m'
    BG_BRIGHT_GREEN = '\033[102m'
    BG_BRIGHT_YELLOW = '\033[103m'
    BG_BRIGHT_BLUE = '\033[104m'
    BG_BRIGHT_MAGENTA = '\033[105m'
    BG_BRIGHT_CYAN = '\033[106m'
    BG_BRIGHT_WHITE = '\033[107m'
    
    # Text styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'
    STRIKETHROUGH = '\033[9m'

# Box drawing characters (Unicode)
class BoxChars:
    # Single line
    TL = '┌'  # Top-left
    TR = '┐'  # Top-right
    BL = '└'  # Bottom-left
    BR = '┘'  # Bottom-right
    H = '─'   # Horizontal
    V = '│'   # Vertical
    T = '┬'   # T-junction top
    B = '┴'   # T-junction bottom
    L = '├'   # T-junction left
    R = '┤'   # T-junction right
    X = '┼'   # Cross
    
    # Double line
    D_TL = '╔'
    D_TR = '╗'
    D_BL = '╚'
    D_BR = '╝'
    D_H = '═'
    D_V = '║'
    
    # Mixed
    S_TO_D_L = '╞'
    S_TO_D_R = '╡'
    D_TO_S_L = '╟'
    D_TO_S_R = '╢'

# LLM Type for UI
class LLMType(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"

# LLM Color mapping with icons
LLM_COLORS = {
    LLMType.CLAUDE: {
        'bg': Colors.BG_MAGENTA,
        'fg': Colors.WHITE,
        'icon': '🧠',
        'accent': Colors.BRIGHT_MAGENTA
    },
    LLMType.OPENAI: {
        'bg': Colors.BG_BLUE,
        'fg': Colors.WHITE,
        'icon': '🤖',
        'accent': Colors.BRIGHT_BLUE
    },
    LLMType.GEMINI: {
        'bg': Colors.BG_GREEN,
        'fg': Colors.WHITE,
        'icon': '✨',
        'accent': Colors.BRIGHT_GREEN
    }
}


class BaseTerminalUI(ABC):
    """Abstract base class for terminal UI implementations"""
    
    def __init__(self):
        self.last_terminal_size = None
        self.install_resize_handler()
        
    @abstractmethod
    def refresh_display(self):
        """Refresh the entire display"""
        pass
    
    @abstractmethod
    def add_message(self, source: str, message: str, message_type: str = "info"):
        """Add a message to the display"""
        pass
    
    def install_resize_handler(self):
        """Install handler for terminal resize events"""
        try:
            if hasattr(signal, 'SIGWINCH'):
                signal.signal(signal.SIGWINCH, self._handle_resize)
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
    
    def move_cursor(self, row: int, col: int):
        """Move cursor to specific position"""
        print(f"\033[{row};{col}H", end='')
        sys.stdout.flush()
    
    def get_terminal_size(self) -> Tuple[int, int]:
        """Get terminal dimensions (columns, lines)"""
        try:
            size = shutil.get_terminal_size()
            return (size.columns, size.lines)
        except:
            return (120, 40)
    
    def hide_cursor(self):
        """Hide terminal cursor"""
        print('\033[?25l', end='')
        sys.stdout.flush()
    
    def show_cursor(self):
        """Show terminal cursor"""
        print('\033[?25h', end='')
        sys.stdout.flush()
    
    def save_cursor_position(self):
        """Save current cursor position"""
        print('\033[s', end='')
        sys.stdout.flush()
    
    def restore_cursor_position(self):
        """Restore saved cursor position"""
        print('\033[u', end='')
        sys.stdout.flush()


class StandardUI(BaseTerminalUI):
    """Standard single-window terminal UI"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.messages = []
        self.max_messages = 1000
        
    def add_message(self, source: str, message: str, message_type: str = "info"):
        """Add a message to the display"""
        timestamp = ""
        if self.config.get('SHOW_TIMESTAMPS', True):
            timestamp = datetime.now().strftime("[%H:%M:%S] ")
        
        # Format message based on type
        if message_type == "user":
            formatted = f"{timestamp}{Colors.GREEN}{Colors.BOLD}You:{Colors.RESET} {message}"
        elif message_type == "assistant":
            formatted = f"{timestamp}{Colors.CYAN}{Colors.BOLD}{source}:{Colors.RESET} {message}"
        elif message_type == "error":
            formatted = f"{timestamp}{Colors.RED}{Colors.BOLD}Error:{Colors.RESET} {message}"
        elif message_type == "system":
            formatted = f"{timestamp}{Colors.YELLOW}{message}{Colors.RESET}"
        elif message_type == "tool":
            formatted = f"{timestamp}{Colors.MAGENTA}🔧 {message}{Colors.RESET}"
        else:
            formatted = f"{timestamp}{message}"
        
        self.messages.append(formatted)
        
        # Limit message history
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Print immediately in standard mode
        print(formatted)
    
    def refresh_display(self):
        """In standard mode, we don't need to refresh"""
        pass
    
    def show_statistics(self, stats: Dict[str, Any]):
        """Display statistics"""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}📊 Session Statistics:{Colors.RESET}")
        print("=" * 60)
        print(f"Total messages: {stats['total_messages']}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"Total cost: ${stats['total_cost']:.4f}")
        print(f"Session duration: {stats['session_duration_minutes']:.1f} minutes")
        
        if 'llm_stats' in stats:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}Per-Model Statistics:{Colors.RESET}")
            for llm_name, llm_stats in stats['llm_stats'].items():
                if llm_stats.get('message_count', 0) > 0:
                    print(f"\n{Colors.CYAN}{llm_name}:{Colors.RESET}")
                    print(f"  Messages: {llm_stats['message_count']}")
                    print(f"  Tokens: {llm_stats['total_tokens']:,}")
                    print(f"  Cost: ${llm_stats['total_cost']:.4f}")
        print("=" * 60)


class MultiWindowUI(BaseTerminalUI):
    """Multi-window terminal UI with separate windows for each LLM - Compact version"""
    
    def __init__(self, main_llm: LLMType, sub_llms: List[LLMType], config: Dict[str, Any]):
        super().__init__()
        self.main_llm = main_llm
        self.sub_llms = sub_llms
        self.config = config
        self.windows = {}  # Dict of LLMType -> window data
        
        # Initialize window data
        self.windows[main_llm] = {
            'lines': [],
            'stats': self._init_stats(main_llm),
            'is_main': True,
            'scroll_offset': 0,
            'title': f"{main_llm.value.upper()} [MAIN]"
        }
        
        for llm in sub_llms:
            self.windows[llm] = {
                'lines': [],
                'stats': self._init_stats(llm),
                'is_main': False,
                'scroll_offset': 0,
                'title': f"{llm.value.upper()} [SUB]"
            }
        
        self.max_lines_per_window = 2000
        self.status_height = 1  # Reduced from 3 to 1 - single line for title + stats
        self.divider_height = 1
        self.input_height = 2
        self.active_window = main_llm
        
    def _init_stats(self, llm_type: LLMType) -> Dict[str, Any]:
        """Initialize stats for an LLM"""
        return {
            'message_count': 0,
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'estimated_cost': 0.0,
            'last_model': self.config.get('SELECTED_MODELS', {}).get(llm_type, 'unknown')
        }
    
    def calculate_layout(self, term_height: int, term_width: int) -> Dict[str, Any]:
        """Calculate window layout based on number of windows"""
        available_height = term_height - self.input_height
        num_windows = len(self.windows)
        
        # Calculate height for each window
        total_dividers = (num_windows - 1) * self.divider_height
        window_space = available_height - total_dividers
        
        # Distribute space
        if num_windows == 1:
            window_heights = [window_space]
        elif num_windows == 2:
            # Main window gets 60%
            main_height = int(window_space * 0.6)
            window_heights = [main_height, window_space - main_height]
        elif num_windows == 3:
            # Main window gets 40%
            main_height = int(window_space * 0.4)
            remaining = window_space - main_height
            window_heights = [main_height, remaining // 2, remaining - (remaining // 2)]
        else:
            # Equal distribution
            base_height = window_space // num_windows
            window_heights = [base_height] * num_windows
            # Adjust for rounding
            window_heights[-1] = window_space - sum(window_heights[:-1])
        
        # Build layout
        layout = {'windows': [], 'input_row': term_height - 1}
        current_row = 1
        
        # Main window first
        window_list = [self.main_llm] + self.sub_llms
        
        for i, (llm_type, height) in enumerate(zip(window_list, window_heights)):
            if i > 0:
                current_row += self.divider_height
            
            layout['windows'].append({
                'llm': llm_type,
                'start_row': current_row,
                'status_rows': self.status_height,
                'content_start': current_row + self.status_height,
                'content_height': max(1, height - self.status_height),
                'total_height': height,
                'is_active': llm_type == self.active_window
            })
            current_row += height
        
        return layout
    
    def draw_window(self, window_info: Dict[str, Any], term_width: int):
        """Draw a single window with compact header"""
        llm_type = window_info['llm']
        window_data = self.windows[llm_type]
        colors = LLM_COLORS[llm_type]
        
        # Draw compact status bar (title + stats on one line)
        self.draw_compact_status_bar(
            window_info['start_row'],
            term_width,
            llm_type,
            window_data['stats'],
            window_info['is_active']
        )
        
        # Draw content area
        lines = window_data['lines']
        scroll_offset = window_data.get('scroll_offset', 0)
        content_height = window_info['content_height']
        
        # Calculate visible range
        total_lines = len(lines)
        if scroll_offset > 0:
            start_idx = max(0, total_lines - content_height - scroll_offset)
            end_idx = max(0, total_lines - scroll_offset)
        else:
            start_idx = max(0, total_lines - content_height)
            end_idx = total_lines
        
        display_lines = lines[start_idx:end_idx]
        
        # Clear and draw content
        for i in range(content_height):
            row = window_info['content_start'] + i
            self.move_cursor(row, 1)
            
            if i < len(display_lines):
                line = display_lines[i]
                # Truncate long lines
                if len(line) > term_width - 2:
                    line = line[:term_width - 5] + "..."
                print(line + " " * (term_width - len(line)))
            else:
                print(" " * term_width)
    
    def draw_compact_status_bar(self, row: int, width: int, llm_type: LLMType, stats: Dict[str, Any], is_active: bool):
        """Draw compact status bar with all info on one line"""
        colors = LLM_COLORS[llm_type]
        
        self.move_cursor(row, 1)
        
        # Build status line components
        title = f" {colors['icon']} {self.windows[llm_type]['title']}"
        model = stats['last_model']
        
        # Log what we're drawing
        logger.debug(f"Drawing status bar for {llm_type.value}: {title}")
        
        # Shorten model name if needed
        if len(model) > 20:
            model = model[:17] + "..."
        
        # Compact stats
        msg = f"M:{stats['message_count']}"
        tok = f"T:{stats['total_tokens']:,}"
        if stats['total_tokens'] >= 1000000:
            tok = f"T:{stats['total_tokens']/1000000:.1f}M"
        elif stats['total_tokens'] >= 1000:
            tok = f"T:{stats['total_tokens']/1000:.0f}k"
        
        cost = f"${stats['estimated_cost']:.2f}"
        
        # Add scroll indicator
        scroll = ""
        if self.windows[llm_type].get('scroll_offset', 0) > 0:
            scroll = " [SCROLL]"
        
        # Calculate available space
        stats_text = f" {msg} {tok} {cost}{scroll} "
        model_text = f" [{model}]"
        
        # Available space for spacing
        total_fixed_width = len(title) + len(model_text) + len(stats_text)
        spacing = max(1, width - total_fixed_width)
        
        # Build complete line
        if spacing > 0:
            line = f"{title}{model_text}{' ' * spacing}{stats_text}"
        else:
            # Not enough space, truncate model
            line = f"{title} {msg} {tok} {cost}{scroll} "
        
        # Ensure line fits
        if len(line) > width:
            line = line[:width]
        else:
            line = line.ljust(width)
        
        # Draw with appropriate colors
        if is_active:
            print(f"{colors['bg']}{Colors.BOLD}{Colors.WHITE}{line}{Colors.RESET}")
        else:
            print(f"{Colors.BG_BLUE}{Colors.BRIGHT_WHITE}{line}{Colors.RESET}")
    
    def draw_divider(self, row: int, width: int):
        """Draw divider between windows"""
        self.move_cursor(row, 1)
        print(f"{Colors.BLUE}{'═' * width}{Colors.RESET}")
    
    def add_line(self, llm_type: LLMType, line: str):
        """Add line to specific window with word wrapping"""
        if llm_type not in self.windows:
            logger.error(f"add_line: LLMType {llm_type} not found in windows! Available: {list(self.windows.keys())}")
            return
        
        logger.debug(f"add_line: Adding to {llm_type.value} window: {line[:50]}...")
        
        # Handle newlines
        if '\n' in line:
            for sub_line in line.split('\n'):
                if sub_line:  # Skip empty lines from split
                    self.add_line(llm_type, sub_line)
            return
        
        # Get terminal width for wrapping
        term_width, _ = self.get_terminal_size()
        max_width = max(40, term_width - 4)
        
        # Apply truncation if verbose is off
        if not self.config.get('VERBOSE_DISPLAY', True) and len(line) > max_width * 2:
            line = line[:max_width * 2 - 3] + "..."
        
        # Wrap long lines
        if len(line) > max_width:
            wrapped_lines = textwrap.wrap(
                line, 
                width=max_width,
                break_long_words=True,
                break_on_hyphens=True,
                expand_tabs=False,
                replace_whitespace=False
            )
            for wrapped_line in wrapped_lines:
                self.windows[llm_type]['lines'].append(wrapped_line)
        else:
            self.windows[llm_type]['lines'].append(line)
        
        # Reset scroll offset
        self.windows[llm_type]['scroll_offset'] = 0
        
        # Limit history
        while len(self.windows[llm_type]['lines']) > self.max_lines_per_window:
            self.windows[llm_type]['lines'].pop(0)
    
    def add_message(self, source: str, message: str, message_type: str = "info"):
        """Add a message to the appropriate window"""
        # Determine target window
        target_llm = None
        
        # Check if source matches any window's LLM type
        for llm_type in self.windows:
            # Match both the enum value and the display name
            if (llm_type.value.lower() == source.lower() or 
                llm_type.name.lower() == source.lower()):
                target_llm = llm_type
                break
        
        # If no match found, default to main LLM
        if not target_llm:
            # Check if it's a system message that should go to main
            if source.lower() in ['system', 'you', 'user']:
                target_llm = self.main_llm
            else:
                # Log unmapped source
                logger.warning(f"Could not map source '{source}' to any LLM window, using main LLM")
                target_llm = self.main_llm
        
        # Format message
        timestamp = ""
        if self.config.get('SHOW_TIMESTAMPS', True):
            timestamp = datetime.now().strftime("[%H:%M:%S] ")
        
        # Apply formatting based on message type
        if message_type == "user":
            formatted = f"{timestamp}{Colors.GREEN}{Colors.BOLD}You:{Colors.RESET} {message}"
        elif message_type == "assistant":
            formatted = f"{timestamp}{Colors.CYAN}{Colors.BOLD}{source}:{Colors.RESET} {message}"
        elif message_type == "error":
            formatted = f"{timestamp}{Colors.RED}{Colors.BOLD}Error:{Colors.RESET} {message}"
        elif message_type == "system":
            formatted = f"{timestamp}{Colors.YELLOW}{message}{Colors.RESET}"
        elif message_type == "tool":
            formatted = f"{timestamp}{Colors.MAGENTA}🔧 {message}{Colors.RESET}"
        else:
            formatted = f"{timestamp}{message}"
        
        self.add_line(target_llm, formatted)
    
    def update_stats(self, llm_type: LLMType, **kwargs):
        """Update statistics for specific LLM"""
        if llm_type not in self.windows:
            return
        
        stats = self.windows[llm_type]['stats']
        for key, value in kwargs.items():
            if key in stats:
                if key in ['message_count', 'total_tokens', 'input_tokens', 'output_tokens']:
                    stats[key] += value
                elif key == 'estimated_cost':
                    stats[key] += value
                else:
                    stats[key] = value
    
    def scroll(self, llm_type: LLMType, direction: str, amount: int = 5):
        """Scroll a specific window"""
        if llm_type not in self.windows:
            return
        
        if direction == 'up':
            self.windows[llm_type]['scroll_offset'] += amount
            max_scroll = max(0, len(self.windows[llm_type]['lines']) - 10)
            self.windows[llm_type]['scroll_offset'] = min(
                self.windows[llm_type]['scroll_offset'], max_scroll)
        elif direction == 'down':
            self.windows[llm_type]['scroll_offset'] -= amount
            self.windows[llm_type]['scroll_offset'] = max(
                0, self.windows[llm_type]['scroll_offset'])
        elif direction == 'reset':
            self.windows[llm_type]['scroll_offset'] = 0
    
    def set_active_window(self, llm_type: LLMType):
        """Set the active window"""
        if llm_type in self.windows:
            self.active_window = llm_type
    
    def refresh_display(self):
        """Refresh the entire multi-window display"""
        term_width, term_height = self.get_terminal_size()
        
        # Check for resize
        if self.last_terminal_size != (term_width, term_height):
            self.clear_screen()
            self.last_terminal_size = (term_width, term_height)
        
        # Calculate layout
        layout = self.calculate_layout(term_height, term_width)
        
        # Draw each window
        for i, window_info in enumerate(layout['windows']):
            self.draw_window(window_info, term_width)
            
            # Draw divider if not last window
            if i < len(layout['windows']) - 1:
                divider_row = window_info['start_row'] + window_info['total_height']
                self.draw_divider(divider_row, term_width)
        
        # Position cursor for input
        self.move_cursor(layout['input_row'], 1)
        print(f"{Colors.GREEN}You: {Colors.RESET}", end='')
        sys.stdout.flush()
    
    def clear_all(self):
        """Clear all windows"""
        for llm_type in self.windows:
            self.windows[llm_type]['lines'] = []
            self.windows[llm_type]['scroll_offset'] = 0
            self.windows[llm_type]['stats'] = self._init_stats(llm_type)
    
    def show_statistics(self, stats: Dict[str, Any]):
        """Display statistics in the main window"""
        self.add_message("system", "Session Statistics", "system")
        self.add_line(self.main_llm, "=" * 50)
        self.add_line(self.main_llm, f"Total messages: {stats['total_messages']}")
        self.add_line(self.main_llm, f"Total tokens: {stats['total_tokens']:,}")
        self.add_line(self.main_llm, f"Total cost: ${stats['total_cost']:.4f}")
        self.add_line(self.main_llm, f"Duration: {stats['session_duration_minutes']:.1f} minutes")
        
        if 'llm_stats' in stats:
            self.add_line(self.main_llm, "")
            self.add_message("system", "Per-Model Statistics", "system")
            for llm_name, llm_stats in stats['llm_stats'].items():
                if llm_stats.get('message_count', 0) > 0:
                    self.add_line(self.main_llm, f"\n{llm_name}:")
                    self.add_line(self.main_llm, f"  Messages: {llm_stats['message_count']}")
                    self.add_line(self.main_llm, f"  Tokens: {llm_stats.get('total_tokens', 0):,}")
                    self.add_line(self.main_llm, f"  Cost: ${llm_stats.get('total_cost', 0):.4f}")
        self.add_line(self.main_llm, "=" * 50)


class MultiPaneUI(BaseTerminalUI):
    """Advanced multi-pane terminal UI with sub-agent visualization - Compact version"""
    
    def __init__(self, main_llm: LLMType, sub_llms: List[LLMType], config: Dict[str, Any]):
        super().__init__()
        self.main_llm = main_llm
        self.sub_llms = sub_llms
        self.config = config
        
        # Main conversation window
        self.main_window = {
            'lines': [],
            'stats': self._init_stats(main_llm),
            'scroll_offset': 0
        }
        
        # Sub-agent panes
        self.sub_panes = {}
        for llm in sub_llms:
            self.sub_panes[llm] = {
                'query': 'No activity yet.',
                'response': '',
                'timestamp': None,
                'status': 'idle',  # idle, processing, complete, error
                'stats': self._init_stats(llm)
            }
        
        self.max_lines = 3000
        self.show_sub_panes = config.get('SHOW_SUB_AGENT_PANES', True)
        
    def _init_stats(self, llm_type: LLMType) -> Dict[str, Any]:
        """Initialize stats for an LLM"""
        return {
            'message_count': 0,
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'estimated_cost': 0.0,
            'last_model': self.config.get('SELECTED_MODELS', {}).get(llm_type, 'unknown')
        }
    
    def calculate_pane_layout(self, term_width: int, term_height: int) -> Dict[str, Any]:
        """Calculate layout for sub-agent panes"""
        if not self.show_sub_panes or not self.sub_llms:
            return {
                'show_panes': False,
                'main_start': 1,
                'main_height': term_height - 2
            }
        
        # Calculate pane dimensions
        num_panes = len(self.sub_llms)
        pane_height = min(15, max(10, term_height // 4))
        pane_width = term_width // num_panes
        
        return {
            'show_panes': True,
            'pane_height': pane_height,
            'pane_width': pane_width,
            'main_start': pane_height + 2,
            'main_height': term_height - pane_height - 3
        }
    
    def draw_sub_pane(self, llm_type: LLMType, col_start: int, pane_width: int, pane_height: int):
        """Draw a single sub-agent pane"""
        pane = self.sub_panes[llm_type]
        colors = LLM_COLORS[llm_type]
        
        # Draw pane border and title
        title = f" {colors['icon']} {llm_type.value.upper()} "
        if pane['status'] == 'processing':
            title += "[PROCESSING] "
        elif pane['status'] == 'error':
            title += "[ERROR] "
        
        # Top border
        self.move_cursor(1, col_start)
        border_color = Colors.BRIGHT_YELLOW if pane['status'] == 'processing' else Colors.BLUE
        print(f"{border_color}{BoxChars.TL}{title.center(pane_width - 2, BoxChars.H)}{BoxChars.TR}{Colors.RESET}")
        
        # Query section
        inner_width = pane_width - 4
        query_lines = textwrap.wrap(f"Q: {pane['query']}", width=inner_width)[:3]
        
        row = 2
        for line in query_lines:
            self.move_cursor(row, col_start)
            print(f"{border_color}{BoxChars.V}{Colors.RESET} {Colors.YELLOW}{line[:inner_width].ljust(inner_width)}{Colors.RESET} {border_color}{BoxChars.V}{Colors.RESET}")
            row += 1
        
        # Fill empty query rows
        while row < 5:
            self.move_cursor(row, col_start)
            print(f"{border_color}{BoxChars.V}{Colors.RESET} {' ' * inner_width} {border_color}{BoxChars.V}{Colors.RESET}")
            row += 1
        
        # Separator
        self.move_cursor(row, col_start)
        print(f"{border_color}{BoxChars.L}{BoxChars.H * (pane_width - 2)}{BoxChars.R}{Colors.RESET}")
        row += 1
        
        # Response section
        response_color = Colors.GREEN if pane['status'] == 'complete' else Colors.RED if pane['status'] == 'error' else Colors.WHITE
        response_text = pane['response'] if pane['response'] else '[Waiting...]' if pane['status'] == 'processing' else ''
        response_lines = textwrap.wrap(f"A: {response_text}", width=inner_width)[:(pane_height - row - 2)]
        
        for line in response_lines:
            self.move_cursor(row, col_start)
            print(f"{border_color}{BoxChars.V}{Colors.RESET} {response_color}{line[:inner_width].ljust(inner_width)}{Colors.RESET} {border_color}{BoxChars.V}{Colors.RESET}")
            row += 1
        
        # Fill empty response rows
        while row < pane_height - 1:
            self.move_cursor(row, col_start)
            print(f"{border_color}{BoxChars.V}{Colors.RESET} {' ' * inner_width} {border_color}{BoxChars.V}{Colors.RESET}")
            row += 1
        
        # Stats row
        self.move_cursor(row, col_start)
        stats = pane['stats']
        stats_text = f"T:{stats['total_tokens']} $:{stats['estimated_cost']:.3f}"
        if len(stats_text) > inner_width:
            stats_text = f"${stats['estimated_cost']:.3f}"
        print(f"{border_color}{BoxChars.V}{Colors.RESET} {Colors.DIM}{stats_text.ljust(inner_width)}{Colors.RESET} {border_color}{BoxChars.V}{Colors.RESET}")
        row += 1
        
        # Bottom border
        self.move_cursor(row, col_start)
        print(f"{border_color}{BoxChars.BL}{BoxChars.H * (pane_width - 2)}{BoxChars.BR}{Colors.RESET}")
    
    def draw_main_conversation(self, start_row: int, height: int, width: int):
        """Draw the main conversation pane with compact header"""
        colors = LLM_COLORS[self.main_llm]
        stats = self.main_window['stats']
        
        # Compact header - all info on one line
        self.move_cursor(start_row, 1)
        
        # Build header components
        title = f" {colors['icon']} {self.main_llm.value.upper()}"
        model = stats['last_model']
        if len(model) > 20:
            model = model[:17] + "..."
        
        # Compact stats
        msg = f"M:{stats['message_count']}"
        tok = f"T:{stats['total_tokens']:,}"
        if stats['total_tokens'] >= 1000000:
            tok = f"T:{stats['total_tokens']/1000000:.1f}M"
        elif stats['total_tokens'] >= 1000:
            tok = f"T:{stats['total_tokens']/1000:.0f}k"
        
        cost = f"${stats['estimated_cost']:.3f}"
        
        # Add scroll indicator
        scroll = ""
        if self.main_window.get('scroll_offset', 0) > 0:
            scroll = " [SCROLL]"
        
        # Build complete header line
        stats_text = f" {msg} {tok} {cost}{scroll} "
        model_text = f" [{model}]"
        
        # Calculate spacing
        total_fixed_width = len(title) + len(model_text) + len(stats_text)
        spacing = max(1, width - total_fixed_width)
        
        if spacing > 0:
            header_line = f"{title}{model_text}{' ' * spacing}{stats_text}"
        else:
            # Not enough space, skip model
            header_line = f"{title} {msg} {tok} {cost}{scroll} "
        
        # Ensure it fits
        if len(header_line) > width:
            header_line = header_line[:width]
        else:
            header_line = header_line.ljust(width)
        
        print(f"{colors['bg']}{Colors.BOLD}{Colors.WHITE}{header_line}{Colors.RESET}")
        
        # Content area
        content_start = start_row + 2  # Only 2 rows for header (title + separator)
        content_height = height - 3
        
        # Separator line
        self.move_cursor(start_row + 1, 1)
        print(f"{Colors.BLUE}{'─' * width}{Colors.RESET}")
        
        # Get visible lines
        lines = self.main_window['lines']
        scroll_offset = self.main_window.get('scroll_offset', 0)
        
        total_lines = len(lines)
        if scroll_offset > 0:
            start_idx = max(0, total_lines - content_height - scroll_offset)
            end_idx = max(0, total_lines - scroll_offset)
        else:
            start_idx = max(0, total_lines - content_height)
            end_idx = total_lines
        
        display_lines = lines[start_idx:end_idx]
        
        # Clear and draw content
        for i in range(content_height):
            self.move_cursor(content_start + i, 1)
            if i < len(display_lines):
                line = display_lines[i]
                if len(line) > width - 2:
                    line = line[:width - 5] + "..."
                print(f" {line}" + " " * (width - len(line) - 2))
            else:
                print(" " * width)
        
        # Bottom border
        self.move_cursor(start_row + height - 1, 1)
        print(f"{Colors.BLUE}{'─' * width}{Colors.RESET}")
    
    def update_sub_pane(self, llm_type: LLMType, query: str = None, response: str = None, 
                       status: str = None, stats_update: Dict[str, Any] = None):
        """Update sub-agent pane content"""
        if llm_type not in self.sub_panes:
            return
        
        pane = self.sub_panes[llm_type]
        
        if query is not None:
            pane['query'] = query
            pane['timestamp'] = datetime.now()
        
        if response is not None:
            pane['response'] = response
        
        if status is not None:
            pane['status'] = status
        
        if stats_update:
            stats = pane['stats']
            for key, value in stats_update.items():
                if key in ['message_count', 'total_tokens', 'input_tokens', 'output_tokens']:
                    stats[key] += value
                elif key == 'estimated_cost':
                    stats[key] += value
                else:
                    stats[key] = value
    
    def add_line(self, line: str):
        """Add line to main conversation window"""
        if '\n' in line:
            for sub_line in line.split('\n'):
                if sub_line:
                    self.add_line(sub_line)
            return
        
        # Get terminal width for wrapping
        term_width, _ = self.get_terminal_size()
        max_width = max(40, term_width - 4)
        
        # Apply truncation if verbose is off
        if not self.config.get('VERBOSE_DISPLAY', True) and len(line) > max_width * 2:
            line = line[:max_width * 2 - 3] + "..."
        
        # Wrap long lines
        if len(line) > max_width:
            wrapped_lines = textwrap.wrap(
                line,
                width=max_width,
                break_long_words=True,
                break_on_hyphens=True
            )
            for wrapped_line in wrapped_lines:
                self.main_window['lines'].append(wrapped_line)
        else:
            self.main_window['lines'].append(line)
        
        # Reset scroll offset
        self.main_window['scroll_offset'] = 0
        
        # Limit history
        while len(self.main_window['lines']) > self.max_lines:
            self.main_window['lines'].pop(0)
    
    def add_message(self, source: str, message: str, message_type: str = "info"):
        """Add a message to the main conversation"""
        timestamp = ""
        if self.config.get('SHOW_TIMESTAMPS', True):
            timestamp = datetime.now().strftime("[%H:%M:%S] ")
        
        # Format message based on type
        if message_type == "user":
            formatted = f"{timestamp}{Colors.GREEN}{Colors.BOLD}You:{Colors.RESET} {message}"
        elif message_type == "assistant":
            formatted = f"{timestamp}{Colors.CYAN}{Colors.BOLD}{source}:{Colors.RESET} {message}"
        elif message_type == "error":
            formatted = f"{timestamp}{Colors.RED}{Colors.BOLD}Error:{Colors.RESET} {message}"
        elif message_type == "system":
            formatted = f"{timestamp}{Colors.YELLOW}{message}{Colors.RESET}"
        elif message_type == "tool":
            formatted = f"{timestamp}{Colors.MAGENTA}🔧 {message}{Colors.RESET}"
        elif message_type == "thinking":
            formatted = f"{timestamp}{Colors.DIM}{message}{Colors.RESET}"
        else:
            formatted = f"{timestamp}{message}"
        
        self.add_line(formatted)
    
    def update_main_stats(self, **kwargs):
        """Update main window statistics"""
        stats = self.main_window['stats']
        for key, value in kwargs.items():
            if key in stats:
                if key in ['message_count', 'total_tokens', 'input_tokens', 'output_tokens']:
                    stats[key] += value
                elif key == 'estimated_cost':
                    stats[key] += value
                else:
                    stats[key] = value
    
    def scroll(self, direction: str, amount: int = 5):
        """Scroll the main conversation window"""
        if direction == 'up':
            self.main_window['scroll_offset'] += amount
            max_scroll = max(0, len(self.main_window['lines']) - 10)
            self.main_window['scroll_offset'] = min(
                self.main_window['scroll_offset'], max_scroll)
        elif direction == 'down':
            self.main_window['scroll_offset'] -= amount
            self.main_window['scroll_offset'] = max(
                0, self.main_window['scroll_offset'])
        elif direction == 'reset':
            self.main_window['scroll_offset'] = 0
    
    def toggle_sub_panes(self):
        """Toggle sub-agent panes visibility"""
        self.show_sub_panes = not self.show_sub_panes
        self.clear_screen()
    
    def refresh_display(self):
        """Refresh the entire multi-pane display"""
        term_width, term_height = self.get_terminal_size()
        
        # Check for resize
        if self.last_terminal_size != (term_width, term_height):
            self.clear_screen()
            self.last_terminal_size = (term_width, term_height)
        
        # Calculate layout
        layout = self.calculate_pane_layout(term_width, term_height)
        
        # Draw sub-agent panes if enabled
        if layout['show_panes']:
            for i, llm_type in enumerate(self.sub_llms):
                col_start = i * layout['pane_width'] + 1
                self.draw_sub_pane(llm_type, col_start, layout['pane_width'], layout['pane_height'])
        
        # Draw main conversation
        self.draw_main_conversation(
            layout['main_start'],
            layout['main_height'],
            term_width
        )
        
        # Position cursor for input
        self.move_cursor(term_height, 1)
        print(f"{Colors.GREEN}You: {Colors.RESET}", end='')
        sys.stdout.flush()
    
    def clear_all(self):
        """Clear all windows and panes"""
        self.main_window['lines'] = []
        self.main_window['scroll_offset'] = 0
        self.main_window['stats'] = self._init_stats(self.main_llm)
        
        for llm_type in self.sub_panes:
            self.sub_panes[llm_type] = {
                'query': 'No activity yet.',
                'response': '',
                'timestamp': None,
                'status': 'idle',
                'stats': self._init_stats(llm_type)
            }
    
    def show_statistics(self, overall_stats: Dict[str, Any]):
        """Display statistics in the main window"""
        self.add_line("")
        self.add_message("system", "Session Statistics", "system")
        self.add_line("=" * 50)
        self.add_line(f"Total messages: {overall_stats['total_messages']}")
        self.add_line(f"Total tokens: {overall_stats['total_tokens']:,}")
        self.add_line(f"Total cost: ${overall_stats['total_cost']:.4f}")
        self.add_line(f"Duration: {overall_stats['session_duration_minutes']:.1f} minutes")
        
        if 'llm_stats' in overall_stats:
            self.add_line("")
            self.add_message("system", "Per-Model Statistics", "system")
            for llm_name, llm_stats in overall_stats['llm_stats'].items():
                if llm_stats.get('message_count', 0) > 0:
                    self.add_line(f"\n{llm_name}:")
                    self.add_line(f"  Messages: {llm_stats['message_count']}")
                    self.add_line(f"  Tokens: {llm_stats.get('total_tokens', 0):,}")
                    self.add_line(f"  Cost: ${llm_stats.get('total_cost', 0):.4f}")
        self.add_line("=" * 50)


# Factory function to create appropriate UI
def create_ui(display_mode: str, main_llm: LLMType, sub_llms: List[LLMType], config: Dict[str, Any]) -> BaseTerminalUI:
    """Create the appropriate UI based on display mode"""
    if display_mode == "multi-window":
        return MultiWindowUI(main_llm, sub_llms, config)
    elif display_mode == "multi-pane":
        return MultiPaneUI(main_llm, sub_llms, config)
    else:
        return StandardUI(config)


# Progress indicator for long operations
class ProgressIndicator:
    """Simple progress indicator for terminal"""
    
    def __init__(self, message: str = "Processing"):
        self.message = message
        self.symbols = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current = 0
        self.active = False
    
    def start(self):
        """Start the progress indicator"""
        self.active = True
        self._update()
    
    def stop(self):
        """Stop the progress indicator"""
        self.active = False
        print("\r" + " " * (len(self.message) + 10) + "\r", end='')
        sys.stdout.flush()
    
    def _update(self):
        """Update the progress indicator"""
        if self.active:
            symbol = self.symbols[self.current % len(self.symbols)]
            print(f"\r{symbol} {self.message}...", end='')
            sys.stdout.flush()
            self.current += 1


# Utility function for terminal capabilities check
def check_terminal_capabilities() -> Dict[str, bool]:
    """Check terminal capabilities"""
    capabilities = {
        'color': True,  # Most modern terminals support color
        'unicode': True,  # Most modern terminals support Unicode
        'size': True  # Can get terminal size
    }
    
    # Check if running in a basic terminal
    term = os.environ.get('TERM', '')
    if term in ['dumb', 'unknown']:
        capabilities['color'] = False
        capabilities['unicode'] = False
    
    # Check if running in Windows console (older versions)
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Enable virtual terminal processing for Windows 10+
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass
    
    # Test Unicode support
    try:
        test_char = "█"
        test_char.encode(sys.stdout.encoding or 'utf-8')
    except:
        capabilities['unicode'] = False
    
    return capabilities
