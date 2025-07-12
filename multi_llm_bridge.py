#!/usr/bin/env python3
"""
Multi-LLM Bridge - Ultimate Edition with Configuration Persistence
Version: 5.1.0 (2025-01-11)

Main script that orchestrates multiple LLMs with flexible architecture and advanced UI.
Enhanced: Saves and loads LLM configuration for convenience

Features:
- Saves last used configuration (models, display mode, etc.)
- Prompts to reuse previous configuration on startup
- Any LLM can be the main AI (Claude, OpenAI, or Gemini)
- Main AI can call other LLMs as tools
- Three display modes: Standard, Multi-window, Multi-pane
- Full conversation persistence
- Comprehensive statistics and cost tracking
- Advanced terminal UI with scrolling and proper formatting

Usage:
    python multi_llm_bridge.py

Requires:
    - llm_interfaces.py
    - terminal_ui.py
"""

import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum

# Check if modules are available
try:
    from llm_interfaces import (
        LLMInterface, ClaudeInterface, OpenAIInterface, GeminiInterface,
        create_llm_interface, get_model_info, format_model_comparison,
        PRICE_MAPPING, CLAUDE_MODELS, GEMINI_MODELS
    )
    from terminal_ui import (
        BaseTerminalUI, StandardUI, MultiWindowUI, MultiPaneUI,
        create_ui, Colors, LLMType, check_terminal_capabilities
    )
except ImportError as e:
    print(f"Error: Required modules not found. {e}")
    print("Make sure llm_interfaces.py and terminal_ui.py are in the same directory.")
    sys.exit(1)

# Version
VERSION = "5.1.0"
VERSION_DATE = "2025-01-11"

# Configuration paths
CONFIG_DIR = Path.home() / ".multi_llm_bridge"
LAST_CONFIG_FILE = CONFIG_DIR / "last_config.json"

# Configuration class
class Config:
    """Global configuration"""
    MAIN_LLM = None  # Which LLM is the main one
    ENABLED_SUB_LLMS = []  # Which LLMs can be called as tools
    SELECTED_MODELS = {}  # Model selections for each LLM type
    VERBOSE_DISPLAY = True  # Show full content vs truncated
    SHOW_TIMESTAMPS = True  # Show timestamps on messages
    SHOW_SUB_AGENT_PANES = True  # Show multi-pane layout
    DISPLAY_MODE = "standard"  # "standard", "multi-window", "multi-pane"
    AUTO_SAVE = False  # Auto-save conversations
    SAVE_INTERVAL = 300  # Auto-save interval in seconds

config = Config()

# Set up logging
def setup_logging():
    """Set up logging with file handler"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"multi_llm_bridge_{timestamp}.log"
    
    logger = logging.getLogger("LLM_Bridge")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler for errors only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger, log_file

# Initialize logging
logger, current_log_file = setup_logging()
logger.info(f"Multi-LLM Bridge v{VERSION} starting...")
logger.info(f"Session started. Logging to: {current_log_file}")


def save_last_configuration():
    """Save the current configuration for future use"""
    try:
        # Ensure config directory exists
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'version': VERSION,
            'main_llm': config.MAIN_LLM.value if config.MAIN_LLM else None,
            'enabled_sub_llms': [llm.value for llm in config.ENABLED_SUB_LLMS],
            'selected_models': {
                llm_type.value: model 
                for llm_type, model in config.SELECTED_MODELS.items()
            },
            'display_mode': config.DISPLAY_MODE,
            'verbose_display': config.VERBOSE_DISPLAY,
            'show_timestamps': config.SHOW_TIMESTAMPS,
            'show_sub_agent_panes': config.SHOW_SUB_AGENT_PANES,
            'auto_save': config.AUTO_SAVE,
            'save_interval': config.SAVE_INTERVAL,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(LAST_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {LAST_CONFIG_FILE}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {str(e)}", exc_info=True)
        return False


def load_last_configuration() -> Optional[Dict[str, Any]]:
    """Load the last saved configuration"""
    try:
        if not LAST_CONFIG_FILE.exists():
            return None
        
        with open(LAST_CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Check version compatibility
        saved_version = config_data.get('version', '0.0.0')
        if saved_version.split('.')[0] != VERSION.split('.')[0]:
            logger.warning(f"Configuration version mismatch: saved={saved_version}, current={VERSION}")
            return None
        
        return config_data
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}", exc_info=True)
        return None


def apply_saved_configuration(config_data: Dict[str, Any], available_keys: Dict[str, bool]) -> bool:
    """Apply saved configuration if all required LLMs are available"""
    try:
        # Check if main LLM is available
        main_llm_str = config_data.get('main_llm')
        if not main_llm_str or not available_keys.get(main_llm_str, False):
            print(f"{Colors.YELLOW}⚠️  Main LLM '{main_llm_str}' is not available with current API keys{Colors.RESET}")
            return False
        
        # Check if all sub-LLMs are available
        sub_llms_str = config_data.get('enabled_sub_llms', [])
        for sub_llm_str in sub_llms_str:
            if not available_keys.get(sub_llm_str, False):
                print(f"{Colors.YELLOW}⚠️  Sub-LLM '{sub_llm_str}' is not available with current API keys{Colors.RESET}")
                return False
        
        # Apply configuration
        config.MAIN_LLM = LLMType(main_llm_str)
        config.ENABLED_SUB_LLMS = [LLMType(llm_str) for llm_str in sub_llms_str]
        
        # Apply model selections
        saved_models = config_data.get('selected_models', {})
        for llm_str, model in saved_models.items():
            llm_type = LLMType(llm_str)
            config.SELECTED_MODELS[llm_type] = model
        
        # Apply display settings
        config.DISPLAY_MODE = config_data.get('display_mode', 'standard')
        config.VERBOSE_DISPLAY = config_data.get('verbose_display', True)
        config.SHOW_TIMESTAMPS = config_data.get('show_timestamps', True)
        config.SHOW_SUB_AGENT_PANES = config_data.get('show_sub_agent_panes', True)
        config.AUTO_SAVE = config_data.get('auto_save', False)
        config.SAVE_INTERVAL = config_data.get('save_interval', 300)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply configuration: {str(e)}", exc_info=True)
        return False


def display_saved_configuration(config_data: Dict[str, Any]):
    """Display saved configuration details"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}📋 Previous Configuration:{Colors.RESET}")
    print("=" * 60)
    
    # Time info
    timestamp = config_data.get('timestamp', 'Unknown')
    try:
        saved_time = datetime.fromisoformat(timestamp)
        time_diff = datetime.now() - saved_time
        if time_diff.days > 0:
            time_str = f"{time_diff.days} days ago"
        elif time_diff.seconds > 3600:
            time_str = f"{time_diff.seconds // 3600} hours ago"
        else:
            time_str = f"{time_diff.seconds // 60} minutes ago"
        print(f"Saved: {saved_time.strftime('%Y-%m-%d %H:%M')} ({time_str})")
    except:
        print(f"Saved: {timestamp}")
    
    # LLM configuration
    print(f"\nMain LLM: {Colors.GREEN}{config_data.get('main_llm', 'None').upper()}{Colors.RESET}")
    
    sub_llms = config_data.get('enabled_sub_llms', [])
    if sub_llms:
        print(f"Sub-LLMs: {Colors.BLUE}{', '.join(llm.upper() for llm in sub_llms)}{Colors.RESET}")
    else:
        print(f"Sub-LLMs: {Colors.DIM}None (standalone mode){Colors.RESET}")
    
    # Model selections
    print(f"\n{Colors.YELLOW}Models:{Colors.RESET}")
    for llm_str, model in config_data.get('selected_models', {}).items():
        print(f"  {llm_str.upper()}: {model}")
    
    # Display mode
    display_mode = config_data.get('display_mode', 'standard')
    print(f"\nDisplay: {display_mode.replace('-', ' ').title()}")
    
    # Settings
    print(f"\n{Colors.YELLOW}Settings:{Colors.RESET}")
    print(f"  Auto-save: {'Yes' if config_data.get('auto_save', False) else 'No'}")
    print(f"  Timestamps: {'Yes' if config_data.get('show_timestamps', True) else 'No'}")
    print(f"  Verbose: {'Yes' if config_data.get('verbose_display', True) else 'No'}")
    
    print("=" * 60)


class MultiLLMBridge:
    """Universal bridge for multiple LLMs with multi-UI support"""
    
    def __init__(self, main_llm: LLMType, main_interface: LLMInterface, 
                 sub_llms: Dict[LLMType, LLMInterface], ui: BaseTerminalUI):
        self.main_llm = main_llm
        self.main_interface = main_interface
        self.sub_llms = sub_llms
        self.ui = ui
        self.conversation_history = []
        self.session_start = datetime.now()
        
        # Log the configuration
        logger.info(f"Bridge initialized - Main: {main_llm.value} ({type(main_interface).__name__})")
        for llm_type, interface in sub_llms.items():
            logger.info(f"  Sub-LLM: {llm_type.value} ({type(interface).__name__})")
        
        # Statistics
        self.stats = {llm: {
            'input_tokens': 0,
            'output_tokens': 0,
            'message_count': 0,
            'total_cost': 0.0,
            'tool_calls': 0,
            'errors': 0
        } for llm in [main_llm] + list(sub_llms.keys())}
        
        # Build tools for main LLM
        self.tools = self._build_tools()
        
        # Session metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_save_time = datetime.now()
        
        logger.info(f"Bridge initialized - Main: {main_llm.value}, Subs: {[llm.value for llm in sub_llms.keys()]}")
    
    def _build_tools(self) -> List[Dict[str, Any]]:
        """Build tool definitions for available sub-LLMs"""
        tools = []
        
        for llm_type, interface in self.sub_llms.items():
            # Check if the interface supports tools
            supports_tools = interface.supports_tools()
            
            # Query tool
            tool_name = f"query_{llm_type.value}"
            description = (f"Query the {llm_type.value.upper()} AI for its perspective, capabilities, "
                         f"or to delegate tasks. This model {'supports' if supports_tools else 'does not support'} "
                         f"tool/function calling.")
            
            logger.info(f"Building tool: {tool_name} for LLMType {llm_type}")
            
            tools.append({
                "name": tool_name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A clear, self-contained prompt for the AI"
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature (0-2)",
                            "default": 0.7
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens in response",
                            "default": 4096
                        },
                        "include_context": {
                            "type": "boolean",
                            "description": "Include conversation context",
                            "default": False
                        }
                    },
                    "required": ["prompt"]
                }
            })
            
            # Model info tool
            tools.append({
                "name": f"get_{llm_type.value}_info",
                "description": f"Get information about the current {llm_type.value.upper()} model",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            })
        
        # Add general tools
        tools.append({
            "name": "compare_models",
            "description": "Compare capabilities and pricing of available models",
            "input_schema": {
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of model names to compare"
                    }
                }
            }
        })
        
        return tools
    
    def handle_tool_use(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls from main LLM"""
        logger.info(f"Tool call: {tool_name} with input: {json.dumps(tool_input, indent=2)}")
        
        # Handle query tools
        if tool_name.startswith("query_"):
            llm_name = tool_name[6:]  # Remove "query_" prefix
            logger.info(f"Extracted LLM name from tool: {llm_name}")
            
            llm_type = None
            for lt in LLMType:
                if lt.value == llm_name:
                    llm_type = lt
                    break
            
            if not llm_type:
                logger.error(f"Could not find LLMType for: {llm_name}")
                return {"success": False, "error": f"Unknown LLM: {llm_name}"}
            
            logger.info(f"Mapped to LLMType: {llm_type}")
            
            if llm_type not in self.sub_llms:
                logger.error(f"LLMType {llm_type} not in sub_llms: {list(self.sub_llms.keys())}")
                return {"success": False, "error": f"LLM {llm_name} not available as sub-LLM"}
            
            return self._query_sub_llm(llm_type, tool_input)
        
        # Handle info tools
        elif tool_name.startswith("get_") and tool_name.endswith("_info"):
            llm_name = tool_name[4:-5]  # Remove "get_" and "_info"
            llm_type = None
            for lt in LLMType:
                if lt.value == llm_name:
                    llm_type = lt
                    break
            
            if llm_type:
                model = config.SELECTED_MODELS.get(llm_type, 'unknown')
                info = get_model_info(model)
                return {
                    "success": True,
                    "content": json.dumps(info, indent=2)
                }
        
        # Handle compare models
        elif tool_name == "compare_models":
            models = tool_input.get('models', [])
            if not models:
                models = list(config.SELECTED_MODELS.values())
            comparison = format_model_comparison(models)
            return {
                "success": True,
                "content": comparison
            }
        
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    def _query_sub_llm(self, llm_type: LLMType, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Query a sub-LLM and handle the response"""
        prompt = tool_input.get('prompt', '')
        temperature = tool_input.get('temperature', 0.7)
        max_tokens = tool_input.get('max_tokens', 4096)
        include_context = tool_input.get('include_context', False)
        
        # Debug logging
        logger.info(f"Querying sub-LLM: {llm_type.value}")
        logger.info(f"Available sub-LLMs: {list(self.sub_llms.keys())}")
        
        # Check if the LLM type is actually in our sub_llms
        if llm_type not in self.sub_llms:
            logger.error(f"LLM type {llm_type.value} not found in sub_llms!")
            return {"success": False, "error": f"Sub-LLM {llm_type.value} not available"}
        
        # Get the correct interface
        interface = self.sub_llms[llm_type]
        logger.info(f"Using interface: {type(interface).__name__} for {llm_type.value}")
        
        # Update UI based on type
        if isinstance(self.ui, MultiPaneUI):
            self.ui.update_sub_pane(llm_type, query=prompt, status='processing')
        elif isinstance(self.ui, MultiWindowUI):
            self.ui.add_line(llm_type, f"\n🔧 Receiving query from {self.main_llm.value}:")
            self.ui.add_line(llm_type, f"Prompt: {prompt}")
            self.ui.add_line(llm_type, "Processing...")
        else:
            self.ui.add_message("system", f"Querying {llm_type.value}...", "tool")
        
        self.ui.refresh_display()
        
        # Build messages
        messages = []
        if include_context and len(self.conversation_history) > 0:
            # Include recent context
            context_messages = self.conversation_history[-10:]  # Last 10 messages
            messages.extend(context_messages)
        
        messages.append({"role": "user", "content": prompt})
        
        # Make the API call
        interface = self.sub_llms[llm_type]
        logger.info(f"About to call {llm_type.value} using interface {type(interface).__name__} with model {config.SELECTED_MODELS.get(llm_type, 'unknown')}")
        result = interface.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Log which model was actually used
        if result['success']:
            actual_model = result.get('model', 'unknown')
            logger.info(f"Sub-LLM {llm_type.value} responded using model: {actual_model}")
        
        # Process result
        if result['success']:
            # Extract text content
            text_content = ""
            for content_item in result['content']:
                if isinstance(content_item, dict) and content_item.get('type') == 'text':
                    text_content += content_item.get('text', '')
            
            # Update statistics
            self.stats[llm_type]['message_count'] += 1
            self.stats[llm_type]['tool_calls'] += 1
            if 'usage' in result:
                self.stats[llm_type]['input_tokens'] += result['usage'].get('input_tokens', 0)
                self.stats[llm_type]['output_tokens'] += result['usage'].get('output_tokens', 0)
            
            # Calculate cost
            model = config.SELECTED_MODELS.get(llm_type, '')
            cost = interface.estimate_cost(
                result['usage'].get('input_tokens', 0),
                result['usage'].get('output_tokens', 0),
                model
            )
            self.stats[llm_type]['total_cost'] += cost
            
            # Update UI
            if isinstance(self.ui, MultiPaneUI):
                self.ui.update_sub_pane(
                    llm_type, 
                    response=text_content, 
                    status='complete',
                    stats_update={
                        'message_count': 1,
                        'total_tokens': result['usage'].get('total_tokens', 0),
                        'estimated_cost': cost
                    }
                )
            elif isinstance(self.ui, MultiWindowUI):
                self.ui.add_line(llm_type, f"\n✅ Response from {llm_type.value}:")
                self.ui.add_line(llm_type, text_content)
                self.ui.add_line(llm_type, f"\nTokens: {result['usage'].get('total_tokens', 0)}, Cost: ${cost:.4f}")
                self.ui.update_stats(llm_type,
                    message_count=1,
                    total_tokens=result['usage'].get('total_tokens', 0),
                    estimated_cost=cost
                )
            
            logger.info(f"{llm_type.value} response - Tokens: {result['usage'].get('total_tokens', 0)}, Cost: ${cost:.4f}")
            return {"success": True, "content": text_content}
            
        else:
            # Handle error
            self.stats[llm_type]['errors'] += 1
            error_msg = result.get('error', 'Unknown error')
            
            if isinstance(self.ui, MultiPaneUI):
                self.ui.update_sub_pane(llm_type, response=f"Error: {error_msg}", status='error')
            elif isinstance(self.ui, MultiWindowUI):
                self.ui.add_line(llm_type, f"\n❌ Error: {error_msg}")
            
            logger.error(f"{llm_type.value} error: {error_msg}")
            return result
    
    def chat(self, user_message: str, maintain_history: bool = True) -> str:
        """Send a message to the main LLM with access to sub-LLMs"""
        # Build system prompt
        system_prompt = self._build_system_prompt()
        
        # Add to history
        if maintain_history:
            self.conversation_history.append({"role": "user", "content": user_message})
            messages = self.conversation_history.copy()
        else:
            messages = [{"role": "user", "content": user_message}]
        
        # Handle system prompt based on LLM type
        if self.main_llm != LLMType.CLAUDE:
            has_system = any(msg['role'] == 'system' for msg in messages)
            if not has_system and system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Show thinking indicator
        if isinstance(self.ui, (MultiWindowUI, MultiPaneUI)):
            self.ui.add_message(self.main_llm.value, f"[{self.main_llm.value} thinking...]", "thinking")
        
        self.ui.refresh_display()
        
        # Main conversation loop
        max_iterations = 10
        iteration = 0
        final_response = ""
        tool_use_count = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                logger.info(f"Chat iteration {iteration} - Messages: {len(messages)}")
                
                # Get response from main LLM
                result = self.main_interface.chat(
                    messages=messages,
                    system_prompt=system_prompt if self.main_llm == LLMType.CLAUDE else None,
                    tools=self.tools if self.sub_llms and self.main_interface.supports_tools() else None
                )
                
                if not result['success']:
                    self.stats[self.main_llm]['errors'] += 1
                    error_msg = f"Error: {result['error']}"
                    self.ui.add_message(self.main_llm.value, error_msg, "error")
                    return error_msg
                
                # Update statistics
                if 'usage' in result:
                    self.stats[self.main_llm]['input_tokens'] += result['usage'].get('input_tokens', 0)
                    self.stats[self.main_llm]['output_tokens'] += result['usage'].get('output_tokens', 0)
                    self.stats[self.main_llm]['message_count'] += 1
                    
                    model = config.SELECTED_MODELS.get(self.main_llm, '')
                    cost = self.main_interface.estimate_cost(
                        result['usage'].get('input_tokens', 0),
                        result['usage'].get('output_tokens', 0),
                        model
                    )
                    self.stats[self.main_llm]['total_cost'] += cost
                    
                    # Update UI stats
                    if isinstance(self.ui, (MultiWindowUI, MultiPaneUI)):
                        if isinstance(self.ui, MultiWindowUI):
                            self.ui.update_stats(self.main_llm,
                                message_count=1,
                                total_tokens=result['usage'].get('total_tokens', 0),
                                estimated_cost=cost
                            )
                        else:
                            self.ui.update_main_stats(
                                message_count=1,
                                total_tokens=result['usage'].get('total_tokens', 0),
                                estimated_cost=cost
                            )
                
                # Extract text content
                text_content = ""
                for content_item in result['content']:
                    if isinstance(content_item, dict) and content_item.get('type') == 'text':
                        text_content += content_item.get('text', '')
                
                # Check for tool calls
                has_tool_calls = bool(result.get('tool_calls'))
                
                # If there's text content, add it to response
                if text_content:
                    if iteration == 1:
                        final_response = text_content
                    else:
                        final_response += "\n" + text_content
                
                # If no tool calls, we're done
                if not has_tool_calls:
                    break
                
                # Handle tool calls
                tool_use_count += len(result['tool_calls'])
                self.stats[self.main_llm]['tool_calls'] += len(result['tool_calls'])
                
                # For Claude, we need to add the assistant message before processing tools
                if self.main_llm == LLMType.CLAUDE and result.get('tool_calls'):
                    if maintain_history:
                        # Add the complete assistant response including tool calls
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": result['raw_response'].content
                        })
                        messages = self.conversation_history.copy()
                    else:
                        # Add to temporary messages
                        messages.append({
                            "role": "assistant",
                            "content": result['raw_response'].content
                        })
                
                for tool_call in result['tool_calls']:
                    logger.info(f"Executing tool: {tool_call['name']}")
                    
                    # Show tool usage in UI
                    self.ui.add_message(
                        self.main_llm.value,
                        f"Using tool: {tool_call['name']}",
                        "tool"
                    )
                    self.ui.refresh_display()
                    
                    # Execute tool
                    tool_result = self.handle_tool_use(
                        tool_call['name'],
                        tool_call.get('input', {})
                    )
                    
                    # Handle tool result based on LLM type
                    if self.main_llm == LLMType.CLAUDE:
                        # Claude needs the tool result message added to the conversation
                        tool_result_message = {
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_call['id'],
                                "content": json.dumps(tool_result) if not isinstance(tool_result.get('content', tool_result), str) else tool_result.get('content', str(tool_result))
                            }]
                        }
                        
                        # Add tool result to messages
                        messages.append(tool_result_message)
                        if maintain_history:
                            self.conversation_history.append(tool_result_message)
                        
                        # Get Claude's response after tool use
                        follow_up = self.main_interface.chat(
                            messages=messages,
                            system_prompt=system_prompt,
                            tools=self.tools
                        )
                        
                        if follow_up['success']:
                            # Extract text from follow-up
                            follow_up_text = ""
                            for content_item in follow_up['content']:
                                if isinstance(content_item, dict) and content_item.get('type') == 'text':
                                    follow_up_text += content_item.get('text', '')
                            
                            if follow_up_text:
                                final_response += "\n" + follow_up_text
                            
                            # Update stats for follow-up
                            if 'usage' in follow_up:
                                self._update_main_stats(follow_up['usage'])
                            
                            # Update conversation history with the follow-up response
                            if maintain_history and follow_up.get('content'):
                                # Only add text responses to avoid duplicating tool calls
                                has_text = any(c.get('type') == 'text' and c.get('text') 
                                             for c in follow_up['content'])
                                if has_text and 'raw_response' in follow_up:
                                    self.conversation_history.append({
                                        "role": "assistant", 
                                        "content": follow_up['raw_response'].content
                                    })
                                    messages = self.conversation_history.copy()
                            
                            # Check if follow-up has more tool calls
                            if not follow_up.get('tool_calls'):
                                break
                            else:
                                result = follow_up  # Continue with new tool calls
                    
                    elif self.main_llm == LLMType.OPENAI:
                        # Add the assistant message with tool calls
                        messages.append(result['raw_response'].choices[0].message.model_dump())
                        
                        # Add tool result
                        formatted_result = self.main_interface.format_tool_response(
                            tool_call['id'],
                            tool_result.get('content', str(tool_result))
                        )
                        messages.append(formatted_result)
                    
                    elif self.main_llm == LLMType.GEMINI:
                        # Gemini handles tools differently
                        messages.append({
                            "role": "assistant",
                            "content": text_content if text_content else "Processing tool results..."
                        })
                        messages.append({
                            "role": "user",
                            "content": f"Tool {tool_call['name']} returned: {tool_result.get('content', str(tool_result))}"
                        })
                
                # For OpenAI/Gemini, continue the loop to get final response
                if self.main_llm in [LLMType.OPENAI, LLMType.GEMINI] and has_tool_calls:
                    continue
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
                self.stats[self.main_llm]['errors'] += 1
                error_msg = f"Error: {str(e)}"
                self.ui.add_message(self.main_llm.value, error_msg, "error")
                return error_msg
        
        # Add final response to history if not already added
        if maintain_history and final_response:
            # For Claude with tool use, check if response was already added
            last_message = self.conversation_history[-1] if self.conversation_history else None
            
            # Only add if the last message isn't already an assistant message with our content
            if not (last_message and last_message.get('role') == 'assistant'):
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })
        
        # Log summary
        logger.info(f"Chat complete - Iterations: {iteration}, Tool calls: {tool_use_count}, Response length: {len(final_response)}")
        
        # Auto-save if enabled
        if config.AUTO_SAVE and (datetime.now() - self.last_save_time).seconds > config.SAVE_INTERVAL:
            self.auto_save()
        
        return final_response
    
    def _update_main_stats(self, usage: Dict[str, Any]):
        """Update main LLM statistics"""
        self.stats[self.main_llm]['input_tokens'] += usage.get('input_tokens', 0)
        self.stats[self.main_llm]['output_tokens'] += usage.get('output_tokens', 0)
        
        model = config.SELECTED_MODELS.get(self.main_llm, '')
        cost = self.main_interface.estimate_cost(
            usage.get('input_tokens', 0),
            usage.get('output_tokens', 0),
            model
        )
        self.stats[self.main_llm]['total_cost'] += cost
        
        # Update UI
        if isinstance(self.ui, MultiWindowUI):
            self.ui.update_stats(self.main_llm,
                input_tokens=usage.get('input_tokens', 0),
                output_tokens=usage.get('output_tokens', 0),
                total_tokens=usage.get('total_tokens', 0),
                estimated_cost=cost
            )
        elif isinstance(self.ui, MultiPaneUI):
            self.ui.update_main_stats(
                input_tokens=usage.get('input_tokens', 0),
                output_tokens=usage.get('output_tokens', 0),
                total_tokens=usage.get('total_tokens', 0),
                estimated_cost=cost
            )
    
    def _build_system_prompt(self) -> str:
        """Build system prompt based on available sub-LLMs"""
        if not self.sub_llms:
            return (f"You are {self.main_llm.value.upper()}, an AI assistant. "
                   f"Be helpful, accurate, and concise. Model: {config.SELECTED_MODELS.get(self.main_llm, 'unknown')}")
        
        # Get sub-LLM details
        sub_llm_info = []
        for llm_type, interface in self.sub_llms.items():
            model = config.SELECTED_MODELS.get(llm_type, 'unknown')
            model_info = get_model_info(model)
            supports_tools = interface.supports_tools()
            
            info = f"- {llm_type.value.upper()} ({model}): {model_info['description']}"
            if supports_tools:
                info += " [Supports tool calling]"
            sub_llm_info.append(info)
        
        tools_desc = "\n".join(sub_llm_info)
        
        return f"""You are {self.main_llm.value.upper()}, a sophisticated AI agent using model {config.SELECTED_MODELS.get(self.main_llm, 'unknown')}.

You have access to other AI models as tools:
{tools_desc}

You can query these models to:
- Get different perspectives on complex topics
- Verify information or cross-check answers
- Delegate specialized tasks based on model strengths
- Compare approaches to problem-solving

Guidelines:
1. Use sub-models when their capabilities would enhance your response
2. Consider model costs when deciding which to use
3. Synthesize information from multiple sources when appropriate
4. Be transparent about when you're consulting other models
5. Focus on providing the best possible answer to the user

Current date: {datetime.now().strftime('%Y-%m-%d')}"""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        # Calculate totals
        total_tokens = 0
        total_cost = 0.0
        total_messages = 0
        total_tool_calls = 0
        total_errors = 0
        
        llm_stats = {}
        for llm_type, stats in self.stats.items():
            total_tokens += stats['input_tokens'] + stats['output_tokens']
            total_cost += stats['total_cost']
            total_messages += stats['message_count']
            total_tool_calls += stats['tool_calls']
            total_errors += stats['errors']
            
            llm_stats[llm_type.value] = {
                'model': config.SELECTED_MODELS.get(llm_type, 'unknown'),
                'message_count': stats['message_count'],
                'total_tokens': stats['input_tokens'] + stats['output_tokens'],
                'input_tokens': stats['input_tokens'],
                'output_tokens': stats['output_tokens'],
                'total_cost': stats['total_cost'],
                'tool_calls': stats['tool_calls'],
                'errors': stats['errors'],
                'avg_tokens_per_message': (stats['input_tokens'] + stats['output_tokens']) / max(1, stats['message_count'])
            }
        
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        
        return {
            'session_id': self.session_id,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'total_messages': total_messages,
            'total_tool_calls': total_tool_calls,
            'total_errors': total_errors,
            'session_duration_minutes': round(session_duration, 2),
            'conversation_length': len(self.conversation_history),
            'messages_per_minute': round(total_messages / max(1, session_duration), 2),
            'cost_per_message': round(total_cost / max(1, total_messages), 4),
            'llm_stats': llm_stats,
            'main_llm': self.main_llm.value,
            'sub_llms': [llm.value for llm in self.sub_llms.keys()]
        }
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        
        if isinstance(self.ui, (MultiWindowUI, MultiPaneUI)):
            # Show in UI
            self.ui.show_statistics(stats)
        else:
            # Standard output
            print(f"\n{Colors.BOLD}{Colors.YELLOW}📊 Session Statistics:{Colors.RESET}")
            print("=" * 80)
            print(f"Session ID: {stats['session_id']}")
            print(f"Duration: {stats['session_duration_minutes']:.1f} minutes")
            print(f"Total messages: {stats['total_messages']} ({stats['messages_per_minute']:.1f}/min)")
            print(f"Total tokens: {stats['total_tokens']:,}")
            print(f"Total cost: ${stats['total_cost']:.4f} (${stats['cost_per_message']:.4f}/msg)")
            print(f"Tool calls: {stats['total_tool_calls']}")
            if stats['total_errors'] > 0:
                print(f"{Colors.RED}Errors: {stats['total_errors']}{Colors.RESET}")
            
            print(f"\n{Colors.BOLD}{Colors.YELLOW}📈 Per-Model Breakdown:{Colors.RESET}")
            print("-" * 80)
            print(f"{'Model':<40} {'Messages':<10} {'Tokens':<15} {'Cost':<10} {'Tools':<8}")
            print("-" * 80)
            
            for llm_name, llm_stats in stats['llm_stats'].items():
                if llm_stats['message_count'] > 0:
                    model_display = llm_stats['model']
                    if len(model_display) > 38:
                        model_display = model_display[:35] + "..."
                    
                    is_main = llm_name == stats['main_llm']
                    prefix = "🎯 " if is_main else "   "
                    
                    print(f"{prefix}{model_display:<38} "
                          f"{llm_stats['message_count']:<10} "
                          f"{llm_stats['total_tokens']:<15,} "
                          f"${llm_stats['total_cost']:<9.4f} "
                          f"{llm_stats['tool_calls']:<8}")
            
            print("=" * 80)
    
    def clear_conversation(self):
        """Clear conversation history and reset statistics"""
        self.conversation_history = []
        for llm in self.stats:
            self.stats[llm] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'message_count': 0,
                'total_cost': 0.0,
                'tool_calls': 0,
                'errors': 0
            }
        self.session_start = datetime.now()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clear UI
        if isinstance(self.ui, (MultiWindowUI, MultiPaneUI)):
            self.ui.clear_all()
        
        logger.info("Conversation history and statistics cleared")
    
    def save_conversation(self, filename: str = None) -> bool:
        """Save conversation history and metadata"""
        if not filename:
            filename = f"conversation_{self.session_id}.json"
        
        try:
            save_data = {
                'version': VERSION,
                'session_id': self.session_id,
                'conversation': self.conversation_history,
                'main_llm': self.main_llm.value,
                'sub_llms': [llm.value for llm in self.sub_llms.keys()],
                'models': {llm.value: model for llm, model in config.SELECTED_MODELS.items()},
                'statistics': self.get_statistics(),
                'config': {
                    'verbose_display': config.VERBOSE_DISPLAY,
                    'show_timestamps': config.SHOW_TIMESTAMPS,
                    'display_mode': config.DISPLAY_MODE
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            save_path = Path(filename)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with pretty formatting
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Conversation saved to {save_path}")
            self.last_save_time = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}", exc_info=True)
            return False
    
    def load_conversation(self, filename: str) -> bool:
        """Load conversation history"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            # Load conversation
            self.conversation_history = save_data.get('conversation', [])
            
            # Reset statistics (can't restore token counts accurately)
            self.clear_conversation()
            
            # Update session info
            self.session_id = save_data.get('session_id', self.session_id)
            
            logger.info(f"Loaded {len(self.conversation_history)} messages from {filename}")
            
            # Display metadata
            info_lines = [
                f"📅 Saved: {save_data.get('timestamp', 'Unknown')}",
                f"🤖 Main: {save_data.get('main_llm', 'Unknown')}",
                f"🔧 Subs: {', '.join(save_data.get('sub_llms', []))}",
                f"📊 Messages: {len(self.conversation_history)}"
            ]
            
            for line in info_lines:
                self.ui.add_message("system", line, "system")
            
            return True
            
        except FileNotFoundError:
            logger.warning(f"File not found: {filename}")
            self.ui.add_message("system", f"File not found: {filename}", "error")
            return False
        except Exception as e:
            logger.error(f"Error loading conversation: {str(e)}", exc_info=True)
            self.ui.add_message("system", f"Error loading: {str(e)}", "error")
            return False
    
    def auto_save(self):
        """Auto-save conversation"""
        filename = f"autosave_{self.session_id}.json"
        if self.save_conversation(filename):
            logger.info(f"Auto-saved to {filename}")


def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    packages = [
        ("anthropic", "anthropic"),
        ("openai", "openai"),
        ("google.generativeai", "google-generativeai"),
        ("dotenv", "python-dotenv")
    ]
    
    for import_name, pip_name in packages:
        try:
            if import_name == "google.generativeai":
                import google.generativeai
            else:
                __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    
    if missing:
        print(f"\n⚠️  Missing required packages:")
        print(f"   pip install {' '.join(missing)}")
        print("\nOr install all dependencies:")
        print("   pip install anthropic openai google-generativeai python-dotenv")
        return False
    
    return True


def load_api_keys() -> Dict[str, Optional[str]]:
    """Load API keys from environment or .env file"""
    # Check environment variables first
    keys = {
        'anthropic': os.getenv("ANTHROPIC_API_KEY"),
        'openai': os.getenv("OPENAI_API_KEY"),
        'gemini': os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    }
    
    # Check if all keys are present
    if all(keys.values()):
        logger.info("✅ Found all API keys in environment variables")
        return keys
    
    # Try loading from .env file
    try:
        from dotenv import load_dotenv
        
        # Look for .env in current directory and parent directories
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents)[:3]:
            dotenv_path = parent / '.env'
            if dotenv_path.is_file():
                if load_dotenv(dotenv_path=dotenv_path):
                    logger.info(f"📄 Loading API keys from: {dotenv_path}")
                    # Reload keys
                    keys = {
                        'anthropic': os.getenv("ANTHROPIC_API_KEY"),
                        'openai': os.getenv("OPENAI_API_KEY"),
                        'gemini': os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                    }
                    break
    except ImportError:
        logger.warning("python-dotenv not installed")
    
    return keys


def setup_env_file():
    """Interactive setup for .env file"""
    env_path = Path(".env")
    
    if env_path.exists():
        overwrite = input("A .env file already exists. Overwrite it? (y/n): ").strip().lower()
        if overwrite != 'y':
            return False
    
    print("\n🔧 Setting up .env file for API keys...")
    print("=" * 60)
    print("You can get API keys from:")
    print("- Anthropic: https://console.anthropic.com/")
    print("- OpenAI: https://platform.openai.com/api-keys")
    print("- Google: https://makersuite.google.com/app/apikey")
    print()
    
    anthropic_key = input("Enter your Anthropic API key (or press Enter to skip): ").strip()
    openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    gemini_key = input("Enter your Google/Gemini API key (or press Enter to skip): ").strip()
    
    env_content = f"""# API Keys for Multi-LLM Bridge
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANTHROPIC_API_KEY={anthropic_key or 'your-anthropic-api-key-here'}
OPENAI_API_KEY={openai_key or 'your-openai-api-key-here'}
GEMINI_API_KEY={gemini_key or 'your-gemini-api-key-here'}

# Optional settings
# LOG_LEVEL=INFO
# AUTO_SAVE=true
# SAVE_INTERVAL=300
"""
    
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print(f"\n✅ Created .env file at: {env_path.absolute()}")
    
    # Create .gitignore if it doesn't exist
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, "w") as f:
            f.write(""".env
.env.*
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
conversation_*.json
autosave_*.json
logs/
*.log
.venv/
venv/
ENV/
.DS_Store
Thumbs.db
*.swp
*.swo
*~
.idea/
.vscode/
*.sublime-*
""")
        print("✅ Created .gitignore to protect your secrets")
    
    return bool(anthropic_key or openai_key or gemini_key)


def select_model_interactive(provider: str, available_models: List[str]) -> str:
    """Interactive model selection with grouping and pricing info"""
    print(f"\n🤖 Select {provider.upper()} Model:")
    print("=" * 80)
    
    # Group models by family
    model_groups = {
        # Claude groups
        "Claude Opus 4": [],
        "Claude Sonnet 4": [],
        "Claude 3.5 Series": [],
        "Claude 3 Series": [],
        
        # OpenAI groups
        "GPT-4o Series": [],
        "GPT-4 Turbo": [],
        "GPT-4 Classic": [],
        "O1 Reasoning": [],
        "GPT-3.5 Series": [],
        
        # Gemini groups
        "Gemini 2.0": [],
        "Gemini 1.5": [],
        "Gemini 1.0": [],
        
        # Other
        "Other Models": []
    }
    
    # Categorize models
    for model in available_models:
        categorized = False
        
        # Claude categorization
        if 'claude-opus-4' in model:
            model_groups["Claude Opus 4"].append(model)
            categorized = True
        elif 'claude-sonnet-4' in model:
            model_groups["Claude Sonnet 4"].append(model)
            categorized = True
        elif 'claude-3-5' in model or 'claude-3.5' in model:
            model_groups["Claude 3.5 Series"].append(model)
            categorized = True
        elif 'claude-3' in model:
            model_groups["Claude 3 Series"].append(model)
            categorized = True
        
        # OpenAI categorization
        elif 'gpt-4o' in model:
            model_groups["GPT-4o Series"].append(model)
            categorized = True
        elif 'gpt-4' in model and 'turbo' in model:
            model_groups["GPT-4 Turbo"].append(model)
            categorized = True
        elif 'gpt-4' in model:
            model_groups["GPT-4 Classic"].append(model)
            categorized = True
        elif 'o1' in model:
            model_groups["O1 Reasoning"].append(model)
            categorized = True
        elif 'gpt-3.5' in model:
            model_groups["GPT-3.5 Series"].append(model)
            categorized = True
        
        # Gemini categorization
        elif 'gemini-2' in model:
            model_groups["Gemini 2.0"].append(model)
            categorized = True
        elif 'gemini-1.5' in model:
            model_groups["Gemini 1.5"].append(model)
            categorized = True
        elif 'gemini-1' in model or 'gemini-pro' in model:
            model_groups["Gemini 1.0"].append(model)
            categorized = True
        
        if not categorized:
            model_groups["Other Models"].append(model)
    
    # Display models - FIXED: Check if we should display this group for the current provider
    numbered_models = []
    for group_name, models in model_groups.items():
        # Check if this group is relevant to the current provider
        should_display = False
        
        if provider.lower() == 'claude' and 'Claude' in group_name:
            should_display = True
        elif provider.lower() == 'openai' and any(x in group_name for x in ['GPT', 'O1']):
            should_display = True
        elif provider.lower() == 'gemini' and 'Gemini' in group_name:
            should_display = True
        elif group_name == "Other Models" and models:
            # Always show "Other Models" if there are any
            should_display = True
        
        if models and should_display:
            print(f"\n{Colors.BOLD}{Colors.CYAN}{group_name}:{Colors.RESET}")
            
            for model in sorted(models, reverse=True):
                numbered_models.append(model)
                num = len(numbered_models)
                
                # Get pricing info
                pricing_str = ""
                if model in PRICE_MAPPING:
                    pricing = PRICE_MAPPING[model]
                    pricing_str = f" {Colors.DIM}[${pricing['input']}/1M in, ${pricing['output']}/1M out]{Colors.RESET}"
                    if pricing['input'] == 0 and pricing['output'] == 0:
                        pricing_str = f" {Colors.GREEN}[FREE]{Colors.RESET}"
                
                # Get model info
                info = get_model_info(model)
                desc = f" - {info['description']}" if info['description'] else ""
                
                print(f"  {num}. {model}{pricing_str}{desc}")
    
    # Add custom option
    print(f"\n{len(numbered_models) + 1}. Enter custom model name")
    print("=" * 80)
    
    # Get selection
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(numbered_models) + 1}): ").strip()
            
            if choice == str(len(numbered_models) + 1):
                custom_model = input("Enter custom model name: ").strip()
                if custom_model:
                    print(f"✅ Selected: {custom_model}")
                    return custom_model
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(numbered_models):
                    selected = numbered_models[idx]
                    print(f"✅ Selected: {selected}")
                    return selected
        except (ValueError, IndexError):
            pass
        
        print("❌ Invalid choice. Please try again.")


def select_main_llm(available_keys: Dict[str, bool]) -> Optional[LLMType]:
    """Select which LLM should be the main one"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}🎯 Select Main LLM:{Colors.RESET}")
    print("=" * 60)
    print("The main LLM will handle your conversations and can call other LLMs as tools.")
    print()
    
    options = []
    for llm_type in LLMType:
        if available_keys.get(llm_type.value, False):
            options.append(llm_type)
            status = f"{Colors.GREEN}✅ Available{Colors.RESET}"
        else:
            status = f"{Colors.RED}❌ No API key{Colors.RESET}"
        
        print(f"{len(options) if llm_type in options else '-'}. {llm_type.value.upper()} {status}")
    
    if not options:
        print(f"\n{Colors.RED}No LLMs available! Please set up API keys.{Colors.RESET}")
        return None
    
    while True:
        try:
            choice = input(f"\nSelect main LLM (1-{len(options)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                selected = options[idx]
                print(f"✅ Selected {selected.value.upper()} as main LLM")
                return selected
        except ValueError:
            pass
        print("❌ Invalid choice. Please try again.")


def select_sub_llms(available_keys: Dict[str, bool], main_llm: LLMType) -> List[LLMType]:
    """Select which LLMs can be called as tools"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}🔧 Select Sub-LLMs (tools):{Colors.RESET}")
    print("=" * 60)
    print("Select which LLMs the main AI can call as tools.")
    print("You can enable all, some, or none.")
    print()
    
    # Get available sub-LLMs (excluding main)
    options = []
    for llm_type in LLMType:
        if llm_type != main_llm and available_keys.get(llm_type.value, False):
            options.append(llm_type)
            print(f"{len(options)}. {llm_type.value.upper()}")
    
    if not options:
        print("⚠️  No other LLMs available as tools.")
        return []
    
    print(f"\n{len(options) + 1}. All available LLMs")
    print(f"{len(options) + 2}. None (main LLM only)")
    
    while True:
        try:
            choice = input(f"\nSelect option (1-{len(options) + 2}) or comma-separated numbers: ").strip()
            
            if choice == str(len(options) + 1):
                # All
                print(f"✅ Selected all available LLMs as tools")
                return options
            elif choice == str(len(options) + 2):
                # None
                print("✅ Main LLM will work standalone (no tools)")
                return []
            elif ',' in choice:
                # Multiple selections
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected = [options[i] for i in indices if 0 <= i < len(options)]
                if selected:
                    print(f"✅ Selected {', '.join(llm.value.upper() for llm in selected)} as tools")
                    return selected
            else:
                # Single selection
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    selected = [options[idx]]
                    print(f"✅ Selected {selected[0].value.upper()} as tool")
                    return selected
        except (ValueError, IndexError):
            pass
        print("❌ Invalid choice. Please try again.")


def select_display_mode(num_llms: int) -> str:
    """Select display mode based on available options"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}🖥️  Select Display Mode:{Colors.RESET}")
    print("=" * 60)
    
    modes = [
        ("standard", "Standard", "Simple, reliable display for all terminals"),
        ("multi-window", "Multi-Window", f"Separate windows for each LLM ({num_llms} windows)"),
        ("multi-pane", "Multi-Pane", "Advanced layout with sub-agent visualization")
    ]
    
    # Check terminal capabilities
    capabilities = check_terminal_capabilities()
    term_width, term_height = shutil.get_terminal_size((80, 24))
    
    print(f"Terminal: {term_width}x{term_height}")
    if not capabilities['unicode']:
        print("⚠️  Limited Unicode support detected")
    
    for i, (mode_id, name, desc) in enumerate(modes, 1):
        # Add warnings for advanced modes
        warning = ""
        if mode_id == "multi-window" and term_height < 40:
            warning = f" {Colors.YELLOW}(Terminal may be too small){Colors.RESET}"
        elif mode_id == "multi-pane" and (term_width < 120 or term_height < 40):
            warning = f" {Colors.YELLOW}(Requires large terminal){Colors.RESET}"
        
        print(f"{i}. {name} - {desc}{warning}")
    
    print("\n💡 Tip: You can resize your terminal and restart for better experience")
    
    while True:
        choice = input(f"\nSelect display mode (1-{len(modes)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(modes):
                selected = modes[idx][0]
                print(f"✅ Selected {modes[idx][1]} mode")
                return selected
        except ValueError:
            pass
        print("❌ Invalid choice. Please try again.")


def print_commands_help():
    """Print available commands"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}📚 Available Commands:{Colors.RESET}")
    print("=" * 80)
    
    commands = [
        ("Basic Commands", [
            ("exit/quit/q", "Exit the conversation"),
            ("help", "Show this help message"),
            ("clear", "Clear conversation history"),
            ("history [n]", "Show conversation history (last n messages)")
        ]),
        ("Statistics & Info", [
            ("stats", "Show detailed statistics"),
            ("models", "Show current model information"),
            ("cost", "Show cost breakdown")
        ]),
        ("Save & Load", [
            ("save [filename]", "Save conversation to file"),
            ("load [filename]", "Load previous conversation"),
            ("autosave on/off", "Toggle auto-save"),
            ("list", "List saved conversations")
        ]),
        ("Display Options", [
            ("verbose on/off", "Toggle full/truncated display"),
            ("timestamps on/off", "Toggle message timestamps"),
            ("resize", "Force refresh display (if it looks wrong)")
        ]),
        ("Multi-Window/Pane Commands", [
            ("scroll up/down [n]", "Scroll main window"),
            ("scroll [llm] up/down [n]", "Scroll specific window (multi-window)"),
            ("toggle-panes", "Show/hide sub-agent panes (multi-pane)"),
            ("switch [llm]", "Switch active window (multi-window)")
        ])
    ]
    
    for category, cmds in commands:
        print(f"\n{Colors.CYAN}{category}:{Colors.RESET}")
        for cmd, desc in cmds:
            print(f"  {Colors.GREEN}{cmd:<25}{Colors.RESET} {desc}")
    
    print("\n💡 Tips:")
    print("  - Use TAB for command completion (if supported)")
    print("  - Commands are case-insensitive")
    print("  - Most commands have short aliases")
    print("=" * 80)


def handle_command(command: str, bridge: MultiLLMBridge, ui: BaseTerminalUI) -> bool:
    """Handle user commands. Returns True if should continue, False if should exit."""
    cmd_parts = command.lower().split()
    if not cmd_parts:
        return True
    
    cmd = cmd_parts[0]
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
    
    # Exit commands
    if cmd in ['exit', 'quit', 'q']:
        ui.add_message("system", "Exiting...", "system")
        return False
    
    # Help
    elif cmd in ['help', 'h', '?']:
        print_commands_help()
        input("\nPress Enter to continue...")
        ui.clear_screen()
        ui.refresh_display()
    
    # Clear conversation
    elif cmd == 'clear':
        confirm = input("Clear conversation history? (y/n): ").lower()
        if confirm == 'y':
            bridge.clear_conversation()
            ui.add_message("system", "🗑️ Conversation cleared!", "system")
    
    # Show history
    elif cmd == 'history':
        count = int(args[0]) if args and args[0].isdigit() else 10
        if not bridge.conversation_history:
            ui.add_message("system", "No conversation history yet.", "system")
        else:
            ui.add_message("system", f"📜 Last {count} messages:", "system")
            for i, msg in enumerate(bridge.conversation_history[-count:]):
                role = msg['role'].upper()
                content = msg['content']
                if isinstance(content, str):
                    preview = content[:100] + '...' if len(content) > 100 else content
                    ui.add_message("system", f"{i+1}. {role}: {preview}", "system")
    
    # Statistics
    elif cmd in ['stats', 'statistics']:
        bridge.print_statistics()
    
    # Model info
    elif cmd in ['models', 'model']:
        ui.add_message("system", "🤖 Current Models:", "system")
        for llm_type, model in config.SELECTED_MODELS.items():
            info = get_model_info(model)
            ui.add_message("system", f"{llm_type.value}: {model}", "system")
            ui.add_message("system", f"  Context: {info['context_window']}, {info['description']}", "system")
    
    # Cost breakdown
    elif cmd == 'cost':
        stats = bridge.get_statistics()
        ui.add_message("system", "💰 Cost Breakdown:", "system")
        total = stats['total_cost']
        for llm_name, llm_stats in stats['llm_stats'].items():
            if llm_stats['total_cost'] > 0:
                pct = (llm_stats['total_cost'] / total * 100) if total > 0 else 0
                ui.add_message("system", 
                    f"{llm_name}: ${llm_stats['total_cost']:.4f} ({pct:.1f}%)", 
                    "system")
    
    # Save conversation
    elif cmd == 'save':
        filename = args[0] if args else None
        if bridge.save_conversation(filename):
            ui.add_message("system", f"💾 Conversation saved!", "system")
        else:
            ui.add_message("system", "❌ Failed to save conversation", "error")
    
    # Load conversation
    elif cmd == 'load':
        filename = args[0] if args else input("Enter filename: ").strip()
        if bridge.load_conversation(filename):
            ui.add_message("system", f"📂 Conversation loaded!", "system")
        else:
            ui.add_message("system", "❌ Failed to load conversation", "error")
    
    # List saved conversations
    elif cmd == 'list':
        pattern = "conversation_*.json"
        files = list(Path(".").glob(pattern)) + list(Path(".").glob("autosave_*.json"))
        if files:
            ui.add_message("system", "📁 Saved conversations:", "system")
            for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                size = f.stat().st_size / 1024
                mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                ui.add_message("system", f"  {f.name} ({size:.1f}KB, {mtime})", "system")
        else:
            ui.add_message("system", "No saved conversations found.", "system")
    
    # Auto-save toggle
    elif cmd == 'autosave':
        if args and args[0] in ['on', 'off']:
            config.AUTO_SAVE = args[0] == 'on'
            ui.add_message("system", f"Auto-save: {'ON' if config.AUTO_SAVE else 'OFF'}", "system")
        else:
            ui.add_message("system", f"Auto-save is {'ON' if config.AUTO_SAVE else 'OFF'}", "system")
    
    # Verbose toggle
    elif cmd == 'verbose':
        if args and args[0] in ['on', 'off']:
            config.VERBOSE_DISPLAY = args[0] == 'on'
        else:
            config.VERBOSE_DISPLAY = not config.VERBOSE_DISPLAY
        ui.add_message("system", f"Verbose display: {'ON' if config.VERBOSE_DISPLAY else 'OFF'}", "system")
    
    # Timestamps toggle
    elif cmd == 'timestamps':
        if args and args[0] in ['on', 'off']:
            config.SHOW_TIMESTAMPS = args[0] == 'on'
        else:
            config.SHOW_TIMESTAMPS = not config.SHOW_TIMESTAMPS
        ui.add_message("system", f"Timestamps: {'ON' if config.SHOW_TIMESTAMPS else 'OFF'}", "system")
    
    # Display refresh
    elif cmd == 'resize':
        ui.clear_screen()
        ui.refresh_display()
        ui.add_message("system", "🔄 Display refreshed", "system")
    
    # Scrolling
    elif cmd == 'scroll':
        if isinstance(ui, (MultiWindowUI, MultiPaneUI)):
            if len(args) >= 2:
                direction = args[0]
                amount = int(args[1]) if len(args) > 1 and args[1].isdigit() else 5
                
                if isinstance(ui, MultiWindowUI):
                    # Check if scrolling specific window
                    if direction in ['claude', 'openai', 'gemini']:
                        llm_type = LLMType(direction)
                        if len(args) > 1:
                            direction = args[1]
                            amount = int(args[2]) if len(args) > 2 and args[2].isdigit() else 5
                        ui.scroll(llm_type, direction, amount)
                    else:
                        ui.scroll(ui.main_llm, direction, amount)
                else:
                    ui.scroll(direction, amount)
                
                ui.refresh_display()
    
    # Toggle panes (multi-pane mode)
    elif cmd == 'toggle-panes' and isinstance(ui, MultiPaneUI):
        ui.toggle_sub_panes()
        ui.add_message("system", f"Sub-agent panes: {'shown' if ui.show_sub_panes else 'hidden'}", "system")
        ui.refresh_display()
    
    # Switch active window (multi-window mode)
    elif cmd == 'switch' and isinstance(ui, MultiWindowUI):
        if args and args[0] in ['claude', 'openai', 'gemini']:
            llm_type = LLMType(args[0])
            ui.set_active_window(llm_type)
            ui.refresh_display()
    
    # Unknown command
    else:
        ui.add_message("system", f"Unknown command: {cmd}. Type 'help' for commands.", "error")
    
    return True


def interactive_chat_loop(bridge: MultiLLMBridge, ui: BaseTerminalUI):
    """Main interactive chat loop"""
    print(f"\n{Colors.BOLD}{Colors.GREEN}💬 Chat Started!{Colors.RESET}")
    print(f"Type 'help' for commands, 'exit' to quit")
    print("=" * 80)
    
    ui.clear_screen()
    ui.refresh_display()
    
    while True:
        try:
            # Get user input
            if isinstance(ui, StandardUI):
                user_input = input(f"\n{Colors.GREEN}You: {Colors.RESET}").strip()
            else:
                # For advanced UIs, input is handled by the UI
                ui.refresh_display()
                user_input = input().strip()
            
            if not user_input:
                continue
            
            # Check if it's a command
            if user_input.startswith('/') or user_input.lower() in [
                'exit', 'quit', 'q', 'help', 'clear', 'stats', 'save', 'load',
                'history', 'verbose', 'timestamps', 'resize', 'models', 'cost'
            ]:
                # Remove leading slash if present
                if user_input.startswith('/'):
                    user_input = user_input[1:]
                
                if not handle_command(user_input, bridge, ui):
                    break
                continue
            
            # Regular chat message
            ui.add_message("You", user_input, "user")
            ui.refresh_display()
            
            # Get response
            response = bridge.chat(user_input)
            
            # Display response
            if response:
                ui.add_message(bridge.main_llm.value, response, "assistant")
                ui.refresh_display()
            
            # Show periodic stats in standard mode
            if isinstance(ui, StandardUI) and bridge.stats[bridge.main_llm]['message_count'] % 10 == 0:
                stats = bridge.get_statistics()
                print(f"\n{Colors.DIM}[Messages: {stats['total_messages']}, "
                      f"Tokens: {stats['total_tokens']:,}, "
                      f"Cost: ${stats['total_cost']:.4f}]{Colors.RESET}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit or press Enter to continue...")
            try:
                if input().lower() in ['exit', 'quit']:
                    break
            except KeyboardInterrupt:
                break
        
        except EOFError:
            break
        
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
            ui.add_message("system", f"Error: {str(e)}", "error")
    
    # Clean exit
    print(f"\n{Colors.YELLOW}Preparing exit...{Colors.RESET}")
    
    # Save configuration
    save_last_configuration()
    
    # Final statistics
    bridge.print_statistics()
    
    # Save conversation if auto-save is on
    if config.AUTO_SAVE:
        print(f"\n{Colors.YELLOW}Auto-saving conversation...{Colors.RESET}")
        bridge.auto_save()
    
    print(f"\n{Colors.GREEN}✨ Thanks for using Multi-LLM Bridge v{VERSION}!{Colors.RESET}")


def main():
    """Main entry point"""
    print(f"\n{Colors.BOLD}{Colors.GREEN}🚀 Multi-LLM Bridge v{VERSION} ({VERSION_DATE}){Colors.RESET}")
    print("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check terminal capabilities
    capabilities = check_terminal_capabilities()
    if not capabilities['color']:
        print("Warning: Terminal may not support colors")
    if not capabilities['unicode']:
        print("Warning: Terminal may not support Unicode characters")
    
    # Load API keys
    api_keys = load_api_keys()
    
    # Check available LLMs
    available_keys = {
        'claude': bool(api_keys.get('anthropic')),
        'openai': bool(api_keys.get('openai')),
        'gemini': bool(api_keys.get('gemini'))
    }
    
    if not any(available_keys.values()):
        print(f"\n{Colors.RED}❌ No API keys found!{Colors.RESET}")
        print("You need at least one API key to use this bridge.")
        
        setup = input("\nWould you like to set up API keys now? (y/n): ").lower()
        if setup == 'y':
            if setup_env_file():
                print("\n✅ API keys configured. Reloading...")
                api_keys = load_api_keys()
                available_keys = {
                    'claude': bool(api_keys.get('anthropic')),
                    'openai': bool(api_keys.get('openai')),
                    'gemini': bool(api_keys.get('gemini'))
                }
            else:
                print("\n❌ No API keys configured. Exiting.")
                return 1
        else:
            return 1
    
    # Show available LLMs
    print(f"\n{Colors.BOLD}{Colors.YELLOW}📋 Available LLMs:{Colors.RESET}")
    print("-" * 40)
    for llm_name, available in available_keys.items():
        status = f"{Colors.GREEN}✅ Ready{Colors.RESET}" if available else f"{Colors.RED}❌ No API key{Colors.RESET}"
        print(f"{llm_name.upper():<10} {status}")
    print("-" * 40)
    
    # Check for previous configuration
    saved_config = load_last_configuration()
    use_saved_config = False
    
    if saved_config:
        # Display saved configuration
        display_saved_configuration(saved_config)
        
        # Ask if user wants to use it
        print(f"\n{Colors.BOLD}{Colors.CYAN}Would you like to use this configuration?{Colors.RESET}")
        choice = input("(y)es, (n)o, or (m)odify: ").lower().strip()
        
        if choice == 'y':
            # Try to apply saved configuration
            if apply_saved_configuration(saved_config, available_keys):
                use_saved_config = True
                print(f"\n{Colors.GREEN}✅ Previous configuration loaded successfully!{Colors.RESET}")
            else:
                print(f"\n{Colors.YELLOW}⚠️  Could not apply saved configuration. Starting fresh...{Colors.RESET}")
        elif choice == 'm':
            # Load but allow modification
            if apply_saved_configuration(saved_config, available_keys):
                use_saved_config = True
                print(f"\n{Colors.GREEN}✅ Configuration loaded. You can now modify it.{Colors.RESET}")
                # Continue to selection process
                use_saved_config = False
    
    if not use_saved_config:
        # Select main LLM
        main_llm = select_main_llm(available_keys)
        if not main_llm:
            return 1
        
        config.MAIN_LLM = main_llm
        
        # Select sub-LLMs
        sub_llms = select_sub_llms(available_keys, main_llm)
        config.ENABLED_SUB_LLMS = sub_llms
    
    # Create interfaces and select models (if not using saved config)
    interfaces = {}
    
    if not use_saved_config:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}📦 Model Selection:{Colors.RESET}")
    
    # Main LLM
    if not use_saved_config or config.MAIN_LLM not in config.SELECTED_MODELS:
        print(f"\n{'='*60}")
        print(f"Setting up {config.MAIN_LLM.value.upper()} as MAIN LLM")
        print(f"{'='*60}")
    
    try:
        # Create temporary interface to get model list
        main_llm = config.MAIN_LLM
        if main_llm == LLMType.CLAUDE:
            temp_interface = ClaudeInterface(api_keys['anthropic'], "claude-3-haiku-20240307")
        elif main_llm == LLMType.OPENAI:
            temp_interface = OpenAIInterface(api_keys['openai'], "gpt-3.5-turbo")
        elif main_llm == LLMType.GEMINI:
            temp_interface = GeminiInterface(api_keys['gemini'], "gemini-1.0-pro")
        
        if not use_saved_config or main_llm not in config.SELECTED_MODELS:
            models = temp_interface.get_model_list()
            selected_model = select_model_interactive(main_llm.value, models)
            config.SELECTED_MODELS[main_llm] = selected_model
        else:
            selected_model = config.SELECTED_MODELS[main_llm]
        
        # Create actual interface with selected model
        interfaces[main_llm] = create_llm_interface(
            main_llm.value,
            api_keys[{'claude': 'anthropic', 'openai': 'openai', 'gemini': 'gemini'}[main_llm.value]],
            selected_model
        )
        
        logger.info(f"Created main interface: {main_llm.value} -> {type(interfaces[main_llm]).__name__}")
        
    except Exception as e:
        print(f"{Colors.RED}Error setting up {main_llm.value}: {str(e)}{Colors.RESET}")
        return 1
    
    # Sub-LLMs
    for llm_type in config.ENABLED_SUB_LLMS:
        if not use_saved_config or llm_type not in config.SELECTED_MODELS:
            print(f"\n{'='*60}")
            print(f"Setting up {llm_type.value.upper()} as SUB-LLM (Tool)")
            print(f"{'='*60}")
        
        try:
            # Create temporary interface to get model list
            if llm_type == LLMType.CLAUDE:
                temp_interface = ClaudeInterface(api_keys['anthropic'], "claude-3-haiku-20240307")
            elif llm_type == LLMType.OPENAI:
                temp_interface = OpenAIInterface(api_keys['openai'], "gpt-3.5-turbo")
            elif llm_type == LLMType.GEMINI:
                temp_interface = GeminiInterface(api_keys['gemini'], "gemini-1.0-pro")
            
            if not use_saved_config or llm_type not in config.SELECTED_MODELS:
                models = temp_interface.get_model_list()
                selected_model = select_model_interactive(llm_type.value, models)
                config.SELECTED_MODELS[llm_type] = selected_model
            else:
                selected_model = config.SELECTED_MODELS[llm_type]
            
            # Create actual interface
            interfaces[llm_type] = create_llm_interface(
                llm_type.value,
                api_keys[{'claude': 'anthropic', 'openai': 'openai', 'gemini': 'gemini'}[llm_type.value]],
                selected_model
            )
            
            logger.info(f"Created sub interface: {llm_type.value} -> {type(interfaces[llm_type]).__name__}")
            
        except Exception as e:
            print(f"{Colors.RED}Error setting up {llm_type.value}: {str(e)}{Colors.RESET}")
            print("Continuing without this sub-LLM...")
            config.ENABLED_SUB_LLMS.remove(llm_type)
    
    # Select display mode (if not using saved config)
    if not use_saved_config:
        total_llms = 1 + len(config.ENABLED_SUB_LLMS)
        config.DISPLAY_MODE = select_display_mode(total_llms)
    
    # Create UI
    ui = create_ui(
        config.DISPLAY_MODE,
        config.MAIN_LLM,
        config.ENABLED_SUB_LLMS,
        {
            'VERBOSE_DISPLAY': config.VERBOSE_DISPLAY,
            'SHOW_TIMESTAMPS': config.SHOW_TIMESTAMPS,
            'SHOW_SUB_AGENT_PANES': config.SHOW_SUB_AGENT_PANES,
            'SELECTED_MODELS': config.SELECTED_MODELS
        }
    )
    
    # Create bridge
    main_interface = interfaces[config.MAIN_LLM]
    sub_interfaces = {llm: interfaces[llm] for llm in config.ENABLED_SUB_LLMS if llm in interfaces}
    
    # Debug logging to verify correct mapping
    logger.info(f"Main interface: {config.MAIN_LLM.value} -> {type(main_interface).__name__}")
    for llm_type, interface in sub_interfaces.items():
        logger.info(f"Sub interface: {llm_type.value} -> {type(interface).__name__}")
    
    bridge = MultiLLMBridge(
        config.MAIN_LLM,
        main_interface,
        sub_interfaces,
        ui
    )
    
    # Show configuration summary
    print(f"\n{Colors.BOLD}{Colors.GREEN}✅ Configuration Complete!{Colors.RESET}")
    print("=" * 80)
    print(f"Main LLM: {config.MAIN_LLM.value.upper()} ({config.SELECTED_MODELS[config.MAIN_LLM]})")
    if config.ENABLED_SUB_LLMS:
        print(f"Sub-LLMs: {', '.join(f'{llm.value.upper()} ({config.SELECTED_MODELS[llm]})' for llm in config.ENABLED_SUB_LLMS)}")
    else:
        print("Sub-LLMs: None (standalone mode)")
    print(f"Display: {config.DISPLAY_MODE.replace('-', ' ').title()}")
    print(f"Log file: {current_log_file}")
    print("=" * 80)
    
    # Save configuration for next time
    save_last_configuration()
    
    time.sleep(1)
    
    # Start interactive chat
    try:
        interactive_chat_loop(bridge, ui)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\n{Colors.RED}Fatal error: {str(e)}{Colors.RESET}")
        print("Check the log file for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
