#!/usr/bin/env python3
"""
Multi-LLM Agent Bridge - Curses UI Version
Version: 6.2.9 (2025-07-12)

An AI agent that can use other LLMs (including itself) as tools, displaying
all interactions in a dynamic, multi-window terminal interface with full
keyboard control for scrolling and window selection.

Changelog:
- v6.2.9: Fixed 'AttributeError: MultiLLMAgentBridge object has no attribute _initialize_stats'
          by restoring the missing `_initialize_stats` method within the `MultiLLMAgentBridge` class.
- v6.2.8: Fixed IndentationError after 'except' statement by adding 'pass'.
- v6.2.7: Fixed 'AttributeError: MultiLLMAgentBridge object has no attribute stdscr'
          by correctly placing `stdscr.nodelay(True)` within `CursesUI.__init__`.
          Ensured `nodelay` state is managed consistently.
- v6.2.6: Further refined Curses UI input handling to eliminate flashing and
          typing issues. Implemented non-blocking input, explicit window refreshes,
          and more robust redraw logic, particularly for the input line.
- v6.2.5: Fixed input prompt flashing and typing issues by optimizing UI redraws.
          Input line is now redrawn only when necessary, improving responsiveness.
          Added specific input line window for better control.
- v6.2.4: Improved Curses UI rendering for scrollback, window geometry adaptation,
          and correct newline/carriage return handling in all panes.
          Increased default scrollback for sub-panes and added better column
          width calculation.
- v6.2.3: Corrected KeyError on startup when loading a saved configuration by
          ensuring consistent use of string keys for model selections.
- v6.2.2: Corrected ImportError for Gemini's 'Candidate' type.
- v6.2.1: Fixed NameError for 'Union' by adding the correct import.
- v6.2.0: Corrected narrow column formatting in sub-agent panes.
- v6.1.x: Bug fixes for CursesUI implementation and error handling.
- v6.0.0: Re-architected with the `curses` library for a full TUI experience.
"""
import os
import sys
import json
import textwrap
from typing import Dict, List, Optional, Any, Tuple, Literal, Type, Union
from enum import Enum
from abc import ABC, abstractmethod
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai
import logging
from pathlib import Path
from datetime import datetime
import time
import shutil
import platform
import signal
import curses

# --- Type Imports for Explicit Checking ---
from anthropic.types import Message as AnthropicMessage
from openai.types.chat.chat_completion_message import ChatCompletionMessage as OpenAIMessage
from google.ai.generativelanguage import Candidate as GeminiCandidate

# --- CONFIGURATION ---
VERSION = "6.2.9" # Updated version
CONFIG_FILE_NAME = ".llm_bridge_config.json"

class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"

PROVIDERS: List[Provider] = list(Provider)

PRICE_MAPPING = {
    "claude-opus-4-20250514": {"input": 20.00, "output": 100.00},
    "claude-sonnet-4-20250514": {"input": 4.00, "output": 20.00},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gemini-1.5-pro-latest": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash-latest": {"input": 0.35, "output": 1.05},
}

class Config:
    SHOW_SUB_AGENT_PANES = True
    MAX_MAIN_SCROLLBACK_LINES = 1000 # Max lines for the main conversation pane
    MAX_SUB_SCROLLBACK_LINES = 500   # Max lines for sub-agent panes

config = Config()

class Colors:
    RESET = '\033[0m'; BOLD = '\033[1m'
    RED = '\033[31m'; GREEN = '\033[32m'; YELLOW = '\033[33m'
    CYAN = '\033[36m'; BRIGHT_BLACK = '\033[90m'

# --- LOGGING SETUP ---
def setup_logging() -> Tuple[logging.Logger, Path]:
    log_dir = Path(".logs"); log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"multi_agent_bridge_{timestamp}.log"
    logger = logging.getLogger("LLM_Bridge"); logger.setLevel(logging.DEBUG)
    if logger.hasHandlers(): logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    return logger, log_file

logger, current_log_file = setup_logging()
logger.info(f"Session started. Logging to: {current_log_file}")

# --- DEPENDENCY & KEY MANAGEMENT ---
def check_dependencies():
    missing = []
    packages = ["anthropic", "openai", "google.generativeai", "dotenv"]
    for pkg in packages:
        try:
            if pkg == "google.generativeai": __import__("google.generativeai")
            elif pkg == "dotenv": __import__("dotenv")
            else: __import__(pkg)
        except ImportError:
            pip_name = "python-dotenv" if pkg == "dotenv" else "google-generativeai" if pkg == "google.generativeai" else pkg
            missing.append(pip_name)
    if missing:
        print(f"\n⚠️ Missing required packages. Please run: pip install {' '.join(missing)}")
        return False
    return True

def load_api_keys() -> Dict[str, Optional[str]]:
    try:
        from dotenv import load_dotenv
        script_dir = Path(__file__).resolve().parent
        dotenv_path = script_dir / '.env'
        if dotenv_path.is_file(): load_dotenv(dotenv_path=dotenv_path, verbose=True)
    except ImportError: print("⚠️ python-dotenv not installed. Relying on system environment variables.")
    return {p.value: os.getenv(f"{p.value.upper()}_API_KEY") for p in PROVIDERS}

def setup_env_file():
    print("\n🔧 API Key Setup: Some keys are missing.")
    if input("Create a .env file? (y/n): ").lower() != 'y': return False
    with open(".env", "w") as f:
        f.write("# API Keys for Multi-LLM Agent Bridge\n")
        for p in PROVIDERS:
            f.write(f"{p.value.upper()}_API_KEY={input(f'Enter your {p.value.capitalize()} API key: ').strip() or 'your-key-here'}\n")
    print("✅ .env file created.")
    if not Path(".gitignore").exists():
        with open(".gitignore", "w") as f: f.write(".env\n.logs/\n")
    print("✅ Created .gitignore to protect secrets.")
    return True

# --- LLM INTERFACE ABSTRACTION ---
class LLMInterface(ABC):
    def __init__(self, api_key: str, model: str):
        if not api_key: raise ValueError(f"API key is missing.")
        self.model = model
    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]: pass
    @abstractmethod
    def list_models(self) -> List[str]: pass
    def _calculate_cost(self, i_tok: int, o_tok: int) -> float:
        p = PRICE_MAPPING.get(self.model)
        if not p: return 0.0
        return ((i_tok / 1_000_000) * p.get("input", 0.0)) + ((o_tok / 1_000_000) * p.get("output", 0.0))

class AnthropicInterface(LLMInterface):
    def __init__(self, api_key: str, model: str): super().__init__(api_key, model); self.client = Anthropic(api_key=api_key)
    def list_models(self) -> List[str]: return sorted([m for m in PRICE_MAPPING if "claude" in m], reverse=True)
    def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        system_prompt = kwargs.get("system_prompt", "")
        tools = kwargs.get("tools", [])
        try:
            r = self.client.messages.create(model=self.model, max_tokens=4096, messages=messages, system=system_prompt, tools=tools)
            cost = self._calculate_cost(r.usage.input_tokens, r.usage.output_tokens)
            return {"success": True, "response_obj": r, "usage": r.usage.to_dict(), "cost": cost}
        except Exception as e: return {"success": False, "error": str(e)}

class OpenaiInterface(LLMInterface):
    def __init__(self, api_key: str, model: str): super().__init__(api_key, model); self.client = OpenAI(api_key=api_key)
    def list_models(self) -> List[str]:
        try: return sorted([m.id for m in self.client.models.list() if "gpt" in m.id or "o1" in m.id], reverse=True)
        except Exception as e: logger.error(f"Failed to fetch OpenAI models: {e}"); return sorted([m for m in PRICE_MAPPING if "gpt" in m], reverse=True)
    def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        all_msg = [{"role": "system", "content": kwargs.get("system_prompt", "")}] + messages
        tools = kwargs.get("tools", [])
        try:
            openai_tools = [{"type": "function", "function": t} for t in tools]
            r = self.client.chat.completions.create(model=self.model, messages=all_msg, tools=openai_tools or None, tool_choice="auto" if openai_tools else None, max_tokens=4096)
            cost = self._calculate_cost(r.usage.prompt_tokens, r.usage.completion_tokens)
            return {"success": True, "response_obj": r.choices[0].message, "usage": r.usage.to_dict(), "cost": cost}
        except Exception as e: return {"success": False, "error": str(e)}

class GeminiInterface(LLMInterface):
    def __init__(self, api_key: str, model: str): super().__init__(api_key, model); genai.configure(api_key=api_key)
    def list_models(self) -> List[str]:
        try:
            models = []
            excluded = ['-tts', '-embedding', '-aqa']
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods and not any(m.name.endswith(s) for s in excluded):
                    models.append(m.name[len('models/'):])
            return sorted(models, reverse=True)
        except Exception as e: logger.error(f"Failed to fetch Gemini models: {e}"); return sorted([m for m in PRICE_MAPPING if "gemini" in m], reverse=True)
    def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        gemini_tools = [{"function_declarations": kwargs.get("tools", [])}] if kwargs.get("tools") else None
        try:
            gemini_model = genai.GenerativeModel(self.model, system_instruction=kwargs.get("system_prompt", ""), tools=gemini_tools, generation_config=genai.GenerationConfig(max_output_tokens=8192))
            history = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                content = msg.get('content')
                if isinstance(content, list):
                    text = " ".join([c.text if hasattr(c, 'text') else c.get('text', '') for c in content if (hasattr(c, 'text') or (isinstance(c, dict) and c.get('type') == 'text'))])
                    history.append({'role': role, 'parts': [text]})
                else: history.append({'role': role, 'parts': [content]})
            latest_prompt = history.pop()['parts'] if history else []
            chat = gemini_model.start_chat(history=history)
            r = chat.send_message(latest_prompt, stream=False)
            
            output_text = "".join([part.text for part in r.candidates[0].content.parts if hasattr(part, 'text')])
            
            i_tok = gemini_model.count_tokens(chat.history).total_tokens
            o_tok = gemini_model.count_tokens(output_text).total_tokens if output_text else 0
            cost = self._calculate_cost(i_tok, o_tok)
            return {"success": True, "response_obj": r.candidates[0], "usage": {"input_tokens": i_tok, "output_tokens": o_tok}, "cost": cost}
        except Exception as e: return {"success": False, "error": str(e)}

class CursesUI:
    def __init__(self, stdscr, active_providers: List[Provider], primary_agent: Provider):
        self.stdscr = stdscr 
        self.active_providers = active_providers
        self.primary_agent = primary_agent
        
        self.panes = {p: {'lines': ["No activity yet."], 'scroll_offset': 0} for p in self.active_providers}
        self.panes['main'] = {'lines': [], 'scroll_offset': 0}
        
        self.window_order: List[Union[Provider, Literal['main']]] = self.active_providers + ['main']
        self.active_window_idx = len(self.window_order) - 1 

        curses.curs_set(1) 
        curses.start_color()
        curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK) 
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  

        self.last_terminal_size = (0, 0)
        
        self.input_win = None
        
        self.stdscr.nodelay(True) 
            
    def get_active_window_key(self) -> Union[Provider, Literal['main']]:
        return self.window_order[self.active_window_idx]

    def cycle_active_window(self):
        self.active_window_idx = (self.active_window_idx + 1) % len(self.window_order)
        self.panes[self.get_active_window_key()]['scroll_offset'] = 0

    def scroll_active_window(self, direction: int, amount: int):
        key = self.get_active_window_key()
        pane = self.panes[key]
        
        if direction == -1: 
            pane['scroll_offset'] = min(pane['scroll_offset'] + amount, len(pane['lines']) -1)
        elif direction == 1: 
            pane['scroll_offset'] = max(0, pane['scroll_offset'] - amount)
        elif direction == 0: 
            pane['scroll_offset'] = 0 
        elif direction == 2: 
            pane['scroll_offset'] = len(pane['lines']) -1 
            
    def _add_lines_to_pane(self, pane_key: Union[Provider, Literal['main']], message: str, max_lines: int):
        lines_to_add = []
        term_height, term_width = self.stdscr.getmaxyx()
        
        if pane_key == 'main':
            pane_width = term_width - 4 
        else:
            num_sub_panes = len(self.active_providers)
            pane_width = max(1, term_width // num_sub_panes) - 4 
        
        drawable_width_for_wrap = max(1, pane_width) 

        for line in message.split('\n'):
            wrapped = textwrap.wrap(line, width=drawable_width_for_wrap, subsequent_indent="  ")
            lines_to_add.extend(wrapped)
        
        self.panes[pane_key]['lines'].extend(lines_to_add)
        
        current_lines = self.panes[pane_key]['lines']
        if len(current_lines) > max_lines:
            self.panes[pane_key]['lines'] = current_lines[-max_lines:]
        
        self.panes[pane_key]['scroll_offset'] = 0
            
    def add_line_to_main_log(self, message: str):
        self._add_lines_to_pane('main', message, config.MAX_MAIN_SCROLLBACK_LINES)

    def update_provider_pane(self, provider: Provider, query: str, response: str):
        self.panes[provider]['lines'] = [] 
        self._add_lines_to_pane(provider, f"Q: {query}", config.MAX_SUB_SCROLLBACK_LINES)
        self._add_lines_to_pane(provider, "---", config.MAX_SUB_SCROLLBACK_LINES)
        self._add_lines_to_pane(provider, f"A: {response}", config.MAX_SUB_SCROLLBACK_LINES)
        
        self.panes[provider]['scroll_offset'] = 0

    def draw(self, stats):
        term_height, term_width = self.stdscr.getmaxyx()

        if (term_height, term_width) != self.last_terminal_size:
            self.stdscr.clear() 
            self.last_terminal_size = (term_height, term_width)
            curses.resizeterm(term_height, term_width) 
            self.input_win = None 

        if config.SHOW_SUB_AGENT_PANES and len(self.active_providers) > 0:
            self._draw_multi_pane_layout(term_width, term_height, stats)
        else:
            self._draw_focused_layout(term_width, term_height)
        
        self.stdscr.noutrefresh()

    def _draw_multi_pane_layout(self, term_width, term_height, stats):
        sub_pane_target_height = 14 
        min_main_area_height = 5 
        
        available_sub_pane_height = term_height - min_main_area_height
        
        pane_height = min(sub_pane_target_height, max(5, available_sub_pane_height // 2))

        num_sub_panes = len(self.active_providers)
        pane_width = max(1, term_width // num_sub_panes)
        
        last_pane_width = pane_width + (term_width - (pane_width * num_sub_panes))

        for i, provider in enumerate(self.active_providers):
            is_active = self.get_active_window_key() == provider
            current_pane_width = last_pane_width if i == num_sub_panes - 1 else pane_width
            
            current_pane_width = max(1, min(current_pane_width, term_width - (i * pane_width)))
            
            win = self.stdscr.subwin(max(1, pane_height), current_pane_width, 0, i * pane_width)
            win.erase() 

            win.attron(curses.color_pair(1) if is_active else curses.color_pair(2))
            win.border()
            win.attroff(curses.color_pair(1) if is_active else curses.color_pair(2))
            
            title = f" {provider.value.upper()} "
            if provider == self.primary_agent: title += "(AGENT) "
            
            try:
                win.addstr(0, 2, title[:current_pane_width-4], curses.A_BOLD if is_active else curses.A_NORMAL)
            except curses.error:
                pass 

            self._draw_pane_content(win, self.panes[provider], pane_height, current_pane_width)
            win.noutrefresh() 

        main_pane_start_row = pane_height
        main_pane_height = term_height - main_pane_start_row - 1 
        self._draw_main_conversation_pane(term_width, term_height, main_pane_start_row, main_pane_height)

    def _draw_focused_layout(self, term_width, term_height):
        main_pane_height = term_height - 1 
        self._draw_main_conversation_pane(term_width, term_height, 0, main_pane_height)

    def _draw_main_conversation_pane(self, term_width, term_height, start_row, main_pane_height):
        is_main_active = self.get_active_window_key() == 'main'
        
        main_pane_height = max(1, main_pane_height)
        
        win = self.stdscr.subwin(main_pane_height, term_width, start_row, 0)
        win.erase() 

        win.attron(curses.color_pair(1) if is_main_active else curses.color_pair(2))
        win.border()
        win.attroff(curses.color_pair(1) if is_main_active else curses.color_pair(2))
        
        try:
            win.addstr(0, 2, " MAIN CONVERSATION ", curses.A_BOLD if is_main_active else curses.A_NORMAL)
        except curses.error:
            pass 

        self._draw_pane_content(win, self.panes['main'], main_pane_height, term_width)
        win.noutrefresh() 

    def _draw_pane_content(self, win, pane_data, win_height, win_width):
        content_h = win_height - 2 
        drawable_width = win_width - 4 

        if content_h <= 0 or drawable_width <= 0:
            return 

        lines = pane_data['lines']
        scroll_offset = pane_data['scroll_offset']

        display_start_idx = max(0, len(lines) - content_h - scroll_offset)
        display_end_idx = min(len(lines), display_start_idx + content_h)

        for r, line_idx in enumerate(range(display_start_idx, display_end_idx)):
            line = lines[line_idx]
            display_line = line[:drawable_width]
            
            try:
                if pane_data == self.panes['main'] and display_line.startswith("You:"):
                    win.addstr(r + 1, 2, "You:", curses.color_pair(3) | curses.A_BOLD)
                    win.addstr(display_line[4:])
                else:
                    win.addstr(r + 1, 2, display_line)
            except curses.error:
                pass

    def _draw_input_line(self, input_buffer: str):
        h, w = self.stdscr.getmaxyx()
        
        if self.input_win is None or self.input_win.getmaxyx() != (1, w) or self.input_win.getbegyx() != (h - 1, 0):
            try:
                if self.input_win is not None:
                    del self.input_win
            except AttributeError:
                pass
            self.input_win = self.stdscr.subwin(1, w, h - 1, 0)
            self.input_win.keypad(True) 

        self.input_win.erase() 
        
        prompt = "> "
        self.input_win.addstr(0, 0, prompt) 
        
        max_input_width = w - len(prompt) - 1 
        display_input = input_buffer[:max_input_width]
        self.input_win.addstr(0, len(prompt), display_input)
        
        self.input_win.move(0, len(prompt) + len(display_input))
        
        self.input_win.noutrefresh() 

class MultiLLMAgentBridge:
    def __init__(self, api_keys: Dict[str, str], primary_agent: Provider, model_selections: Dict[str, str]):
        self.primary_agent_name = primary_agent
        self.model_selections = model_selections
        self.conversation_history = []
        
        self.interfaces: Dict[Provider, LLMInterface] = {}
        for p in PROVIDERS:
            if api_keys.get(p.value):
                class_name = f"{p.name.capitalize()}Interface"
                self.interfaces[p] = globals()[class_name](api_keys[p.value], model_selections[p.value])
        
        self.primary_interface = self.interfaces[primary_agent]
        self.stats = self._initialize_stats() # This line caused the error!
        self.ui: Optional[CursesUI] = None 

    def _initialize_stats(self) -> Dict: # THIS METHOD IS NOW HERE!
        stats = {p.value: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "model": self.model_selections[p.value]} for p in self.interfaces.keys()}
        stats["session_start"] = datetime.now()
        stats["primary_agent"] = self.primary_agent_name.value
        return stats

    def _generate_tools_schema(self) -> List[Dict]:
        tools = []
        for provider in self.interfaces.keys():
            tool_name = f"query_{provider.value}"
            desc = f"Query the {provider.value.capitalize()} API to get its perspective or capabilities. Can be used to query your own model for summarization or re-evaluation."
            props = {"prompt": {"type": "string", "description": "A clear, self-contained prompt."}}
            if self.primary_agent_name == Provider.ANTHROPIC:
                tools.append({"name": tool_name, "description": desc, "input_schema": {"type": "object", "properties": props, "required": ["prompt"]}})
            else:
                tools.append({"name": tool_name, "description": desc, "parameters": {"type": "object", "properties": props, "required": ["prompt"]}})
        return tools

    def _update_stats(self, provider: Provider, usage: Dict, cost: float):
        self.stats[provider.value]['input_tokens'] += usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
        self.stats[provider.value]['output_tokens'] += usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
        self.stats[provider.value]['cost'] += cost

    def _execute_tool_call(self, tool_name: str, tool_input: Dict) -> Dict:
        target_provider = Provider(tool_name.split("_")[1])
        client = self.interfaces[target_provider]
        prompt = tool_input.get("prompt", "")
        self.ui.update_provider_pane(target_provider, prompt, "[Waiting for response...]")
        
        self.ui.draw(self.stats) 
        self.ui._draw_input_line("") 
        curses.doupdate()

        self.ui.stdscr.nodelay(False) 

        response = client.chat(messages=[{"role": "user", "content": prompt}], system_prompt="You are a helpful assistant. Please answer the user's prompt directly and concisely. Do not use any tools.", tools=[])
        
        self.ui.stdscr.nodelay(True) 

        if response["success"]:
            self._update_stats(target_provider, response['usage'], response['cost'])
            text_content = self._extract_text_from_response(response['response_obj'])
            self.ui.update_provider_pane(target_provider, prompt, text_content)
            return {"success": True, "content": text_content}
        else:
            self.ui.update_provider_pane(target_provider, prompt, f"ERROR: {response['error']}")
            return {"success": False, "error": response['error']}

    def _extract_text_from_response(self, response_obj: Any) -> str:
        if isinstance(response_obj, AnthropicMessage):
            return "".join([block.text for block in response_obj.content if hasattr(block, 'text')])
        elif isinstance(response_obj, OpenAIMessage):
            return response_obj.content or ""
        elif isinstance(response_obj, GeminiCandidate):
            if hasattr(response_obj, 'content') and hasattr(response_obj.content, 'parts'):
                return "".join([part.text for part in response_obj.content.parts if hasattr(part, 'text')])
        return ""

    def chat(self, user_prompt: str):
        self.ui.add_line_to_main_log(f"You: {user_prompt}")
        
        self.ui.draw(self.stats) 
        curses.doupdate() 
        
        self.conversation_history.append({"role": "user", "content": user_prompt})
        
        other_tools = [f"`query_{p.value}`" for p in self.interfaces if p != self.primary_agent_name]
        self_tool = f"`query_{self.primary_agent_name.value}`"
        system_prompt = (f"You are a sophisticated AI agent, '{self.primary_agent_name.value}'. Your goal is to provide comprehensive answers. "
                         f"You can use tools to query other AI models: {', '.join(other_tools)}. "
                         f"You can also call your own model using the {self_tool} tool to summarize, reflect, or re-evaluate your own thoughts. "
                         "Synthesize information from all sources into a final, conclusive answer for the user.")

        current_messages = self.conversation_history.copy()
        
        while True: 
            thinking_line = f"🤖 {self.primary_agent_name.value.capitalize()}: [thinking...]"
            self.ui.add_line_to_main_log(thinking_line) 
            self.ui.draw(self.stats) 
            curses.doupdate() 
            
            self.ui.stdscr.nodelay(False) 
            response = self.primary_interface.chat(messages=current_messages, system_prompt=system_prompt, tools=self._generate_tools_schema())
            self.ui.stdscr.nodelay(True) 

            try:
                if self.ui.panes['main']['lines'] and self.ui.panes['main']['lines'][-1] == thinking_line:
                    self.ui.panes['main']['lines'].pop() 
            except IndexError:
                pass 

            if not response["success"]:
                self.ui.add_line_to_main_log(f"ERROR: {response['error']}")
                self.ui.draw(self.stats) 
                curses.doupdate()
                break
            
            self._update_stats(self.primary_agent_name, response['usage'], response['cost'])
            
            response_obj = response['response_obj']
            
            tool_calls = []
            interim_text = self._extract_text_from_response(response_obj)

            if isinstance(response_obj, AnthropicMessage):
                for block in response_obj.content:
                    if block.type == 'tool_use': tool_calls.append({'id': block.id, 'name': block.name, 'input': block.input})
            elif isinstance(response_obj, OpenAIMessage):
                if response_obj.tool_calls:
                    for tc in response_obj.tool_calls: tool_calls.append({'id': tc.id, 'name': tc.function.name, 'input': json.loads(tc.function.arguments)})
            elif isinstance(response_obj, GeminiCandidate):
                if hasattr(response_obj, 'content') and hasattr(response_obj.content, 'parts'):
                    for part in response_obj.content.parts:
                        if part.function_call: tool_calls.append({'id': part.function_call.name, 'name': part.function_call.name, 'input': dict(part.function_call.args)})

            if interim_text:
                self.ui.add_line_to_main_log(f"🤖 {self.primary_agent_name.value.capitalize()}: {interim_text}")
                self.ui.draw(self.stats) 
                curses.doupdate()

            if not tool_calls:
                break 

            self.conversation_history.append({"role": "assistant", "content": response_obj.model_dump_json() if hasattr(response_obj, 'model_dump_json') else str(response_obj)})
            
            tool_results = []
            for tool_call in tool_calls:
                self.ui.add_line_to_main_log(f"⚙️ Calling Tool: {tool_call['name']}({json.dumps(tool_call['input'])})")
                self.ui.draw(self.stats) 
                curses.doupdate()
                
                result = self._execute_tool_call(tool_call['name'], tool_call['input'])
                tool_result_content = result['content'] if result['success'] else f"Error: {result['error']}"
                
                self.ui.add_line_to_main_log(f"✅ Tool Result: {tool_result_content}")
                self.ui.draw(self.stats) 
                curses.doupdate()

                if self.primary_agent_name == Provider.ANTHROPIC: tool_results.append({"type": "tool_result", "tool_use_id": tool_call['id'], "content": tool_result_content})
                else: tool_results.append({"role": "tool", "tool_call_id": tool_call['id'], "name": tool_call['name'], "content": tool_result_content})
            
            current_messages = self.conversation_history.copy()
            if self.primary_agent_name == Provider.ANTHROPIC: current_messages.append({"role": "user", "content": tool_results})
            else: current_messages.extend(tool_results)

    def run_curses(self, stdscr_main_arg): 
        self.ui = CursesUI(stdscr_main_arg, list(self.interfaces.keys()), self.primary_agent_name)
        input_buffer = ""
        
        self.ui.draw(self.stats) 
        self.ui._draw_input_line(input_buffer) 
        curses.doupdate() 

        while True:
            h, w = self.ui.stdscr.getmaxyx() 
            if (h, w) != self.ui.last_terminal_size:
                self.ui.last_terminal_size = (0,0) 
                self.ui.draw(self.stats) 
                self.ui._draw_input_line(input_buffer) 
                curses.doupdate()
                continue 

            key = self.ui.input_win.getch() 

            if key == -1: 
                time.sleep(0.01) 
                continue 
            
            if key == ord('\t'):
                self.ui.cycle_active_window()
                self.ui.draw(self.stats) 
                self.ui._draw_input_line(input_buffer) 
                curses.doupdate()
            elif key == curses.KEY_UP:
                self.ui.scroll_active_window(-1, 1)
                self.ui.draw(self.stats) 
                self.ui._draw_input_line(input_buffer) 
                curses.doupdate()
            elif key == curses.KEY_DOWN:
                self.ui.scroll_active_window(1, 1)
                self.ui.draw(self.stats) 
                self.ui._draw_input_line(input_buffer) 
                curses.doupdate()
            elif key == curses.KEY_PPAGE:
                self.ui.scroll_active_window(-1, h // 2)
                self.ui.draw(self.stats) 
                self.ui._draw_input_line(input_buffer)
                curses.doupdate()
            elif key == curses.KEY_NPAGE:
                self.ui.scroll_active_window(1, h // 2)
                self.ui.draw(self.stats) 
                self.ui._draw_input_line(input_buffer)
                curses.doupdate()
            elif key == curses.KEY_HOME:
                self.ui.scroll_active_window(2, 0) 
                self.ui.draw(self.stats) 
                self.ui._draw_input_line(input_buffer)
                curses.doupdate()
            elif key == curses.KEY_END:
                self.ui.scroll_active_window(0, 0) 
                self.ui.draw(self.stats) 
                self.ui._draw_input_line(input_buffer)
                curses.doupdate()
            elif key == curses.KEY_BACKSPACE or key == 127: 
                input_buffer = input_buffer[:-1]
                self.ui._draw_input_line(input_buffer) 
                curses.doupdate()
            elif key == ord('\n'): 
                if input_buffer.lower() in ['exit', 'quit', 'q']: break
                
                self.ui._draw_input_line("")
                curses.doupdate()

                self.chat(input_buffer) 
                input_buffer = "" 
                
                self.ui.draw(self.stats) 
                self.ui._draw_input_line(input_buffer) 
                curses.doupdate() 
            elif 32 <= key <= 255: 
                try:
                    char = chr(key)
                    input_buffer += char
                    self.ui._draw_input_line(input_buffer) 
                    curses.doupdate()
                except ValueError:
                    pass 

def select_provider_from_list(prompt: str, options: List[Provider]) -> Provider:
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt.value.capitalize()}")
    while True:
        try:
            choice = int(input(f"\nSelect (1-{len(options)}): ")) - 1
            if 0 <= choice < len(options): return options[choice]
            print("Invalid choice.")
        except (ValueError, IndexError): print("Invalid choice.")

def select_model_from_grouped_list(prompt: str, all_models: List[str], provider: Provider) -> str:
    print(prompt)
    model_groups = {
        "GPT-4o Series": [m for m in all_models if "gpt-4o" in m], "GPT-4 Series": [m for m in all_models if m.startswith("gpt-4") and "o" not in m],
        "O1 Series (Reasoning)": [m for m in all_models if m.startswith("o1")], "GPT-3.5 Series": [m for m in all_models if "gpt-3.5" in m],
        "Claude 3.5 Series": [m for m in all_models if "claude-3-5" in m], "Claude 3 Series": [m for m in all_models if m.startswith("claude-3-") and "5" not in m],
        "Gemini 1.5 Series": [m for m in all_models if "gemini-1.5" in m], "Gemini 1.0 Series": [m for m in all_models if "gemini-1.0" in m],
    }
    numbered_models = []
    provider_keys = [provider.value]
    if provider == Provider.OPENAI: provider_keys.extend(["gpt", "o1"])
    
    for group_name, models_in_group in model_groups.items():
        if any(p in group_name.lower() for p in provider_keys):
            if models_in_group:
                print(f"\n--- {group_name} ---")
                for model in sorted(models_in_group, reverse=True):
                    if model not in numbered_models: numbered_models.append(model)
    
    other_models = [m for m in all_models if m not in numbered_models]
    if other_models:
        print("\n--- Other Models ---")
        for model in sorted(other_models, reverse=True): numbered_models.append(model)

    for i, model in enumerate(numbered_models): print(f"  {i+1}. {model}")
    
    while True:
        try:
            choice = int(input(f"\nSelect model (1-{len(numbered_models)}): ")) - 1
            if 0 <= choice < len(numbered_models): return numbered_models[choice]
            print("Invalid choice.")
        except (ValueError, IndexError): print("Invalid choice.")

def save_configuration(primary_agent: Provider, model_selections: Dict[Provider, str]):
    config_path = Path(__file__).resolve().parent / CONFIG_FILE_NAME
    try:
        with open(config_path, 'w') as f:
            json.dump({"primary_agent": primary_agent.value, "model_selections": {p.value: m for p, m in model_selections.items()}}, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")

def load_configuration() -> Optional[Dict]:
    config_path = Path(__file__).resolve().parent / CONFIG_FILE_NAME
    if config_path.is_file():
        try:
            with open(config_path, 'r') as f: return json.load(f)
        except Exception as e: print(f"⚠️ Could not load configuration file: {e}"); return None
    return None

def main_wrapper(stdscr, api_keys, primary_agent, model_selections):
    bridge = MultiLLMAgentBridge(api_keys, primary_agent, model_selections)
    
    try:
        bridge.run_curses(stdscr)
    finally:
        enum_model_selections = {Provider(k) if isinstance(k, str) else k: v for k, v in model_selections.items()}
        save_configuration(primary_agent, enum_model_selections)
        print("\n✅ Configuration saved. Exiting.")

def main():
    print(f"🚀 Initializing Multi-LLM Agent Bridge v{VERSION}...")
    if not check_dependencies(): sys.exit(1)
    api_keys = load_api_keys()
    if any(not k for k in api_keys.values()):
        if not setup_env_file(): print("❌ API keys are required. Exiting."); sys.exit(1)
        api_keys = load_api_keys()
        if any(not k for k in api_keys.values()): print("❌ Still missing API keys. Exiting."); sys.exit(1)

    print("✅ All API keys loaded.")
    
    primary_agent: Optional[Provider] = None
    model_selections: Optional[Dict[str, str]] = None
    
    saved_config = load_configuration()
    if saved_config:
        print("\n--- Found Saved Configuration ---")
        print(f"Primary Agent: {saved_config.get('primary_agent')}")
        for p_val, m in saved_config.get('model_selections', {}).items():
            print(f"  - {p_val.capitalize()}: {m}")
        
        if input("\nUse this configuration? (y/n): ").lower() == 'y':
            primary_agent = Provider(saved_config.get('primary_agent'))
            model_selections = {str(k): v for k, v in saved_config.get('model_selections', {}).items()}

    if not primary_agent or not model_selections:
        interfaces = {p: globals()[f"{p.name.capitalize()}Interface"](api_keys[p.value], "") for p in PROVIDERS if api_keys[p.value]}
        print("Fetching available models...")
        available_models = {p: i.list_models() for p, i in interfaces.items()}
        
        print("\n--- AGENT SETUP ---")
        primary_agent = select_provider_from_list("\n1. Select the Primary Agent (Provider):", list(interfaces.keys()))

        model_selections_enum = {}
        print("\n2. Select the specific model for each provider:")
        for provider in interfaces.keys():
            model_selections_enum[provider] = select_model_from_grouped_list(f"  - {provider.value.capitalize()} Model:", available_models[provider], provider)
        model_selections = {p.value: m for p, m in model_selections_enum.items()}

    logger.info(f"Primary Agent: {primary_agent.value}, Models: {model_selections}")
    print("\n🚀 Starting Multi-Pane Interface...")
    time.sleep(1)

    curses.wrapper(main_wrapper, api_keys, primary_agent, model_selections)

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
