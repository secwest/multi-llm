#!/usr/bin/env python3
"""
Multi-LLM Agent Bridge - Multi-Pane Version
Version: 5.3.0 (2025-07-11)

An AI agent that can use other LLMs (including itself) as tools, displaying
all interactions in a dynamic, multi-window terminal interface.

Changelog:
- v5.3.0: Corrected tool execution logic to prevent passing invalid tool schemas
          to sub-agents, fixing the Anthropic API error.
- v5.2.x: Bug fixes for API key loading, client initialization, and response parsing.
- v5.1.x: Bug fixes for API key loading and client initialization.
- v5.0.0: Refactored core provider logic to use Enum.
- v4.x: Added multi-pane UI, self-calling tools, config management, and bug fixes.
"""
import os
import sys
import json
import textwrap
from typing import Dict, List, Optional, Any, Tuple, Literal, Type
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

# --- Type Imports for Explicit Checking ---
from anthropic.types import Message as AnthropicMessage
from openai.types.chat.chat_completion_message import ChatCompletionMessage as OpenAIMessage
from google.ai.generativelanguage import Candidate as GeminiCandidate

# --- CONFIGURATION ---
VERSION = "5.3.0"
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

class TerminalUI:
    def __init__(self, active_providers: List[Provider]):
        self.last_terminal_size = (0, 0)
        self.active_providers = active_providers
        self.provider_pane_content: Dict[Provider, List[str]] = {p: ["No activity yet."] for p in active_providers}
        self.install_resize_handler()

    def install_resize_handler(self):
        if hasattr(signal, 'SIGWINCH'): signal.signal(signal.SIGWINCH, self._handle_resize)

    def _handle_resize(self, signum, frame): self.last_terminal_size = (0, 0)

    def get_terminal_size(self) -> Tuple[int, int]:
        try: return shutil.get_terminal_size()
        except OSError: return (120, 40)

    def clear_screen(self): os.system('cls' if os.name == 'nt' else 'clear')
    def move_cursor(self, row, col): print(f"\033[{row};{col}H", end='')

    def update_provider_pane(self, provider: Provider, query: str, response: str):
        pane_width = self.get_terminal_size()[0] // len(self.active_providers) - 4
        q_wrapped = textwrap.wrap(f"Q: {query}", width=pane_width)
        r_wrapped = textwrap.wrap(f"A: {response}", width=pane_width)
        self.provider_pane_content[provider] = q_wrapped + ["-" * pane_width] + r_wrapped

    def draw_layout(self, stats: Dict[str, Any], conversation_log: List[str]):
        term_width, term_height = self.get_terminal_size()
        if (term_width, term_height) != self.last_terminal_size:
            self.clear_screen()
            self.last_terminal_size = (term_width, term_height)

        if config.SHOW_SUB_AGENT_PANES and len(self.active_providers) > 1:
            self._draw_multi_pane_layout(term_width, term_height, stats, conversation_log)
        else:
            self._draw_focused_layout(term_width, term_height, conversation_log)

        self.move_cursor(term_height, 1)
        print(f"{Colors.BOLD}{Colors.GREEN}You: {Colors.RESET}", end='')
        sys.stdout.flush()

    def _draw_multi_pane_layout(self, term_width, term_height, stats, conversation_log):
        pane_height = 14
        pane_width = term_width // len(self.active_providers)
        for i, provider in enumerate(self.active_providers):
            col_start = i * pane_width + 1
            title = f" {provider.value.upper()} "
            if stats.get('primary_agent') == provider.value: title += "(AGENT) "
            self.move_cursor(1, col_start); print(f"{Colors.BRIGHT_BLACK}┌" + title.center(pane_width - 2, "─") + f"┐{Colors.RESET}")
            for r in range(2, pane_height):
                self.move_cursor(r, col_start); print(f"{Colors.BRIGHT_BLACK}│{' ' * (pane_width - 2)}│{Colors.RESET}")
            self.move_cursor(pane_height, col_start); print(f"{Colors.BRIGHT_BLACK}└" + "─" * (pane_width - 2) + f"┘{Colors.RESET}")
            content = self.provider_pane_content[provider]
            for r, line in enumerate(content[:pane_height-2]):
                self.move_cursor(r + 2, col_start + 2); print(line[:pane_width-4])

        main_pane_row_start = pane_height + 1
        self._draw_main_conversation_pane(term_width, term_height, main_pane_row_start, conversation_log)

    def _draw_focused_layout(self, term_width, term_height, conversation_log):
        self._draw_main_conversation_pane(term_width, term_height, 1, conversation_log)

    def _draw_main_conversation_pane(self, term_width, term_height, start_row, conversation_log):
        self.move_cursor(start_row, 1)
        title = " MAIN CONVERSATION "
        print(f"{Colors.BRIGHT_BLACK}┌" + title.center(term_width - 2, "─") + f"┐{Colors.RESET}")
        
        conversation_height = term_height - start_row - 2
        for r in range(conversation_height + 1):
            self.move_cursor(start_row + r + 1, 1)
            print(f"{Colors.BRIGHT_BLACK}│{' ' * (term_width - 2)}│{Colors.RESET}")
        
        self.move_cursor(term_height -1, 1)
        print(f"{Colors.BRIGHT_BLACK}└" + "─" * (term_width - 2) + f"┘{Colors.RESET}")
        
        if conversation_height > 0:
            display_lines = conversation_log[-conversation_height:]
            for r, line in enumerate(display_lines):
                self.move_cursor(start_row + r + 1, 3); print(line)

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
        self.ui = TerminalUI(list(self.interfaces.keys()))
        self.conversation_log = []
        self.stats = self._initialize_stats()
    
    def _initialize_stats(self) -> Dict:
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
        self.ui.draw_layout(self.stats, self.conversation_log)
        
        # ** FIX: Sub-agents should never be given tools. **
        system_prompt_for_tool = "You are a helpful assistant. Please answer the user's prompt directly and concisely. Do not use any tools."
        
        response = client.chat(messages=[{"role": "user", "content": prompt}], system_prompt=system_prompt_for_tool, tools=[])
        
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
        self.log_to_display(f"{Colors.BOLD}{Colors.GREEN}You: {Colors.RESET} {user_prompt}")
        self.conversation_history.append({"role": "user", "content": user_prompt})
        
        other_tools = [f"`query_{p.value}`" for p in self.interfaces if p != self.primary_agent_name]
        self_tool = f"`query_{self.primary_agent_name.value}`"
        system_prompt = (f"You are a sophisticated AI agent, '{self.primary_agent_name.value}'. Your goal is to provide comprehensive answers. "
                         f"You can use tools to query other AI models: {', '.join(other_tools)}. "
                         f"You can also call your own model using the {self_tool} tool to summarize, reflect, or re-evaluate your own thoughts. "
                         "Synthesize information from all sources into a final, conclusive answer for the user.")

        current_messages = self.conversation_history.copy()
        
        while True: 
            self.log_to_display(f"{Colors.BOLD}{Colors.CYAN}🤖 {self.primary_agent_name.value.capitalize()}: {Colors.RESET}{Colors.BRIGHT_BLACK}[thinking...]{Colors.RESET}")
            self.ui.draw_layout(self.stats, self.conversation_log)
            response = self.primary_interface.chat(messages=current_messages, system_prompt=system_prompt, tools=self._generate_tools_schema())

            if not response["success"]: self.log_to_display(f"ERROR: {response['error']}", error=True); break
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

            if interim_text: self.log_to_display(f"{Colors.BOLD}{Colors.CYAN}🤖 {self.primary_agent_name.value.capitalize()}: {Colors.RESET}{interim_text}")

            if not tool_calls: break

            self.conversation_history.append({"role": "assistant", "content": response_obj.model_dump_json() if hasattr(response_obj, 'model_dump_json') else str(response_obj)})
            
            tool_results = []
            for tool_call in tool_calls:
                result = self._execute_tool_call(tool_call['name'], tool_call['input'])
                tool_result_content = result['content'] if result['success'] else f"Error: {result['error']}"
                if self.primary_agent_name == Provider.ANTHROPIC: tool_results.append({"type": "tool_result", "tool_use_id": tool_call['id'], "content": tool_result_content})
                else: tool_results.append({"role": "tool", "tool_call_id": tool_call['id'], "name": tool_call['name'], "content": tool_result_content})
            
            current_messages = self.conversation_history.copy()
            if self.primary_agent_name == Provider.ANTHROPIC: current_messages.append({"role": "user", "content": tool_results})
            else: current_messages.extend(tool_results)

    def log_to_display(self, message: str, error: bool = False):
        term_width, term_height = self.ui.get_terminal_size()
        main_pane_width = term_width - 4
        prefix = f"{Colors.BOLD}{Colors.RED}ERROR: {Colors.RESET}" if error else ""
        
        all_wrapped_lines = []
        for line in message.split('\n'):
            wrapped_lines = textwrap.wrap(prefix + line, width=main_pane_width, subsequent_indent="  " if not prefix else "    ")
            all_wrapped_lines.extend(wrapped_lines or [""])
            prefix = ""

        if self.conversation_log and "[thinking...]" in self.conversation_log[-1]: self.conversation_log.pop()
        self.conversation_log.extend(all_wrapped_lines)
        
        max_log_lines = term_height * 2
        if len(self.conversation_log) > max_log_lines: self.conversation_log = self.conversation_log[-max_log_lines:]

    def run(self):
        self.ui.clear_screen()
        while True:
            self.ui.draw_layout(self.stats, self.conversation_log)
            try:
                user_input = input().strip()
                if not user_input: continue
                
                command = user_input.lower()
                if command in ['exit', 'quit', 'q']: self.ui.clear_screen(); break
                elif command == 'clear': self.conversation_log = []; self.conversation_history = []
                elif command == 'toggle-panes': config.SHOW_SUB_AGENT_PANES = not config.SHOW_SUB_AGENT_PANES
                elif command == 'stats': self.print_stats_summary()
                elif command == 'help': self.print_help()
                else: self.chat(user_input)
            except (KeyboardInterrupt, EOFError): self.ui.clear_screen(); break
    
    def print_stats_summary(self):
        self.log_to_display(f"{Colors.BOLD}{Colors.YELLOW}--- Session Statistics ---{Colors.RESET}")
        for p_value, s in self.stats.items():
            if p_value in ["session_start", "primary_agent"]: continue
            cost_str = f"${s['cost']:.5f}"
            if s['cost'] == 0 and s['input_tokens'] > 0: cost_str += " (Price Unknown)"
            self.log_to_display(f"{Colors.YELLOW}{p_value.capitalize()}:{Colors.RESET} Model: {s['model']}, Tokens: {s['input_tokens'] + s['output_tokens']:,}, Cost: {cost_str}")

    def print_help(self):
        self.log_to_display(f"{Colors.BOLD}{Colors.YELLOW}--- Help ---{Colors.RESET}")
        self.log_to_display(f"{Colors.YELLOW}toggle-panes:{Colors.RESET} Show/hide the top sub-agent panes.")
        self.log_to_display(f"{Colors.YELLOW}stats:{Colors.RESET}         Print a detailed summary of token usage and costs.")
        self.log_to_display(f"{Colors.YELLOW}clear:{Colors.RESET}         Clear the main conversation screen.")
        self.log_to_display(f"{Colors.YELLOW}exit:{Colors.RESET}          Exit the application.")

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
    model_selections: Optional[Dict[Provider, str]] = None
    
    saved_config = load_configuration()
    if saved_config:
        print("\n--- Found Saved Configuration ---")
        print(f"Primary Agent: {saved_config.get('primary_agent')}")
        for p_val, m in saved_config.get('model_selections', {}).items():
            print(f"  - {p_val.capitalize()}: {m}")
        
        if input("\nUse this configuration? (y/n): ").lower() == 'y':
            primary_agent = Provider(saved_config.get('primary_agent'))
            model_selections = {Provider(k): v for k,v in saved_config.get('model_selections', {}).items()}

    if not primary_agent or not model_selections:
        interfaces = {p: globals()[f"{p.name.capitalize()}Interface"](api_keys[p.value], "") for p in PROVIDERS if api_keys[p.value]}
        print("Fetching available models...")
        available_models = {p: i.list_models() for p, i in interfaces.items()}
        
        print("\n--- AGENT SETUP ---")
        primary_agent = select_provider_from_list("\n1. Select the Primary Agent (Provider):", list(interfaces.keys()))

        model_selections = {}
        print("\n2. Select the specific model for each provider:")
        for provider in interfaces.keys():
            model_selections[provider] = select_model_from_grouped_list(f"  - {provider.value.capitalize()} Model:", available_models[provider], provider)

    logger.info(f"Primary Agent: {primary_agent.value}, Models: { {p.value: m for p,m in model_selections.items()} }")
    print("\n🚀 Starting Multi-Pane Interface...")
    time.sleep(1)

    str_model_selections = {p.value: m for p, m in model_selections.items()}
    bridge = MultiLLMAgentBridge(api_keys, primary_agent, str_model_selections)
    
    try:
        bridge.run()
    finally:
        save_configuration(primary_agent, model_selections)
        print("\n✅ Configuration saved. Exiting.")

if __name__ == "__main__":
    main()
