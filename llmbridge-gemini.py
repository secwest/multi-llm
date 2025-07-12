#!/usr/bin/env python3
"""
Multi-LLM Agent Bridge - Multi-Pane Version
Version: 4.2.0 (2025-07-11)

A primary LLM (Claude, OpenAI, or Gemini) queries the other two as tools,
displaying all interactions in a real-time, multi-pane terminal interface.

Changelog:
- v4.2.0: Increased sub-agent pane height and fixed multi-line response rendering.
- v4.1.x: Bug fixes for API key loading, client initialization, and response parsing.
- v4.0.0: Merged features from user-provided scripts, including multi-window UI,
          color coding, and enhanced commands.
"""
import os
import sys
import json
import textwrap
from typing import Dict, List, Optional, Any, Tuple, Literal
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai
import logging
from pathlib import Path
from datetime import datetime
import time
import shutil
import subprocess
import platform
import signal

# --- CONFIGURATION ---
VERSION = "4.2.0"

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

Provider = Literal["anthropic", "openai", "gemini"]
PROVIDERS: List[Provider] = ["anthropic", "openai", "gemini"]

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
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path, verbose=True)
            print(f"📄 Loading API keys from: {dotenv_path}")
        else:
            print(f"⚠️ .env file not found at {dotenv_path}. Relying on system environment variables.")
    except ImportError:
        print("⚠️ python-dotenv not installed. Relying on system environment variables.")

    return {
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
    }

def setup_env_file():
    print("\n🔧 API Key Setup: Some keys are missing.")
    if input("Create a .env file? (y/n): ").lower() != 'y': return False
    with open(".env", "w") as f:
        f.write("# API Keys for Multi-LLM Agent Bridge\n")
        f.write(f"ANTHROPIC_API_KEY={input('Enter your Anthropic API key: ').strip() or 'your-key-here'}\n")
        f.write(f"OPENAI_API_KEY={input('Enter your OpenAI API key: ').strip() or 'your-key-here'}\n")
        f.write(f"GEMINI_API_KEY={input('Enter your Gemini API key: ').strip() or 'your-key-here'}\n")
    print("✅ .env file created.")
    if not Path(".gitignore").exists():
        with open(".gitignore", "w") as f: f.write(".env\n.logs/\n")
        print("✅ Created .gitignore to protect secrets.")
    return True

class TerminalUI:
    def __init__(self):
        self.last_terminal_size = (0, 0)
        self.provider_pane_content: Dict[Provider, List[str]] = {p: ["No activity yet."] for p in PROVIDERS}
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
        pane_width = self.get_terminal_size()[0] // 3 - 4
        q_wrapped = textwrap.wrap(f"Q: {query}", width=pane_width)
        r_wrapped = textwrap.wrap(f"A: {response}", width=pane_width)
        self.provider_pane_content[provider] = q_wrapped + ["-" * pane_width] + r_wrapped

    def draw_layout(self, stats: Dict[str, Any], conversation_log: List[str]):
        term_width, term_height = self.get_terminal_size()
        if (term_width, term_height) != self.last_terminal_size:
            self.clear_screen()
            self.last_terminal_size = (term_width, term_height)

        if config.SHOW_SUB_AGENT_PANES:
            self._draw_multi_pane_layout(term_width, term_height, stats, conversation_log)
        else:
            self._draw_focused_layout(term_width, term_height, stats, conversation_log)

        self.move_cursor(term_height, 1)
        print(f"{Colors.BOLD}{Colors.GREEN}You: {Colors.RESET}", end='')
        sys.stdout.flush()

    def _draw_multi_pane_layout(self, term_width, term_height, stats, conversation_log):
        # ** CHANGE: Increased pane height **
        pane_height = 14
        pane_width = term_width // 3
        for i, provider in enumerate(PROVIDERS):
            col_start = i * pane_width + 1
            title = f" {provider.upper()} "
            if stats.get('primary_agent') == provider: title += "(AGENT) "
            self.move_cursor(1, col_start); print(f"{Colors.BRIGHT_BLACK}┌" + title.center(pane_width - 2, "─") + f"┐{Colors.RESET}")
            for r in range(2, pane_height):
                self.move_cursor(r, col_start); print(f"{Colors.BRIGHT_BLACK}│{' ' * (pane_width - 2)}│{Colors.RESET}")
            self.move_cursor(pane_height, col_start); print(f"{Colors.BRIGHT_BLACK}└" + "─" * (pane_width - 2) + f"┘{Colors.RESET}")
            content = self.provider_pane_content[provider]
            for r, line in enumerate(content[:pane_height-2]):
                self.move_cursor(r + 2, col_start + 2); print(line)

        main_pane_row_start = pane_height + 1
        self._draw_main_conversation_pane(term_width, term_height, main_pane_row_start, conversation_log)

    def _draw_focused_layout(self, term_width, term_height, stats, conversation_log):
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

class BaseAPIClient:
    def __init__(self, api_key: str, provider_name: Provider):
        if not api_key: raise ValueError(f"{provider_name.capitalize()} API key is missing.")
        self.provider_name = provider_name
    def chat_completion(self, **kwargs) -> Dict[str, Any]: raise NotImplementedError
    def list_models(self) -> List[str]: raise NotImplementedError
    def _calculate_cost(self, model: str, i_tok: int, o_tok: int) -> float:
        p = PRICE_MAPPING.get(model)
        if not p: return 0.0
        return ((i_tok / 1_000_000) * p.get("input", 0.0)) + ((o_tok / 1_000_000) * p.get("output", 0.0))

class AnthropicClient(BaseAPIClient):
    def __init__(self, api_key: str): super().__init__(api_key, "anthropic"); self.client = Anthropic(api_key=api_key)
    def list_models(self) -> List[str]: return sorted([m for m in PRICE_MAPPING if "claude" in m], reverse=True)
    def chat_completion(self, model: str, messages: List[Dict], system: str, tools: List) -> Dict:
        try:
            r = self.client.messages.create(model=model, max_tokens=4096, messages=messages, system=system, tools=tools)
            cost = self._calculate_cost(model, r.usage.input_tokens, r.usage.output_tokens)
            return {"success": True, "content": r.content, "usage": r.usage.to_dict(), "cost": cost, "model": model}
        except Exception as e: return {"success": False, "error": str(e)}

class OpenAIClient(BaseAPIClient):
    def __init__(self, api_key: str): super().__init__(api_key, "openai"); self.client = OpenAI(api_key=api_key)
    def list_models(self) -> List[str]:
        try: return sorted([m.id for m in self.client.models.list() if "gpt" in m.id or "o1" in m.id], reverse=True)
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {e}"); return sorted([m for m in PRICE_MAPPING if "gpt" in m], reverse=True)
    def chat_completion(self, model: str, messages: List[Dict], system: str, tools: List) -> Dict:
        all_msg = [{"role": "system", "content": system}] + messages
        try:
            r = self.client.chat.completions.create(model=model, messages=all_msg, tools=tools, tool_choice="auto" if tools else None, max_tokens=4096)
            cost = self._calculate_cost(model, r.usage.prompt_tokens, r.usage.completion_tokens)
            content = []
            if r.choices[0].message.content: content.append({"type": "text", "text": r.choices[0].message.content})
            if r.choices[0].message.tool_calls:
                for tc in r.choices[0].message.tool_calls:
                    content.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": json.loads(tc.function.arguments)})
            return {"success": True, "content": content, "usage": r.usage.to_dict(), "cost": cost, "model": model}
        except Exception as e: return {"success": False, "error": str(e)}

class GeminiClient(BaseAPIClient):
    def __init__(self, api_key: str): super().__init__(api_key, "gemini"); genai.configure(api_key=api_key)
    def list_models(self) -> List[str]:
        try:
            models = []
            excluded = ['-tts', '-embedding', '-aqa']
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods and not any(m.name.endswith(s) for s in excluded):
                    models.append(m.name[len('models/'):])
            return sorted(models, reverse=True)
        except Exception as e:
            logger.error(f"Failed to fetch Gemini models: {e}"); return sorted([m for m in PRICE_MAPPING if "gemini" in m], reverse=True)
    def chat_completion(self, model: str, messages: List[Dict], system: str, tools: List) -> Dict:
        gemini_tools = [{"function_declarations": [t['function'] for t in tools]}] if tools else None
        try:
            gemini_model = genai.GenerativeModel(model, system_instruction=system, tools=gemini_tools, generation_config=genai.GenerationConfig(max_output_tokens=8192))
            history = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                content = msg.get('content')
                if isinstance(content, list):
                    text = " ".join([c.get('text', '') for c in content if c.get('type') == 'text'])
                    history.append({'role': role, 'parts': [text]})
                else: history.append({'role': role, 'parts': [content]})
            latest_prompt = history.pop()['parts'] if history else []
            chat = gemini_model.start_chat(history=history)
            r = chat.send_message(latest_prompt, stream=False)
            
            s_content, o_text, o_tok = [], "", 0
            if r.candidates and r.candidates[0].content.parts:
                for part in r.candidates[0].content.parts:
                    if part.function_call: s_content.append({"type": "tool_use", "id": part.function_call.name, "name": part.function_call.name, "input": dict(part.function_call.args)})
                    elif hasattr(part, 'text'): o_text += part.text
            if o_text:
                s_content.append({"type": "text", "text": o_text})
                o_tok = gemini_model.count_tokens(o_text).total_tokens
            i_tok = gemini_model.count_tokens(chat.history).total_tokens
            cost = self._calculate_cost(model, i_tok, o_tok)
            return {"success": True, "content": s_content, "usage": {"input_tokens": i_tok, "output_tokens": o_tok}, "cost": cost, "model": model}
        except Exception as e: return {"success": False, "error": str(e)}

class MultiLLMAgentBridge:
    def __init__(self, api_keys: Dict[str, str], primary_agent: Provider, model_selections: Dict[str, str]):
        self.primary_agent_name = primary_agent
        self.model_selections = model_selections
        self.conversation_history = []
        self.ui = TerminalUI()
        self.conversation_log = []
        self.clients: Dict[Provider, BaseAPIClient] = {}
        for p in PROVIDERS:
            class_name = "OpenAIClient" if p == "openai" else f"{p.capitalize()}Client"
            self.clients[p] = globals()[class_name](api_keys[p])
        self.primary_client = self.clients[primary_agent]
        self.stats = self._initialize_stats()
    
    def _initialize_stats(self) -> Dict:
        stats = {p: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "model": self.model_selections[p]} for p in PROVIDERS}
        stats["session_start"] = datetime.now()
        stats["primary_agent"] = self.primary_agent_name
        return stats

    def _generate_tools_schema(self) -> List[Dict]:
        tools = []
        for provider in PROVIDERS:
            if provider == self.primary_agent_name: continue
            tool_name = f"query_{provider}"
            desc = f"Query the {provider.capitalize()} API to get its perspective or capabilities."
            props = {"prompt": {"type": "string", "description": "A clear, self-contained prompt."}}
            if self.primary_agent_name == "anthropic":
                tools.append({"name": tool_name, "description": desc, "input_schema": {"type": "object", "properties": props, "required": ["prompt"]}})
            elif self.primary_agent_name in ["openai", "gemini"]:
                 tools.append({"type": "function", "function": {"name": tool_name, "description": desc, "parameters": {"type": "object", "properties": props, "required": ["prompt"]}}})
        return tools

    def _update_stats(self, provider: Provider, usage: Dict, cost: float):
        self.stats[provider]['input_tokens'] += usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
        self.stats[provider]['output_tokens'] += usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
        self.stats[provider]['cost'] += cost

    def _execute_tool_call(self, tool_name: str, tool_input: Dict) -> Dict:
        target_provider: Provider = tool_name.split("_")[1] # type: ignore
        client = self.clients[target_provider]
        prompt = tool_input.get("prompt", "")
        self.ui.update_provider_pane(target_provider, prompt, "[Waiting for response...]")
        self.ui.draw_layout(self.stats, self.conversation_log)
        response = client.chat_completion(model=self.model_selections[target_provider], messages=[{"role": "user", "content": prompt}], system="You are a helpful assistant.", tools=[])
        
        if response["success"]:
            self._update_stats(target_provider, response['usage'], response['cost'])
            
            text_content = ""
            if response['content']:
                if isinstance(response['content'][0], str):
                     text_content = " ".join(response['content'])
                else:
                    text_content = " ".join([c.text if hasattr(c, 'text') else c.get('text', '') for c in response['content'] if (hasattr(c, 'text') or (isinstance(c, dict) and c.get('type') == 'text'))])
            
            self.ui.update_provider_pane(target_provider, prompt, text_content)
            return {"success": True, "content": text_content}
        else:
            self.ui.update_provider_pane(target_provider, prompt, f"ERROR: {response['error']}")
            return {"success": False, "error": response['error']}
    
    def chat(self, user_prompt: str):
        self.log_to_display(f"{Colors.BOLD}{Colors.GREEN}You: {Colors.RESET}{user_prompt}")
        self.conversation_history.append({"role": "user", "content": user_prompt})
        
        other_tools = [f"`query_{p}`" for p in PROVIDERS if p != self.primary_agent_name]
        system_prompt = (f"You are a sophisticated AI agent, '{self.primary_agent_name}'. Your goal is to provide comprehensive answers. "
                         f"Use these tools to query other AI models: {', '.join(other_tools)}. "
                         "Use them to gather perspectives, verify facts, or delegate sub-tasks. "
                         "Synthesize the information into a final, conclusive answer for the user.")

        current_messages = self.conversation_history.copy()
        
        while True: 
            self.log_to_display(f"{Colors.BOLD}{Colors.CYAN}🤖 {self.primary_agent_name.capitalize()}: {Colors.RESET}{Colors.BRIGHT_BLACK}[thinking...]{Colors.RESET}")
            self.ui.draw_layout(self.stats, self.conversation_log)
            response = self.primary_client.chat_completion(model=self.model_selections[self.primary_agent_name], messages=current_messages, system=system_prompt, tools=self._generate_tools_schema())

            if not response["success"]: self.log_to_display(f"ERROR: {response['error']}", error=True); break
            self._update_stats(self.primary_agent_name, response['usage'], response['cost'])
            
            assistant_message = {"role": "assistant", "content": response['content']}
            self.conversation_history.append(assistant_message)
            current_messages.append(assistant_message)
            
            has_tool_call = any(c.get('type') == 'tool_use' for c in response['content'] if isinstance(c, dict))
            
            interim_text = ""
            if response['content']:
                 if isinstance(response['content'][0], str):
                      interim_text = " ".join(response['content'])
                 else:
                      interim_text = " ".join([c.text if hasattr(c, 'text') else c.get('text', '') for c in response['content'] if (hasattr(c, 'text') or (isinstance(c, dict) and c.get('type') == 'text'))])

            if interim_text: self.log_to_display(f"{Colors.BOLD}{Colors.CYAN}🤖 {self.primary_agent_name.capitalize()}: {Colors.RESET}{interim_text}")

            if not has_tool_call: break

            tool_results = []
            for content_part in response['content']:
                if isinstance(content_part, dict) and content_part.get('type') == 'tool_use':
                    tool_name, tool_input, tool_id = content_part['name'], content_part['input'], content_part['id']
                    result = self._execute_tool_call(tool_name, tool_input)
                    tool_result_content = result['content'] if result['success'] else f"Error: {result['error']}"
                    if self.primary_agent_name == "anthropic": tool_results.append({"type": "tool_result", "tool_use_id": tool_id, "content": tool_result_content})
                    else: tool_results.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": tool_result_content})
            
            if self.primary_agent_name == "anthropic": current_messages.append({"role": "user", "content": tool_results})
            else: current_messages.extend(tool_results)
            self.conversation_history = current_messages

    def log_to_display(self, message: str, error: bool = False):
        term_width, term_height = self.ui.get_terminal_size()
        main_pane_width = term_width - 4
        prefix = f"{Colors.BOLD}{Colors.RED}ERROR: {Colors.RESET}" if error else ""
        
        # ** FIX: Split message by newline before wrapping **
        all_wrapped_lines = []
        for line in message.split('\n'):
            wrapped_lines = textwrap.wrap(prefix + line, width=main_pane_width, subsequent_indent="  " if not prefix else "    ")
            all_wrapped_lines.extend(wrapped_lines or [""]) # Add empty line for blank lines
            prefix = "" # Prefix only applies to the first line of the message

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
        for p in PROVIDERS:
            s = self.stats[p]
            cost_str = f"${s['cost']:.5f}"
            if s['cost'] == 0 and s['input_tokens'] > 0: cost_str += " (Price Unknown)"
            self.log_to_display(f"{Colors.YELLOW}{p.capitalize()}:{Colors.RESET} Model: {s['model']}, Tokens: {s['input_tokens'] + s['output_tokens']:,}, Cost: {cost_str}")

    def print_help(self):
        self.log_to_display(f"{Colors.BOLD}{Colors.YELLOW}--- Help ---{Colors.RESET}")
        self.log_to_display(f"{Colors.YELLOW}toggle-panes:{Colors.RESET} Show/hide the top 3 sub-agent panes.")
        self.log_to_display(f"{Colors.YELLOW}stats:{Colors.RESET}         Print a detailed summary of token usage and costs.")
        self.log_to_display(f"{Colors.YELLOW}clear:{Colors.RESET}         Clear the main conversation screen.")
        self.log_to_display(f"{Colors.YELLOW}exit:{Colors.RESET}          Exit the application.")

def select_model_from_grouped_list(prompt: str, all_models: List[str], provider: str) -> str:
    print(prompt)
    model_groups = {
        "GPT-4o Series": [m for m in all_models if "gpt-4o" in m], "GPT-4 Series": [m for m in all_models if m.startswith("gpt-4") and "o" not in m],
        "O1 Series (Reasoning)": [m for m in all_models if m.startswith("o1")], "GPT-3.5 Series": [m for m in all_models if "gpt-3.5" in m],
        "Claude 3.5 Series": [m for m in all_models if "claude-3-5" in m], "Claude 3 Series": [m for m in all_models if m.startswith("claude-3-") and "5" not in m],
        "Gemini 1.5 Series": [m for m in all_models if "gemini-1.5" in m], "Gemini 1.0 Series": [m for m in all_models if "gemini-1.0" in m],
    }
    numbered_models = []
    provider_keys = [provider]
    if provider == "openai": provider_keys.extend(["gpt", "o1"])
    if provider == "all": provider_keys.extend(PROVIDERS)
    
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

def main():
    print(f"🚀 Initializing Multi-LLM Agent Bridge v{VERSION}...")
    if not check_dependencies(): sys.exit(1)
    api_keys = load_api_keys()
    if any(not k for k in api_keys.values()):
        if not setup_env_file(): print("❌ API keys are required. Exiting."); sys.exit(1)
        api_keys = load_api_keys()
        if any(not k for k in api_keys.values()): print("❌ Still missing API keys. Exiting."); sys.exit(1)

    print("✅ All API keys loaded.")
    clients = {}
    for p in PROVIDERS:
        class_name = "OpenAIClient" if p == "openai" else f"{p.capitalize()}Client"
        clients[p] = globals()[class_name](api_keys[p])

    print("Fetching available models...")
    available_models = {p: c.list_models() for p, c in clients.items()}
    
    print("\n--- AGENT SETUP ---")
    primary_agent = select_model_from_grouped_list("\n1. Select the Primary Agent (Provider):", PROVIDERS, "all")

    model_selections = {}
    print("\n2. Select the specific model for each provider:")
    for provider in PROVIDERS:
        model_selections[provider] = select_model_from_grouped_list(f"  - {provider.capitalize()} Model:", available_models[provider], provider)

    logger.info(f"Primary Agent: {primary_agent}, Models: {model_selections}")
    print("\n🚀 Starting Multi-Pane Interface...")
    time.sleep(1)

    bridge = MultiLLMAgentBridge(api_keys, primary_agent, model_selections)
    bridge.run()

if __name__ == "__main__":
    main()
