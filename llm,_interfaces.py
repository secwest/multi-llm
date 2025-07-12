#!/usr/bin/env python3
"""
LLM Interfaces Module for Multi-LLM Bridge
Version: 5.0.1

This module contains all LLM API interfaces and related functionality.
Fixed: Gemini implementation based on working code
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai

# Get logger from main module
logger = logging.getLogger("LLM_Bridge")

# Detailed pricing information (per million tokens)
PRICE_MAPPING = {
    # Claude models
    "claude-opus-4-20250514": {"input": 20.00, "output": 100.00},
    "claude-sonnet-4-20250514": {"input": 4.00, "output": 20.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    
    # OpenAI models
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4-0125-preview": {"input": 10.00, "output": 30.00},
    "gpt-4-1106-preview": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-0613": {"input": 30.00, "output": 60.00},
    "gpt-4-32k": {"input": 60.00, "output": 120.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
    "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o1-preview-2024-09-12": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-mini-2024-09-12": {"input": 3.00, "output": 12.00},
    
    # Gemini models
    "gemini-2.0-flash-exp": {"input": 0.00, "output": 0.00},  # Experimental, free
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-1.5-pro-latest": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash": {"input": 0.35, "output": 1.05},
    "gemini-1.5-flash-latest": {"input": 0.35, "output": 1.05},
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    "gemini-pro": {"input": 0.50, "output": 1.50},
    "gemini-ultra": {"input": 7.00, "output": 21.00},  # Estimated
}

# Model configurations with descriptions
CLAUDE_MODELS = [
    ("claude-opus-4-20250514", "Claude Opus 4 - Most capable model for complex tasks"),
    ("claude-sonnet-4-20250514", "Claude Sonnet 4 - Balanced performance and cost"),
    ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet - Latest balanced model"),
    ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku - Fast, cost-effective"),
    ("claude-3-opus-20240229", "Claude 3 Opus (Legacy) - Previous flagship"),
    ("claude-3-sonnet-20240229", "Claude 3 Sonnet (Legacy) - Previous balanced"),
    ("claude-3-haiku-20240307", "Claude 3 Haiku (Legacy) - Previous fast model")
]

GEMINI_MODELS = [
    ("gemini-2.0-flash-exp", "Gemini 2.0 Flash - Fast, experimental (free)"),
    ("gemini-1.5-pro", "Gemini 1.5 Pro - Advanced reasoning, long context"),
    ("gemini-1.5-flash", "Gemini 1.5 Flash - Fast and versatile"),
    ("gemini-1.0-pro", "Gemini 1.0 Pro - Balanced performance"),
    ("gemini-ultra", "Gemini Ultra - Most capable (when available)")
]


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send a chat request to the LLM"""
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate the cost of the API call"""
        pass
    
    @abstractmethod
    def get_model_list(self) -> List[str]:
        """Get list of available models"""
        pass
    
    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if this LLM supports tool/function calling"""
        pass
    
    @abstractmethod
    def format_tool_response(self, tool_call_id: str, tool_result: Any) -> Dict[str, Any]:
        """Format tool response for this LLM's expected format"""
        pass


class ClaudeInterface(LLMInterface):
    """Interface for Claude API with comprehensive error handling"""
    
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("Claude API key is required")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Claude interface with model: {model}")
        
    def chat(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, 
             tools: Optional[List[Dict]] = None, **kwargs) -> Dict[str, Any]:
        """Send a chat request to Claude"""
        try:
            # Claude requires system prompt to be separate
            filtered_messages = []
            combined_system = ""
            
            for msg in messages:
                if msg['role'] == 'system':
                    if combined_system:
                        combined_system += "\n\n" + msg['content']
                    else:
                        combined_system = msg['content']
                else:
                    # Handle complex message content
                    if isinstance(msg.get('content'), list):
                        # Already in Claude's expected format
                        filtered_messages.append(msg)
                    else:
                        # Convert simple string to expected format
                        filtered_messages.append({
                            'role': msg['role'],
                            'content': msg['content']
                        })
            
            # Merge system prompts
            if system_prompt:
                if combined_system:
                    combined_system = system_prompt + "\n\n" + combined_system
                else:
                    combined_system = system_prompt
            
            # Build request parameters
            create_kwargs = {
                'model': self.model,
                'max_tokens': kwargs.get('max_tokens', 4096),
                'messages': filtered_messages,
                'temperature': kwargs.get('temperature', 0.7)
            }
            
            if combined_system:
                create_kwargs['system'] = combined_system
            if tools:
                create_kwargs['tools'] = tools
            if 'stop_sequences' in kwargs:
                create_kwargs['stop_sequences'] = kwargs['stop_sequences']
                
            logger.debug(f"Claude request - Model: {self.model}, Messages: {len(filtered_messages)}, Tools: {len(tools) if tools else 0}")
            response = self.client.messages.create(**create_kwargs)
            
            # Extract content - handle both TextBlock objects and dicts
            content = []
            tool_calls = []
            
            for block in response.content:
                if hasattr(block, 'type'):
                    # Handle Anthropic SDK objects
                    if block.type == "text":
                        content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        tool_calls.append({
                            'id': block.id,
                            'name': block.name,
                            'input': block.input
                        })
                elif isinstance(block, dict):
                    # Handle dict responses
                    if block.get('type') == 'text':
                        content.append(block)
                    elif block.get('type') == 'tool_use':
                        tool_calls.append({
                            'id': block['id'],
                            'name': block['name'],
                            'input': block['input']
                        })
                else:
                    # Fallback for unexpected types
                    logger.warning(f"Unexpected content block type: {type(block)}")
                    if hasattr(block, 'text'):
                        content.append({"type": "text", "text": str(block.text)})
            
            result = {
                'success': True,
                'content': content,
                'tool_calls': tool_calls,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                },
                'raw_response': response,
                'model': self.model,
                'stop_reason': getattr(response, 'stop_reason', None)
            }
            
            logger.debug(f"Claude response - Tokens: {result['usage']['total_tokens']}, Tool calls: {len(tool_calls)}")
            return result
            
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}", exc_info=True)
            return {
                'success': False, 
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def handle_tool_result(self, messages: List[Dict], tool_use_id: str, tool_result: Any,
                          system_prompt: Optional[str] = None, tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Handle tool result and get Claude's response"""
        # Format tool result for Claude
        tool_result_content = tool_result if isinstance(tool_result, str) else json.dumps(tool_result)
        
        # Add tool result to messages
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": tool_result_content
            }]
        })
        
        # Get Claude's response after tool use
        return self.chat(messages, system_prompt=system_prompt, tools=tools)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for Claude usage"""
        pricing = PRICE_MAPPING.get(model, {"input": 15.0, "output": 75.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = round(input_cost + output_cost, 6)
        
        logger.debug(f"Claude cost calculation - Model: {model}, Input: ${input_cost:.6f}, Output: ${output_cost:.6f}, Total: ${total_cost:.6f}")
        return total_cost
    
    def get_model_list(self) -> List[str]:
        """Get list of available Claude models"""
        models = sorted([m for m in PRICE_MAPPING if "claude" in m], reverse=True)
        return models
    
    def supports_tools(self) -> bool:
        """Claude supports tool/function calling"""
        return True
    
    def format_tool_response(self, tool_call_id: str, tool_result: Any) -> Dict[str, Any]:
        """Format tool response for Claude"""
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
        }


class OpenAIInterface(LLMInterface):
    """Interface for OpenAI API with comprehensive features"""
    
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI interface with model: {model}")
        
    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs) -> Dict[str, Any]:
        """Send a chat request to OpenAI"""
        try:
            # Prepare messages - ensure all have proper format
            formatted_messages = []
            for msg in messages:
                if isinstance(msg.get('content'), list):
                    # Handle complex content (e.g., from tool results)
                    formatted_messages.append(msg)
                else:
                    formatted_messages.append({
                        'role': msg['role'],
                        'content': msg.get('content', '')
                    })
            
            create_kwargs = {
                'model': self.model,
                'messages': formatted_messages,
                'temperature': kwargs.get('temperature', 0.7)
            }
            
            # Add optional parameters
            if 'max_tokens' in kwargs:
                create_kwargs['max_tokens'] = kwargs['max_tokens']
            if 'top_p' in kwargs:
                create_kwargs['top_p'] = kwargs['top_p']
            if 'frequency_penalty' in kwargs:
                create_kwargs['frequency_penalty'] = kwargs['frequency_penalty']
            if 'presence_penalty' in kwargs:
                create_kwargs['presence_penalty'] = kwargs['presence_penalty']
            if 'stop' in kwargs:
                create_kwargs['stop'] = kwargs['stop']
            if 'response_format' in kwargs:
                create_kwargs['response_format'] = kwargs['response_format']
                
            if tools:
                # Convert tools to OpenAI format
                openai_tools = []
                for tool in tools:
                    openai_tools.append({
                        'type': 'function',
                        'function': {
                            'name': tool['name'],
                            'description': tool['description'],
                            'parameters': tool['input_schema']
                        }
                    })
                create_kwargs['tools'] = openai_tools
                create_kwargs['tool_choice'] = kwargs.get('tool_choice', 'auto')
            
            logger.debug(f"OpenAI request - Model: {self.model}, Messages: {len(formatted_messages)}, Tools: {len(tools) if tools else 0}")
            response = self.client.chat.completions.create(**create_kwargs)
            
            # Extract content and tool calls
            content = []
            tool_calls = []
            
            message = response.choices[0].message
            if message.content:
                content.append({"type": "text", "text": message.content})
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        'id': tc.id,
                        'name': tc.function.name,
                        'input': json.loads(tc.function.arguments)
                    })
            
            result = {
                'success': True,
                'content': content,
                'tool_calls': tool_calls,
                'usage': {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'raw_response': response,
                'model': self.model,
                'finish_reason': response.choices[0].finish_reason
            }
            
            logger.debug(f"OpenAI response - Tokens: {result['usage']['total_tokens']}, Tool calls: {len(tool_calls)}")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}", exc_info=True)
            return {
                'success': False, 
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def handle_tool_result(self, messages: List[Dict], tool_call_id: str, tool_result: Any,
                          tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Handle tool result and get OpenAI's response"""
        # Add tool result message
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
        }
        messages.append(tool_message)
        
        # Get OpenAI's response after tool use
        return self.chat(messages, tools=tools)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for OpenAI usage"""
        pricing = PRICE_MAPPING.get(model)
        if not pricing:
            # Default pricing based on model family
            if 'gpt-4o' in model:
                pricing = {"input": 5.0, "output": 15.0}
            elif 'gpt-4' in model:
                if '32k' in model:
                    pricing = {"input": 60.0, "output": 120.0}
                elif 'turbo' in model:
                    pricing = {"input": 10.0, "output": 30.0}
                else:
                    pricing = {"input": 30.0, "output": 60.0}
            elif 'o1' in model:
                if 'mini' in model:
                    pricing = {"input": 3.0, "output": 12.0}
                else:
                    pricing = {"input": 15.0, "output": 60.0}
            else:
                pricing = {"input": 0.5, "output": 1.5}
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = round(input_cost + output_cost, 6)
        
        logger.debug(f"OpenAI cost calculation - Model: {model}, Input: ${input_cost:.6f}, Output: ${output_cost:.6f}, Total: ${total_cost:.6f}")
        return total_cost
    
    def get_model_list(self) -> List[str]:
        """Get list of available OpenAI models"""
        try:
            models = self.client.models.list()
            gpt_models = []
            
            # Filter for relevant models
            relevant_prefixes = ["gpt-4o", "gpt-4", "gpt-3.5", "o1", "o3", "chatgpt"]
            excluded_suffixes = ["-instruct", "-edit", "-search", "-similarity", "-embedding"]
            
            for model in models.data:
                model_id = model.id
                
                # Check if it's a relevant model
                if any(prefix in model_id for prefix in relevant_prefixes):
                    # Exclude certain model types
                    if not any(suffix in model_id for suffix in excluded_suffixes):
                        gpt_models.append(model_id)
            
            # Sort with custom logic to put newer models first
            def model_sort_key(m):
                # Priority order for model families
                if 'gpt-4o' in m:
                    priority = 0
                elif 'o1' in m:
                    priority = 1
                elif 'gpt-4' in m and 'turbo' in m:
                    priority = 2
                elif 'gpt-4' in m:
                    priority = 3
                elif 'gpt-3.5' in m:
                    priority = 4
                else:
                    priority = 5
                
                # Extract date if present (e.g., "2024-04-09")
                import re
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', m)
                date_str = date_match.group(1) if date_match else "0000-00-00"
                
                return (priority, date_str, m)
            
            gpt_models.sort(key=model_sort_key, reverse=True)
            
            logger.info(f"Found {len(gpt_models)} OpenAI models")
            return gpt_models
            
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {e}")
            # Fallback to known models
            return sorted([m for m in PRICE_MAPPING if any(p in m for p in ["gpt", "o1"])], reverse=True)
    
    def supports_tools(self) -> bool:
        """OpenAI supports function calling for most models"""
        # O1 models don't support tools yet
        return not ('o1' in self.model.lower())
    
    def format_tool_response(self, tool_call_id: str, tool_result: Any) -> Dict[str, Any]:
        """Format tool response for OpenAI"""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
        }


class GeminiInterface(LLMInterface):
    """Interface for Google Gemini API with advanced features - Fixed implementation"""
    
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("Gemini API key is required")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = None  # Will be created per request with different configs
        logger.info(f"Initialized Gemini interface with model: {model}")
        
    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, **kwargs) -> Dict[str, Any]:
        """Send a chat request to Gemini - Fixed implementation"""
        try:
            # Extract system prompt from messages
            system_content = ""
            history = []
            
            # Convert messages to Gemini format
            for msg in messages:
                if msg['role'] == 'system':
                    system_content += msg['content'] + "\n\n"
                else:
                    # Convert role names
                    role = "user" if msg["role"] == "user" else "model"
                    content = msg.get('content')
                    
                    # Handle content that might be a list
                    if isinstance(content, list):
                        # Extract text from content list
                        text_parts = []
                        for c in content:
                            if isinstance(c, dict) and c.get('type') == 'text':
                                text_parts.append(c.get('text', ''))
                            elif isinstance(c, dict) and c.get('type') == 'tool_result':
                                # Handle tool results
                                text_parts.append(f"Tool result: {c.get('content', '')}")
                        text = " ".join(text_parts)
                        history.append({'role': role, 'parts': [text]})
                    else:
                        # Simple string content
                        history.append({'role': role, 'parts': [content]})
            
            # Prepare system instruction
            system_instruction = system_content.strip() if system_content else None
            
            # Convert tools to Gemini format
            gemini_tools = None
            if tools:
                function_declarations = []
                for tool in tools:
                    function_declarations.append({
                        "name": tool['name'],
                        "description": tool['description'],
                        "parameters": tool['input_schema']
                    })
                gemini_tools = [{"function_declarations": function_declarations}]
            
            # Create generation config
            generation_config = genai.GenerationConfig(
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.95),
                top_k=kwargs.get('top_k', 40),
                max_output_tokens=kwargs.get('max_tokens', 8192),
                stop_sequences=kwargs.get('stop_sequences', None)
            )
            
            # Create model with configuration
            self.model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_instruction,
                tools=gemini_tools,
                generation_config=generation_config
            )
            
            # Extract the latest prompt from history
            latest_prompt = []
            if history and history[-1]['role'] == 'user':
                latest_prompt = history[-1]['parts']
                history = history[:-1]  # Remove the last message from history
            elif not history:
                # No history, no prompt
                latest_prompt = ["Hello"]
            
            logger.debug(f"Gemini request - Model: {self.model_name}, History: {len(history)}, Tools: {len(tools) if tools else 0}")
            
            # Start chat with history
            chat = self.model.start_chat(history=history)
            
            # Send message
            response = chat.send_message(latest_prompt, stream=False)
            
            # Extract content from response
            content = []
            tool_calls = []
            output_text = ""
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        tool_calls.append({
                            'id': fc.name,  # Gemini uses name as ID
                            'name': fc.name,
                            'input': dict(fc.args) if fc.args else {}
                        })
                    elif hasattr(part, 'text'):
                        output_text += part.text
            
            if output_text:
                content.append({"type": "text", "text": output_text})
            
            # Calculate token usage
            try:
                # Count tokens for input (entire chat history)
                input_tokens = self.model.count_tokens(chat.history).total_tokens
                
                # Count tokens for output
                output_tokens = 0
                if output_text:
                    output_tokens = self.model.count_tokens(output_text).total_tokens
            except:
                # Fallback token estimation
                input_tokens = sum(len(str(h['parts'])) for h in history) // 4
                output_tokens = len(output_text) // 4 if output_text else 0
            
            result = {
                'success': True,
                'content': content,
                'tool_calls': tool_calls,
                'usage': {
                    'input_tokens': int(input_tokens),
                    'output_tokens': int(output_tokens),
                    'total_tokens': int(input_tokens + output_tokens)
                },
                'raw_response': response,
                'model': self.model_name,
                'finish_reason': str(response.candidates[0].finish_reason) if response.candidates else None
            }
            
            logger.debug(f"Gemini response - Tokens: {result['usage']['total_tokens']}, Tool calls: {len(tool_calls)}")
            return result
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}", exc_info=True)
            return {
                'success': False, 
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def handle_tool_result(self, messages: List[Dict], tool_call_id: str, tool_result: Any,
                          tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Handle tool result and get Gemini's response"""
        # Gemini handles tool results as regular user messages
        tool_message = {
            "role": "user",
            "content": f"Function {tool_call_id} returned: {json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result}"
        }
        messages.append(tool_message)
        
        # Get Gemini's response after tool use
        return self.chat(messages, tools=tools)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for Gemini usage"""
        pricing = PRICE_MAPPING.get(model)
        if not pricing:
            # Default pricing based on model family
            if 'gemini-2' in model:
                pricing = {"input": 0.0, "output": 0.0}  # Free experimental
            elif 'flash' in model:
                pricing = {"input": 0.35, "output": 1.05}
            elif 'pro' in model:
                pricing = {"input": 3.5, "output": 10.5}
            elif 'ultra' in model:
                pricing = {"input": 7.0, "output": 21.0}
            else:
                pricing = {"input": 1.0, "output": 3.0}
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = round(input_cost + output_cost, 6)
        
        logger.debug(f"Gemini cost calculation - Model: {model}, Input: ${input_cost:.6f}, Output: ${output_cost:.6f}, Total: ${total_cost:.6f}")
        return total_cost
    
    def get_model_list(self) -> List[str]:
        """Get list of available Gemini models"""
        try:
            models = []
            excluded_keywords = ['-tts', '-embedding', '-aqa', 'embedding', 'aqa']
            
            for m in genai.list_models():
                # Check if model supports content generation
                if 'generateContent' in m.supported_generation_methods:
                    model_name = m.name
                    
                    # Clean up model name
                    if model_name.startswith('models/'):
                        model_name = model_name[len('models/'):]
                    
                    # Exclude non-generation models
                    if not any(keyword in model_name.lower() for keyword in excluded_keywords):
                        models.append(model_name)
            
            # Add known models if not in list
            known_models = [m for m in PRICE_MAPPING if "gemini" in m]
            for known_model in known_models:
                if known_model not in models:
                    models.append(known_model)
            
            # Sort with newest first
            models.sort(reverse=True)
            
            logger.info(f"Found {len(models)} Gemini models")
            return models
            
        except Exception as e:
            logger.error(f"Failed to fetch Gemini models: {e}")
            # Fallback to known models
            return sorted([m for m in PRICE_MAPPING if "gemini" in m], reverse=True)
    
    def supports_tools(self) -> bool:
        """Gemini supports function calling for Pro and Ultra models"""
        return 'pro' in self.model_name.lower() or 'ultra' in self.model_name.lower()
    
    def format_tool_response(self, tool_call_id: str, tool_result: Any) -> Dict[str, Any]:
        """Format tool response for Gemini"""
        # Gemini expects tool results as regular messages
        return {
            "role": "user",
            "content": f"Function {tool_call_id} returned: {json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result}"
        }


# Factory function to create interfaces
def create_llm_interface(provider: str, api_key: str, model: str) -> LLMInterface:
    """Factory function to create the appropriate LLM interface"""
    if provider.lower() in ['claude', 'anthropic']:
        return ClaudeInterface(api_key, model)
    elif provider.lower() in ['openai', 'gpt']:
        return OpenAIInterface(api_key, model)
    elif provider.lower() in ['gemini', 'google']:
        return GeminiInterface(api_key, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Utility functions for model information
def get_model_info(model: str) -> Dict[str, Any]:
    """Get detailed information about a model"""
    info = {
        'name': model,
        'provider': 'unknown',
        'family': 'unknown',
        'context_window': 'unknown',
        'pricing': PRICE_MAPPING.get(model, {'input': 0, 'output': 0}),
        'supports_tools': False,
        'description': ''
    }
    
    # Determine provider and characteristics
    if 'claude' in model:
        info['provider'] = 'anthropic'
        info['supports_tools'] = True
        
        if 'opus' in model:
            info['family'] = 'opus'
            info['context_window'] = '200K'
            info['description'] = 'Most capable, best for complex tasks'
        elif 'sonnet' in model:
            info['family'] = 'sonnet'
            info['context_window'] = '200K'
            info['description'] = 'Balanced performance and cost'
        elif 'haiku' in model:
            info['family'] = 'haiku'
            info['context_window'] = '200K'
            info['description'] = 'Fast and cost-effective'
            
    elif any(prefix in model for prefix in ['gpt', 'o1', 'o3']):
        info['provider'] = 'openai'
        
        if 'gpt-4' in model:
            info['family'] = 'gpt-4'
            info['supports_tools'] = True
            if '32k' in model:
                info['context_window'] = '32K'
            elif 'turbo' in model and '128k' in model:
                info['context_window'] = '128K'
            else:
                info['context_window'] = '8K'
            
            if 'turbo' in model:
                info['description'] = 'Faster GPT-4 with improved capabilities'
            elif 'o' in model:
                info['description'] = 'Multimodal GPT-4 with vision'
            else:
                info['description'] = 'Original GPT-4, highly capable'
                
        elif 'gpt-3.5' in model:
            info['family'] = 'gpt-3.5'
            info['supports_tools'] = True
            info['context_window'] = '16K' if '16k' in model else '4K'
            info['description'] = 'Fast and cost-effective'
            
        elif 'o1' in model:
            info['family'] = 'o1'
            info['supports_tools'] = False  # O1 doesn't support tools yet
            info['context_window'] = '128K'
            if 'mini' in model:
                info['description'] = 'Reasoning model, cost-effective'
            else:
                info['description'] = 'Advanced reasoning and problem-solving'
                
    elif 'gemini' in model:
        info['provider'] = 'google'
        
        if 'ultra' in model:
            info['family'] = 'ultra'
            info['supports_tools'] = True
            info['context_window'] = '1M'
            info['description'] = 'Most capable Gemini model'
        elif 'pro' in model:
            info['family'] = 'pro'
            info['supports_tools'] = True
            info['context_window'] = '1M' if '1.5' in model else '32K'
            info['description'] = 'Advanced reasoning with long context'
        elif 'flash' in model:
            info['family'] = 'flash'
            info['supports_tools'] = True
            info['context_window'] = '1M' if '1.5' in model else '32K'
            info['description'] = 'Fast and efficient'
        
        if '2.0' in model:
            info['description'] += ' (Latest generation)'
    
    return info


def format_model_comparison(models: List[str]) -> str:
    """Format a comparison table of models"""
    lines = []
    lines.append("\n📊 Model Comparison:")
    lines.append("=" * 100)
    lines.append(f"{'Model':<40} {'Context':<10} {'Input $/1M':<12} {'Output $/1M':<12} {'Features':<20}")
    lines.append("-" * 100)
    
    for model in models:
        info = get_model_info(model)
        pricing = info['pricing']
        features = []
        if info['supports_tools']:
            features.append("Tools")
        if 'vision' in model or 'o' in model:
            features.append("Vision")
        if info['family'] in ['o1', 'opus', 'ultra']:
            features.append("Advanced")
        
        lines.append(
            f"{model:<40} {info['context_window']:<10} "
            f"${pricing['input']:<11.2f} ${pricing['output']:<11.2f} "
            f"{', '.join(features):<20}"
        )
    
    lines.append("=" * 100)
    return '\n'.join(lines)
