#!/usr/bin/env python3
"""
Multi LLM Bridge - Status Message Fix
Replace these methods in your multi_llm_bridge.py to use the status line system

Key changes:
- Tool execution messages go to status line
- Thinking indicators use status line
- Tool result processing messages use status line
"""

# === In the MultiLLMBridge class, update these methods: ===

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
    
    # FIX: Show thinking indicator in status line
    if hasattr(self.ui, 'set_status'):
        self.ui.set_status(f"{self.main_llm.value} thinking...")
    else:
        # Fallback for UIs without status support
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
            
            # FIX: Update status for iteration
            if hasattr(self.ui, 'set_status'):
                self.ui.set_status(f"Processing message (iteration {iteration})...")
            
            # Get response from main LLM
            result = self.main_interface.chat(
                messages=messages,
                system_prompt=system_prompt if self.main_llm == LLMType.CLAUDE else None,
                tools=self.tools if self.sub_llms and self.main_interface.supports_tools() else None
            )
            
            if not result['success']:
                self.stats[self.main_llm]['errors'] += 1
                error_msg = f"Error: {result['error']}"
                # FIX: Clear status on error
                if hasattr(self.ui, 'clear_status'):
                    self.ui.clear_status()
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
                # FIX: Clear status when done
                if hasattr(self.ui, 'clear_status'):
                    self.ui.clear_status()
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
                
                # FIX: Show tool usage in status line
                if hasattr(self.ui, 'set_status'):
                    self.ui.set_status(f"Executing tool: {tool_call['name']}")
                else:
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
                    # FIX: Update status for tool result processing
                    if hasattr(self.ui, 'set_status'):
                        self.ui.set_status("Processing tool result...")
                    
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
            # FIX: Clear status on error
            if hasattr(self.ui, 'clear_status'):
                self.ui.clear_status()
            self.ui.add_message(self.main_llm.value, error_msg, "error")
            return error_msg
    
    # FIX: Clear status when complete
    if hasattr(self.ui, 'clear_status'):
        self.ui.clear_status()
    
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


def _query_sub_llm(self, llm_type: LLMType, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Query a sub-LLM and handle the response (enhanced for self-calling)"""
    prompt = tool_input.get('prompt', '')
    temperature = tool_input.get('temperature', 0.7)
    max_tokens = tool_input.get('max_tokens', 4096)
    include_context = tool_input.get('include_context', False)
    
    # Check if this is a self-call
    is_self_call = (llm_type == self.main_llm)
    
    # Debug logging
    logger.info(f"Querying {'SELF' if is_self_call else 'sub'}-LLM: {llm_type.value}")
    logger.info(f"Self-call depth: {self.self_call_depth}")
    
    # Check self-call depth limit
    if is_self_call and self.self_call_depth >= self.max_self_call_depth:
        logger.warning(f"Self-call depth limit reached: {self.self_call_depth}")
        return {
            "success": False, 
            "error": f"Self-call depth limit ({self.max_self_call_depth}) reached to prevent deep recursion"
        }
    
    # Get the correct interface
    if is_self_call:
        interface = self.main_interface
    else:
        if llm_type not in self.sub_llms:
            logger.error(f"LLM type {llm_type.value} not found in sub_llms!")
            return {"success": False, "error": f"Sub-LLM {llm_type.value} not available"}
        interface = self.sub_llms[llm_type]
    
    logger.info(f"Using interface: {type(interface).__name__} for {llm_type.value}")
    
    # FIX: Update UI with status instead of adding messages
    if isinstance(self.ui, MultiPaneUI):
        if is_self_call:
            if hasattr(self.ui, 'set_status'):
                self.ui.set_status(f"Self-querying {llm_type.value}...")
            else:
                self.ui.add_message("system", f"🔄 Self-querying {llm_type.value}...", "tool")
        else:
            self.ui.update_sub_pane(llm_type, query=prompt, status='processing')
    elif isinstance(self.ui, MultiWindowUI):
        target = llm_type if not is_self_call else self.main_llm
        # Don't add these as regular messages
        if hasattr(self.ui, 'set_status'):
            self.ui.set_status(f"{'Self-query' if is_self_call else 'Query'} to {llm_type.value}: {prompt[:50]}...")
        else:
            self.ui.add_line(target, f"\n{'🔄 Self-query' if is_self_call else '🔧 Receiving query'} from {self.main_llm.value}:")
            self.ui.add_line(target, f"Prompt: {prompt}")
            self.ui.add_line(target, "Processing...")
    else:
        action = "Self-querying" if is_self_call else "Querying"
        if hasattr(self.ui, 'set_status'):
            self.ui.set_status(f"{action} {llm_type.value}...")
        else:
            self.ui.add_message("system", f"{action} {llm_type.value}...", "tool")
    
    self.ui.refresh_display()
    
    # ... rest of the method remains the same until the response handling ...
    
    # After getting the result, update the UI
    if result['success']:
        # ... existing success handling ...
        
        # FIX: Clear status after successful response
        if hasattr(self.ui, 'clear_status'):
            self.ui.clear_status()
        
        # Update UI
        if is_self_call:
            # For self-calls, update main window
            if isinstance(self.ui, (MultiWindowUI, MultiPaneUI)):
                # Don't add "Self-query completed" message, just the response
                self.ui.add_message(llm_type.value, text_content, "assistant")
        # ... rest of the success handling ...
    
    else:
        # Handle error
        self.stats[llm_type]['errors'] += 1
        error_msg = result.get('error', 'Unknown error')
        
        # FIX: Clear status on error
        if hasattr(self.ui, 'clear_status'):
            self.ui.clear_status()
        
        # ... rest of error handling ...
