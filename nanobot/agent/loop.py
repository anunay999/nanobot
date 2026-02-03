"""Agent loop: the core processing engine."""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import SessionManager
from nanobot.utils.helpers import truncate_string
from nanobot.utils.opik import init_opik


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        firecrawl_api_key: str | None = None,
        web_search_provider: str = "auto",
        opik_config: Any | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.firecrawl_api_key = firecrawl_api_key
        self.web_search_provider = web_search_provider
        self.opik = init_opik(opik_config)
        
        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            firecrawl_api_key=firecrawl_api_key,
            web_search_provider=web_search_provider,
        )
        
        self._running = False
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools
        self.tools.register(ReadFileTool())
        self.tools.register(WriteFileTool())
        self.tools.register(EditFileTool())
        self.tools.register(ListDirTool())
        
        # Shell tool
        self.tools.register(ExecTool(working_dir=str(self.workspace)))
        
        # Web tools
        self.tools.register(WebSearchTool(
            brave_api_key=self.brave_api_key,
            firecrawl_api_key=self.firecrawl_api_key,
            default_provider=self.web_search_provider,
        ))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
    
    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                log = self._get_logger(msg)
                # Process it
                try:
                    response = await self._process_message(msg, log)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    log.exception("Error processing message")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=(
                            "Sorry, I encountered an error while processing your message. "
                            f"Reference ID: {msg.message_id}"
                        ),
                        metadata={"error_id": msg.message_id},
                        correlation_id=msg.message_id
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(
        self,
        msg: InboundMessage,
        log: Any | None = None,
    ) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
        
        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        log = log or self._get_logger(msg)
        if msg.channel == "system":
            return await self._process_system_message(msg, log)

        log.info("Processing message")
        start_time = time.monotonic()
        trace = self._start_opik_trace(msg, log, name="agent_message")
        
        try:
            # Get or create session
            session = self.sessions.get_or_create(msg.session_key)
            
            # Update tool contexts
            message_tool = self.tools.get("message")
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(msg.channel, msg.chat_id)
            
            spawn_tool = self.tools.get("spawn")
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(msg.channel, msg.chat_id)
            
            # Build initial messages (use get_history for LLM-formatted messages)
            messages = self.context.build_messages(
                history=session.get_history(),
                current_message=msg.content
            )
            
            # Agent loop
            iteration = 0
            final_content = None
            
            while iteration < self.max_iterations:
                iteration += 1
                
                # Call LLM
                llm_start = time.monotonic()
                llm_start_dt = datetime.now(timezone.utc)
                response = await self.provider.chat(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=self.model
                )
                llm_elapsed = time.monotonic() - llm_start
                llm_end_dt = datetime.now(timezone.utc)
                log.debug(
                    f"LLM response in {llm_elapsed:.2f}s "
                    f"(iteration={iteration}, tool_calls={len(response.tool_calls)})"
                )
                self._log_opik_llm_span(
                    trace=trace,
                    messages=messages,
                    response=response,
                    model=self.model,
                    start_time=llm_start_dt,
                    end_time=llm_end_dt,
                    iteration=iteration,
                )
                
                # Handle tool calls
                if response.has_tool_calls:
                    # Add assistant message with tool calls
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)  # Must be JSON string
                            }
                        }
                        for tc in response.tool_calls
                    ]
                    messages = self.context.add_assistant_message(
                        messages, response.content, tool_call_dicts
                    )
                    
                    # Execute tools
                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments)
                        tool_start = time.monotonic()
                        tool_start_dt = datetime.now(timezone.utc)
                        log.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                        tool_elapsed = time.monotonic() - tool_start
                        tool_end_dt = datetime.now(timezone.utc)
                        log.debug(f"Tool {tool_call.name} finished in {tool_elapsed:.2f}s")
                        if isinstance(result, str) and result.startswith("Error:"):
                            log.warning(f"Tool {tool_call.name} failed: {result}")
                        self._log_opik_tool_span(
                            trace=trace,
                            tool_name=tool_call.name,
                            tool_args=tool_call.arguments,
                            tool_result=result,
                            start_time=tool_start_dt,
                            end_time=tool_end_dt,
                            iteration=iteration,
                        )
                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )
                else:
                    # No tool calls, we're done
                    final_content = response.content
                    break
            
            if final_content is None:
                final_content = "I've completed processing but have no response to give."
            
            # Save to session
            session.add_message("user", msg.content)
            session.add_message("assistant", final_content)
            self.sessions.save(session)
            
            total_elapsed = time.monotonic() - start_time
            log.info(f"Message processed in {total_elapsed:.2f}s")
            self._end_opik_trace(trace, output={"response": final_content})
            
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=final_content,
                correlation_id=msg.message_id
            )
        except Exception as e:
            self._end_opik_trace(trace, output={"response": f"Error: {e}"}, error_message=str(e))
            raise
    
    async def _process_system_message(
        self,
        msg: InboundMessage,
        log: Any | None = None,
    ) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        log = log or self._get_logger(msg)
        log.info(f"Processing system message from {msg.sender_id}")
        trace = self._start_opik_trace(msg, log, name="system_message")
        
        try:
            # Parse origin from chat_id (format: "channel:chat_id")
            if ":" in msg.chat_id:
                parts = msg.chat_id.split(":", 1)
                origin_channel = parts[0]
                origin_chat_id = parts[1]
            else:
                # Fallback
                origin_channel = "cli"
                origin_chat_id = msg.chat_id
            
            # Use the origin session for context
            session_key = f"{origin_channel}:{origin_chat_id}"
            session = self.sessions.get_or_create(session_key)
            
            # Update tool contexts
            message_tool = self.tools.get("message")
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(origin_channel, origin_chat_id)
            
            spawn_tool = self.tools.get("spawn")
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(origin_channel, origin_chat_id)
            
            # Build messages with the announce content
            messages = self.context.build_messages(
                history=session.get_history(),
                current_message=msg.content
            )
            
            # Agent loop (limited for announce handling)
            iteration = 0
            final_content = None
            
            while iteration < self.max_iterations:
                iteration += 1
                
                llm_start = time.monotonic()
                llm_start_dt = datetime.now(timezone.utc)
                response = await self.provider.chat(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=self.model
                )
                llm_end_dt = datetime.now(timezone.utc)
                llm_elapsed = time.monotonic() - llm_start
                log.debug(
                    f"LLM response in {llm_elapsed:.2f}s "
                    f"(iteration={iteration}, tool_calls={len(response.tool_calls)})"
                )
                self._log_opik_llm_span(
                    trace=trace,
                    messages=messages,
                    response=response,
                    model=self.model,
                    start_time=llm_start_dt,
                    end_time=llm_end_dt,
                    iteration=iteration,
                )
                
                if response.has_tool_calls:
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                            }
                        }
                        for tc in response.tool_calls
                    ]
                    messages = self.context.add_assistant_message(
                        messages, response.content, tool_call_dicts
                    )
                    
                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments)
                        tool_start = time.monotonic()
                        tool_start_dt = datetime.now(timezone.utc)
                        log.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                        tool_elapsed = time.monotonic() - tool_start
                        tool_end_dt = datetime.now(timezone.utc)
                        log.debug(f"Tool {tool_call.name} finished in {tool_elapsed:.2f}s")
                        if isinstance(result, str) and result.startswith("Error:"):
                            log.warning(f"Tool {tool_call.name} failed: {result}")
                        self._log_opik_tool_span(
                            trace=trace,
                            tool_name=tool_call.name,
                            tool_args=tool_call.arguments,
                            tool_result=result,
                            start_time=tool_start_dt,
                            end_time=tool_end_dt,
                            iteration=iteration,
                        )
                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )
                else:
                    final_content = response.content
                    break
            
            if final_content is None:
                final_content = "Background task completed."
            
            # Save to session (mark as system message in history)
            session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
            session.add_message("assistant", final_content)
            self.sessions.save(session)
            self._end_opik_trace(trace, output={"response": final_content})
            
            return OutboundMessage(
                channel=origin_channel,
                chat_id=origin_chat_id,
                content=final_content,
                correlation_id=msg.message_id
            )
        except Exception as e:
            self._end_opik_trace(trace, output={"response": f"Error: {e}"}, error_message=str(e))
            raise
    
    async def process_direct(self, content: str, session_key: str = "cli:direct") -> str:
        """
        Process a message directly (for CLI usage).
        
        Args:
            content: The message content.
            session_key: Session identifier.
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content=content
        )
        
        response = await self._process_message(msg, self._get_logger(msg))
        return response.content if response else ""

    def _get_logger(self, msg: InboundMessage) -> Any:
        """Get a logger bound with message context."""
        return logger.bind(
            message_id=msg.message_id,
            channel=msg.channel,
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            session_key=msg.session_key,
        )

    def _start_opik_trace(self, msg: InboundMessage, log: Any, name: str) -> Any | None:
        if not self.opik:
            return None
        try:
            return self.opik.trace(
                name=name,
                input={
                    "message": truncate_string(msg.content, max_len=4000),
                },
                metadata={
                    "message_id": msg.message_id,
                    "channel": msg.channel,
                    "chat_id": msg.chat_id,
                    "sender_id": msg.sender_id,
                    "session_key": msg.session_key,
                },
                tags=[msg.channel],
                thread_id=msg.session_key,
            )
        except Exception as e:
            log.warning(f"Opik trace init failed: {e}")
            return None

    def _end_opik_trace(
        self,
        trace: Any | None,
        output: dict[str, Any],
        error_message: str | None = None,
    ) -> None:
        if not trace:
            return
        try:
            trace.end(
                output={
                    "response": truncate_string(str(output.get("response", "")), max_len=8000),
                },
                error_info={"message": error_message} if error_message else None,
            )
        except Exception:
            return

    def _log_opik_llm_span(
        self,
        trace: Any | None,
        messages: list[dict[str, Any]],
        response: Any,
        model: str,
        start_time: datetime,
        end_time: datetime,
        iteration: int,
    ) -> None:
        if not trace:
            return
        try:
            trace.span(
                name="llm_call",
                type="llm",
                start_time=start_time,
                end_time=end_time,
                input={"messages": self._trim_messages(messages)},
                output={
                    "content": truncate_string(response.content or "", max_len=8000),
                    "finish_reason": response.finish_reason,
                },
                usage=response.usage or None,
                model=model,
                provider=self._provider_from_model(model),
                metadata={"iteration": iteration},
            )
        except Exception:
            return

    def _log_opik_tool_span(
        self,
        trace: Any | None,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any,
        start_time: datetime,
        end_time: datetime,
        iteration: int,
    ) -> None:
        if not trace:
            return
        try:
            trace.span(
                name=f"tool:{tool_name}",
                type="tool",
                start_time=start_time,
                end_time=end_time,
                input={"args": tool_args},
                output={"result": truncate_string(str(tool_result), max_len=8000)},
                metadata={"iteration": iteration},
            )
        except Exception:
            return

    def _trim_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trimmed = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                content = truncate_string(content, max_len=2000)
            trimmed.append({"role": msg.get("role"), "content": content})
        return trimmed

    def _provider_from_model(self, model: str) -> str | None:
        lower = (model or "").lower()
        if lower.startswith("openrouter/"):
            return "openrouter"
        if lower.startswith("anthropic/"):
            return "anthropic"
        if lower.startswith("openai/") or "gpt" in lower:
            return "openai"
        if lower.startswith("gemini/") or "gemini" in lower:
            return "gemini"
        if lower.startswith("zhipu/") or "glm" in lower or "zai" in lower:
            return "zhipu"
        if lower.startswith("hosted_vllm/"):
            return "vllm"
        return None
