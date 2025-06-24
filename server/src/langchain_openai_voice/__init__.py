import asyncio
import json
import websockets

from contextlib import asynccontextmanager
from typing import AsyncGenerator, AsyncIterator, Any, Callable, Coroutine
from langchain_openai_voice.utils import amerge

from langchain_core.tools import BaseTool
from langchain_core._api import beta
from langchain_core.utils import secret_from_env

from pydantic import BaseModel, Field, SecretStr, PrivateAttr

DEFAULT_MODEL = "gpt-4o-realtime-preview-2024-10-01"
DEFAULT_URL = "wss://api.openai.com/v1/realtime"

EVENTS_TO_IGNORE = {
    "response.function_call_arguments.delta",
    "rate_limits.updated",
    "response.audio_transcript.delta",
    "response.created",
    "response.content_part.added",
    "response.content_part.done",
    "conversation.item.created",
    "response.audio.done",
    "session.created",
    "session.updated",
    "response.done",
    "response.output_item.done",
}


@asynccontextmanager
async def connect(*, api_key: str, model: str, url: str) -> AsyncGenerator[
    tuple[
        Callable[[dict[str, Any] | str], Coroutine[Any, Any, None]],
        AsyncIterator[dict[str, Any]],
    ],
    None,
]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    url = url or DEFAULT_URL
    url += f"?model={model}"

    websocket = await websockets.connect(url, extra_headers=headers)

    try:
        async def send_event(event: dict[str, Any] | str) -> None:
            formatted_event = json.dumps(event) if isinstance(event, dict) else event
            await websocket.send(formatted_event)

        async def event_stream() -> AsyncIterator[dict[str, Any]]:
            async for raw_event in websocket:
                yield json.loads(raw_event)

        yield send_event, event_stream()
    finally:
        await websocket.close()


class VoiceToolExecutor(BaseModel):
    tools_by_name: dict[str, BaseTool]
    _trigger_future: asyncio.Future = PrivateAttr(default_factory=asyncio.Future)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def _trigger_func(self) -> dict:
        return await self._trigger_future

    async def add_tool_call(self, tool_call: dict) -> None:
        async with self._lock:
            if self._trigger_future.done():
                raise ValueError("Tool call adding already in progress")
            self._trigger_future.set_result(tool_call)

    async def _create_tool_call_task(self, tool_call: dict) -> asyncio.Task[dict]:
        tool = self.tools_by_name.get(tool_call["name"])
        if tool is None:
            raise ValueError(f"tool {tool_call['name']} not found. Must be one of {list(self.tools_by_name.keys())}")

        try:
            args = json.loads(tool_call["arguments"])
        except json.JSONDecodeError:
            raise ValueError(
                f"failed to parse arguments `{tool_call['arguments']}`. Must be valid JSON."
            )

        async def run_tool() -> dict:
            result = await tool.ainvoke(args)
            try:
                result_str = json.dumps(result)
            except TypeError:
                result_str = str(result)
            return {
                "type": "conversation.item.create",
                "item": {
                    "role": "tool",
                    "id": tool_call["call_id"],
                    "call_id": tool_call["call_id"],
                    "type": "function_call_output",
                    "output": result_str,
                },
            }

        return asyncio.create_task(run_tool())

    async def output_iterator(self) -> AsyncIterator[dict]:
        trigger_task = asyncio.create_task(self._trigger_func())
        tasks = {trigger_task}
        while True:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                tasks.remove(task)
                if task == trigger_task:
                    async with self._lock:
                        self._trigger_future = asyncio.Future()
                    trigger_task = asyncio.create_task(self._trigger_func())
                    tasks.add(trigger_task)
                    tool_call = task.result()
                    try:
                        new_task = await self._create_tool_call_task(tool_call)
                        tasks.add(new_task)
                    except ValueError as e:
                        yield {
                            "type": "conversation.item.create",
                            "item": {
                                "role": "assistant",
                                "id": tool_call["call_id"],
                                "call_id": tool_call["call_id"],
                                "type": "function_call_output",
                                "output": f"Error: {str(e)}",
                            },
                        }
                else:
                    yield task.result()


@beta()
class OpenAIVoiceReactAgent(BaseModel):
    model: str
    api_key: SecretStr = Field(
        alias="openai_api_key",
        default_factory=secret_from_env("OPENAI_API_KEY", default=""),
    )
    instructions: str | None = None
    tools: list[BaseTool] | None = None
    url: str = Field(default=DEFAULT_URL)

    async def aconnect(
        self,
        input_stream: AsyncIterator[str],
        send_output_chunk: Callable[[str], Coroutine[Any, Any, None]],
    ) -> None:
        tools_by_name = {tool.name: tool for tool in self.tools or []}
        tool_executor = VoiceToolExecutor(tools_by_name=tools_by_name)
        transcripts: list[tuple[str, str]] = []
        close_session_called = False

        pending_analysis: asyncio.Task[str] | None = None
        pending_analysis_call_id: str | None = None
        pending_close_call_id: str | None = None

        retry_delay = 1.0

        async with connect(
            model=self.model, api_key=self.api_key.get_secret_value(), url=self.url
        ) as (model_send, model_receive_stream):
            # send initial session update
            tool_defs = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": tool.args},
                }
                for tool in tools_by_name.values()
            ]
            await model_send({
                "type": "session.update",
                "session": {
                    "instructions": self.instructions,
                    "input_audio_transcription": {"model": "whisper-1"},
                    "tools": tool_defs,
                },
            })

            try:
                async for stream_key, data_raw in amerge(
                    input_mic=input_stream,
                    output_speaker=model_receive_stream,
                    tool_outputs=tool_executor.output_iterator(),
                ):
                    try:
                        data = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
                    except json.JSONDecodeError:
                        continue

                    if stream_key == "input_mic":
                        await model_send(data)

                    elif stream_key == "tool_outputs":
                        await model_send(data)
                        await model_send({"type": "response.create", "response": {}})

                    elif stream_key == "output_speaker":
                        t = data.get("type")

                        if t == "response.audio.delta":
                            await send_output_chunk(json.dumps(data))

                        elif t == "input_audio_buffer.speech_started":
                            await send_output_chunk(json.dumps(data))

                        elif t == "error":
                            # handle rate limits
                            err = data.get("error", {})
                            if err.get("code") == 429:
                                await asyncio.sleep(retry_delay)
                                retry_delay = min(retry_delay * 2, 32)
                                continue
                            else:
                                retry_delay = 1.0

                        elif t == "response.function_call_arguments.done":
                            name = data.get("name")
                            if name == "end_negotiation":
                                transcript_text = "\n".join(f"{s}: {t}" for s, t in transcripts)
                                pending_analysis = asyncio.create_task(
                                    self._analyze_transcript(transcript_text)
                                )
                                pending_analysis_call_id = data["call_id"]

                            elif name == "close_session":
                                pending_close_call_id = data["call_id"]
                                close_session_called = True

                            else:
                                await tool_executor.add_tool_call(data)

                        elif t == "response.audio_transcript.done":
                            transcripts.append(("model", data["transcript"]))

                        elif t == "conversation.item.input_audio_transcription.completed":
                            transcripts.append(("user", data["transcript"]))

                        elif t in EVENTS_TO_IGNORE:
                            pass

                        if t == "response.done":
                            retry_delay = 1.0
                            if pending_analysis and pending_analysis_call_id:
                                analysis = await pending_analysis
                                await model_send({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "role": "assistant",
                                        "id": pending_analysis_call_id,
                                        "call_id": pending_analysis_call_id,
                                        "type": "function_call_output",
                                        "output": analysis,
                                    },
                                })
                                await model_send({"type": "response.create", "response": {}})
                                pending_analysis = None
                                pending_analysis_call_id = None

                            if pending_close_call_id:
                                await model_send({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "role": "assistant",
                                        "id": pending_close_call_id,
                                        "call_id": pending_close_call_id,
                                        "type": "function_call_output",
                                        "output": "END",
                                    },
                                })
                                await model_send({"type": "response.create", "response": {}})
                                pending_close_call_id = None

                        if close_session_called and t == "response.done":
                            break

            except StopAsyncIteration:
                # clean shutdown
                pass

    async def _analyze_transcript(self, transcript: str) -> str:
        import openai

        client = openai.AsyncOpenAI(api_key=self.api_key.get_secret_value())
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a negotiation coach. Provide feedback on the conversation transcript."},
                {"role": "user", "content": transcript},
            ],
        )
        return resp.choices[0].message.content


__all__ = ["OpenAIVoiceReactAgent"]
