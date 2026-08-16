"""
03-A. MCP 서버
MCP(Model Context Protocol)는 LLM에게 도구를 제공하는 표준 프로토콜입니다.
이 파일은 stdio 방식으로 동작하는 MCP 서버입니다.
클라이언트(03_mcp_client.py)가 이 서버를 subprocess로 띄워서 통신합니다.
"""

import json
import os
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


server = Server("study-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """서버가 제공하는 도구 목록을 반환합니다."""
    return [
        types.Tool(
            name="get_time",
            description="현재 날짜와 시간을 반환합니다.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="list_files",
            description="지정한 디렉토리의 파일 목록을 반환합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "조회할 디렉토리 경로"},
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="read_file",
            description="파일 내용을 읽어 반환합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "읽을 파일 경로"},
                },
                "required": ["path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """도구를 실행하고 결과를 반환합니다."""
    if name == "get_time":
        result = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elif name == "list_files":
        path = arguments["path"]
        if not os.path.isdir(path):
            result = f"디렉토리가 존재하지 않습니다: {path}"
        else:
            entries = os.listdir(path)
            result = json.dumps(entries, ensure_ascii=False)

    elif name == "read_file":
        path = arguments["path"]
        if not os.path.isfile(path):
            result = f"파일이 존재하지 않습니다: {path}"
        else:
            with open(path, encoding="utf-8") as f:
                result = f.read()

    else:
        result = f"알 수 없는 도구: {name}"

    return [types.TextContent(type="text", text=result)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
