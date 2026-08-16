"""
03-B. MCP 클라이언트 + LangGraph
langchain-mcp-adapters를 통해 MCP 서버의 도구를 LangGraph 에이전트에 연결합니다.

흐름:
  1. MultiServerMCPClient가 03_mcp_server.py를 subprocess로 실행
  2. MCP 프로토콜로 도구 목록을 가져옴
  3. 도구를 LangChain Tool 형식으로 변환
  4. LangGraph ReAct 에이전트가 도구를 사용해 질문에 답변
"""

import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


SERVER_SCRIPT = str(Path(__file__).parent / "03_mcp_server.py")

llm = ChatOllama(model="qwen3:8b", temperature=0)


async def run(query: str):
    print(f"\nQuery: {query}")
    print("-" * 40)

    # MCP 클라이언트 — 서버를 subprocess로 띄워 stdio로 통신
    client = MultiServerMCPClient(
        {
            "study": {
                "command": sys.executable,
                "args": [SERVER_SCRIPT],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

    for msg in result["messages"]:
        role = msg.__class__.__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"[{role}] 도구 호출: {[tc['name'] for tc in msg.tool_calls]}")
        elif hasattr(msg, "name") and msg.name:
            print(f"[Tool:{msg.name}] {msg.content}")
        else:
            print(f"[{role}] {msg.content}")


async def main():
    project_dir = str(Path(__file__).parent)
    await run("지금 몇 시야?")
    await run(f"{project_dir} 디렉토리에 어떤 파일들이 있어?")


if __name__ == "__main__":
    asyncio.run(main())
