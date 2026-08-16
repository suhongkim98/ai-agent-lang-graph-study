"""
06. 스트리밍
LangGraph는 두 가지 스트리밍 모드를 제공합니다.

- stream(mode="updates"): 노드가 완료될 때마다 State 변경분을 반환
- stream(mode="messages"): LLM 토큰을 실시간으로 반환 (타이핑 효과)
"""

import sys
from typing import Annotated
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict


# --- 기본 채팅 그래프 ---

class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="qwen3:8b", temperature=0)


def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


graph = (
    StateGraph(State)
    .add_node("chatbot", chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile()
)


# --- 예제 1: updates 모드 — 노드 단위 스트리밍 ---

def demo_updates():
    print("=== mode='updates': 노드 완료 시 State 변경분 출력 ===\n")
    query = "파이썬의 장점 세 가지를 알려줘"
    print(f"Q: {query}\n")

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="updates",
    ):
        for node_name, state_delta in chunk.items():
            print(f"[{node_name} 완료]")
            last_msg = state_delta["messages"][-1]
            print(f"  → {last_msg.content[:100]}...\n")


# --- 예제 2: messages 모드 — 토큰 단위 스트리밍 ---

def demo_tokens():
    print("=== mode='messages': 토큰 단위 실시간 출력 ===\n")
    query = "한국의 수도는 어디야? 한 문장으로 답해줘"
    print(f"Q: {query}")
    print("A: ", end="", flush=True)

    for token, metadata in graph.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="messages",
    ):
        # AIMessageChunk만 출력 (ToolMessageChunk 등 제외)
        if hasattr(token, "content") and token.content:
            print(token.content, end="", flush=True)

    print("\n")


# --- 예제 3: 도구 호출 + 스트리밍 ---

@tool
def get_weather(city: str) -> str:
    """도시의 날씨를 반환합니다. Args: city: 도시 이름"""
    weather = {"서울": "맑음 25°C", "부산": "구름 22°C", "제주": "비 18°C"}
    return weather.get(city, f"{city}의 날씨 정보 없음")


def demo_agent_stream():
    print("=== 에이전트 스트리밍 (도구 호출 포함) ===\n")
    agent = create_react_agent(llm, [get_weather])
    query = "서울이랑 제주 날씨 비교해줘"
    print(f"Q: {query}\n")

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="updates",
    ):
        for node_name, state_delta in chunk.items():
            msgs = state_delta.get("messages", [])
            for msg in msgs:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    names = [tc["name"] for tc in msg.tool_calls]
                    print(f"[{node_name}] 도구 호출: {names}")
                elif hasattr(msg, "name") and msg.name:
                    print(f"[Tool:{msg.name}] {msg.content}")
                elif msg.content:
                    print(f"[{node_name}] {msg.content}")


if __name__ == "__main__":
    demo_updates()
    demo_tokens()
    demo_agent_stream()
