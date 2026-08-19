"""
04. 멀티턴 대화 메모리 — MemorySaver 기반
LangGraph의 MemorySaver 체크포인터로 thread_id별 대화를 자동 관리합니다.
운영환경에서는 PostgresSaver를 사용하자

구조:
  MemorySaver: thread_id별 체크포인트를 인메모리 dict에 저장
  그래프를 checkpointer=memory로 컴파일하고, config에 thread_id를 전달합니다.
"""

from typing import Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ── 그래프 ─────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="qwen3:8b", temperature=0)


def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


memory = MemorySaver()

graph = (
    StateGraph(State)
    .add_node("chatbot", chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile(checkpointer=memory)
)


# ── 대화 함수 ──────────────────────────────────────

def invoke(thread_id: str, user_message: str) -> str:
    """
    thread_id 기준으로 MemorySaver가 이전 대화를 자동으로 복원해
    LLM에 전달하고, 응답을 체크포인트에 저장합니다.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )
    return result["messages"][-1].content


def get_history(thread_id: str) -> list[BaseMessage]:
    """thread_id의 전체 대화 기록을 반환합니다."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    return state.values.get("messages", []) if state.values else []


def clear_history(thread_id: str) -> None:
    """thread_id의 대화 기록을 삭제합니다."""
    memory.storage.pop(thread_id, None)


# ── 데모 ───────────────────────────────────────────

def demo_memory():
    print("=== thread-A: 이름 기억 ===\n")
    print(f"You: 내 이름은 수홍이야.")
    print(f"AI : {invoke('thread-A', '내 이름은 수홍이야.')}\n")

    print(f"You: 내 이름이 뭐야?")
    print(f"AI : {invoke('thread-A', '내 이름이 뭐야?')}\n")

    print(f"저장된 메시지 수: {len(get_history('thread-A'))}개\n")

    print("=== thread-B: 독립된 대화 ===\n")
    print(f"You: 내 이름이 뭐야?")
    print(f"AI : {invoke('thread-B', '내 이름이 뭐야?')}\n")

    print("=== 체크포인트 현황 ===")
    for tid in ("thread-A", "thread-B"):
        msgs = get_history(tid)
        roles = [m.__class__.__name__ for m in msgs]
        print(f"  [{tid}] {len(msgs)}개 메시지: {roles}")


def chat(thread_id: str = "default"):
    """대화 루프 (종료: quit)"""
    config = {"configurable": {"thread_id": thread_id}}
    print(f"대화 시작 (thread: {thread_id}) — 종료: quit\n")
    while True:
        user_input = input("You: ").encode("utf-8", errors="replace").decode("utf-8").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print(f"AI: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    demo_memory()
