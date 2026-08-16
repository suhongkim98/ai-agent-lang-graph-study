"""
04. 멀티턴 대화 메모리 — Python dict 기반
MemorySaver(LangGraph 체크포인터) 대신 dict로 thread_id별 메시지를 직접 관리합니다.

구조:
  store: dict[thread_id, list[message]] — 대화 저장소
  그래프는 체크포인터 없이 컴파일하고, 호출 전에 이전 메시지를 직접 주입합니다.
"""

from typing import Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ── 대화 저장소 ────────────────────────────────────
# { thread_id: [HumanMessage, AIMessage, ...] }
store: dict[str, list[BaseMessage]] = {}


# ── 그래프 ─────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="qwen3:8b", temperature=0)


def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


# 체크포인터 없이 컴파일 — 상태 관리를 직접 담당
graph = (
    StateGraph(State)
    .add_node("chatbot", chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile()
)


# ── 대화 함수 ──────────────────────────────────────

def invoke(thread_id: str, user_message: str) -> str:
    """
    thread_id 기준으로 이전 대화를 불러와 LLM에 전달하고,
    응답을 다시 store에 저장합니다.
    """
    history = store.setdefault(thread_id, [])

    # 이전 메시지 + 새 메시지를 함께 전달
    result = graph.invoke({
        "messages": history + [{"role": "user", "content": user_message}]
    })

    # 전체 메시지 목록을 저장소에 갱신
    store[thread_id] = result["messages"]

    return result["messages"][-1].content


def get_history(thread_id: str) -> list[BaseMessage]:
    """thread_id의 전체 대화 기록을 반환합니다."""
    return store.get(thread_id, [])


def clear_history(thread_id: str) -> None:
    """thread_id의 대화 기록을 삭제합니다."""
    store.pop(thread_id, None)


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

    print("=== store 현황 ===")
    for tid, messages in store.items():
        roles = [m.__class__.__name__ for m in messages]
        print(f"  [{tid}] {len(messages)}개 메시지: {roles}")


def chat(thread_id: str = "default"):
    """대화 루프 (종료: quit)"""
    print(f"대화 시작 (thread: {thread_id}) — 종료: quit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        print(f"AI: {invoke(thread_id, user_input)}\n")


if __name__ == "__main__":
    demo_memory()
