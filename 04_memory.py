"""
04. 멀티턴 대화 메모리 — MemorySaver
MemorySaver는 thread_id별로 State를 저장합니다.
같은 thread_id로 호출하면 이전 대화를 이어갑니다.
"""

from typing import Annotated
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="qwen3:8b", temperature=0)


def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


# MemorySaver: 인메모리 체크포인터 — thread_id별로 State를 스냅샷으로 보존
checkpointer = MemorySaver()

graph = (
    StateGraph(State)
    .add_node("chatbot", chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile(checkpointer=checkpointer)  # 체크포인터 주입
)


def chat(thread_id: str = "default"):
    """thread_id가 같으면 이전 대화를 기억합니다."""
    config = {"configurable": {"thread_id": thread_id}}
    print(f"대화 시작 (thread: {thread_id}) — 종료: quit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print(f"AI: {result['messages'][-1].content}\n")


def demo_memory():
    """같은 thread로 두 번 호출해서 메모리 동작 확인."""
    config = {"configurable": {"thread_id": "demo-1"}}

    def ask(q):
        result = graph.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config=config,
        )
        print(f"You: {q}")
        print(f"AI : {result['messages'][-1].content}\n")

    ask("내 이름은 수홍이야.")
    ask("내 이름이 뭐야?")  # 메모리가 있으면 '수홍'을 기억함

    # 다른 thread — 기억 없음
    config2 = {"configurable": {"thread_id": "demo-2"}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "내 이름이 뭐야?"}]},
        config=config2,
    )
    print(f"[새 thread] AI: {result['messages'][-1].content}")


if __name__ == "__main__":
    demo_memory()
