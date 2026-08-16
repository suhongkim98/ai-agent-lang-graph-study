"""
01. 기본 채팅 — LangGraph의 가장 단순한 형태
상태(State)를 정의하고, 노드(Node)를 연결해 그래프를 만드는 방법을 익힙니다.
"""

from typing import Annotated
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# 1. 상태 정의 — 그래프가 기억하는 데이터
class State(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages: 덮어쓰지 않고 누적


# 2. LLM 초기화
llm = ChatOllama(model="qwen3:8b", temperature=0)


# 3. 노드 정의 — LLM을 호출하는 함수
def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 4. 그래프 조립
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


# 5. 대화 루프
def chat():
    print("LangGraph 채팅 시작 (종료: quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
        print(f"AI: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    chat()
