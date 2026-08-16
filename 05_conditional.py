"""
05. 조건부 분기 그래프
LLM이 질문의 성격을 판단해서 다른 노드로 라우팅합니다.

그래프 구조:
  START → classify → (조건부) → math_node  → END
                              → search_node → END
                              → chat_node   → END
"""

from typing import Annotated, Literal
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: str  # 라우팅 결과를 저장


llm = ChatOllama(model="qwen3:8b", temperature=0)

CLASSIFY_PROMPT = """사용자의 질문을 다음 중 하나로 분류하세요. 단어 하나만 답하세요.
- math: 수학 계산, 숫자 관련
- search: 사실, 지식, 정보 검색
- chat: 일상 대화, 감정, 의견

질문: {question}
분류:"""


# --- 노드 ---

def classify(state: State) -> State:
    """질문 유형을 판단해서 route에 저장."""
    question = state["messages"][-1].content
    prompt = CLASSIFY_PROMPT.format(question=question)
    result = llm.invoke([{"role": "user", "content": prompt}])
    route = result.content.strip().lower()
    # 예상 외 값 방어
    if route not in ("math", "search", "chat"):
        route = "chat"
    print(f"  → 라우팅: {route}")
    return {"route": route}


def math_node(state: State) -> State:
    system = SystemMessage("당신은 수학 전문가입니다. 계산 과정을 단계별로 설명하세요.")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def search_node(state: State) -> State:
    system = SystemMessage("당신은 지식 전문가입니다. 정확한 정보를 간결하게 제공하세요.")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def chat_node(state: State) -> State:
    system = SystemMessage("당신은 친근한 대화 상대입니다. 자연스럽게 대화하세요.")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


# --- 라우터 함수 ---

def router(state: State) -> Literal["math_node", "search_node", "chat_node"]:
    return f"{state['route']}_node"


# --- 그래프 조립 ---

graph = (
    StateGraph(State)
    .add_node("classify", classify)
    .add_node("math_node", math_node)
    .add_node("search_node", search_node)
    .add_node("chat_node", chat_node)
    .add_edge(START, "classify")
    # add_conditional_edges: 라우터 함수 결과에 따라 다음 노드 결정
    .add_conditional_edges("classify", router)
    .add_edge("math_node", END)
    .add_edge("search_node", END)
    .add_edge("chat_node", END)
    .compile()
)


def ask(question: str):
    print(f"\nQ: {question}")
    result = graph.invoke({
        "messages": [{"role": "user", "content": question}],
        "route": "",
    })
    print(f"A: {result['messages'][-1].content}")


if __name__ == "__main__":
    ask("357 * 489 계산해줘")
    ask("파이썬이 만들어진 연도는?")
    ask("오늘 기분이 좀 안 좋아")
