"""
09. 종합 챗봇 — 지금까지 배운 것을 하나로
메모리 + 조건부 라우팅 + 도구 호출 + 스트리밍을 결합합니다.

그래프 구조:
  START → router → tool_agent  → END  (도구 필요 시)
                 → direct_chat → END  (일반 대화)

메모리: MemorySaver 대신 Python dict로 thread_id별 메시지를 직접 관리합니다.
"""

from datetime import datetime
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict


# ── 도구 ──────────────────────────────────────────

@tool
def get_time() -> str:
    """현재 날짜와 시간을 반환합니다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


@tool
def calculate(expression: str) -> str:
    """
    수식을 계산합니다.
    Args:
        expression: 계산할 수식 (예: "2 ** 10")
    """
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "허용되지 않는 문자"
        return str(eval(expression))  # noqa: S307
    except Exception as e:
        return f"오류: {e}"


@tool
def remember_fact(fact: str) -> str:
    """
    사용자가 알려준 중요한 사실을 기억합니다.
    Args:
        fact: 기억할 내용
    """
    return f"기억했습니다: {fact}"


TOOLS = [get_time, calculate, remember_fact]

# ── LLM ───────────────────────────────────────────

llm = ChatOllama(model="qwen3:8b", temperature=0)

SYSTEM = SystemMessage(
    "당신은 친절하고 유능한 AI 어시스턴트입니다. "
    "계산이나 시간 조회가 필요하면 도구를 사용하세요. "
    "사용자가 중요한 정보를 알려주면 remember_fact 도구로 기억하세요."
)

# ── 상태 ───────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    use_tools: bool


# ── 노드 ───────────────────────────────────────────

ROUTE_PROMPT = """사용자 메시지가 다음 중 하나에 해당하면 'tools'라고만 답하고, 아니면 'chat'이라고만 답하세요.
- 수학 계산
- 현재 시간/날짜 조회
- 기억 요청 ("기억해줘", "저장해줘" 등)

메시지: {message}
분류:"""


def router_node(state: State) -> State:
    last = state["messages"][-1].content
    result = llm.invoke([{"role": "user", "content": ROUTE_PROMPT.format(message=last)}])
    use_tools = "tools" in result.content.lower()
    return {"use_tools": use_tools}


# 도구 에이전트 — create_react_agent를 노드로 래핑
_tool_agent = create_react_agent(llm, TOOLS, prompt=SYSTEM)


def tool_agent_node(state: State) -> State:
    result = _tool_agent.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


def direct_chat_node(state: State) -> State:
    response = llm.invoke([SYSTEM] + state["messages"])
    return {"messages": [response]}


def route(state: State) -> Literal["tool_agent_node", "direct_chat_node"]:
    return "tool_agent_node" if state["use_tools"] else "direct_chat_node"


# ── 대화 저장소 ────────────────────────────────────
# { thread_id: [HumanMessage, AIMessage, ...] }
store: dict[str, list[BaseMessage]] = {}

# ── 그래프 ─────────────────────────────────────────

graph = (
    StateGraph(State)
    .add_node("router_node", router_node)
    .add_node("tool_agent_node", tool_agent_node)
    .add_node("direct_chat_node", direct_chat_node)
    .add_edge(START, "router_node")
    .add_conditional_edges("router_node", route)
    .add_edge("tool_agent_node", END)
    .add_edge("direct_chat_node", END)
    .compile()
)

ANSWER_NODES = {"tool_agent_node", "direct_chat_node"}


# ── 실행 ───────────────────────────────────────────

def chat(thread_id: str = "user-1"):
    print("종합 챗봇 시작 (종료: quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        history = store.setdefault(thread_id, [])

        # 스트리밍 — 토큰 출력과 동시에 청크 누적
        print("AI: ", end="", flush=True)
        ai_chunk = None

        for token, metadata in graph.stream(
            {"messages": history + [{"role": "user", "content": user_input}], "use_tools": False},
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") in ANSWER_NODES and token.content:
                print(token.content, end="", flush=True)
                # AIMessageChunk끼리 + 연산으로 하나의 메시지로 병합
                ai_chunk = token if ai_chunk is None else ai_chunk + token

        # 스트리밍 1회로 저장소 갱신 (LLM 이중 호출 없음)
        if ai_chunk is not None:
            store[thread_id] = history + [HumanMessage(content=user_input), ai_chunk]

        print("\n")


if __name__ == "__main__":
    chat()
