"""
08. 멀티 에이전트 — Supervisor 패턴
Supervisor가 질문을 분석해 전문 에이전트에게 위임합니다.

구조:
  START → supervisor → math_agent   → supervisor → ... → END
                     → research_agent
                     → FINISH (직접 답변)

각 전문 에이전트는 독립적인 서브그래프로, supervisor가 결과를 받아
추가 위임이 필요한지 판단합니다.
"""

from typing import Annotated, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.agents import create_agent as create_react_agent
from typing_extensions import TypedDict


llm = ChatOllama(model="qwen3:8b", temperature=0)

MEMBERS = ["math_agent", "research_agent"]

# --- 전문 에이전트 도구 ---

@tool
def calculate(expression: str) -> str:
    """수식을 계산합니다. Args: expression: 계산할 수식"""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "허용되지 않는 문자"
        return f"{expression} = {eval(expression)}"  # noqa: S307
    except Exception as e:
        return f"오류: {e}"


@tool
def search_knowledge(query: str) -> str:
    """지식 베이스에서 정보를 검색합니다. Args: query: 검색 쿼리"""
    # 실제로는 외부 검색 API나 벡터DB를 사용
    kb = {
        "python": "Python은 1991년 Guido van Rossum이 만든 고급 프로그래밍 언어입니다.",
        "langgraph": "LangGraph는 LLM 애플리케이션을 그래프 구조로 구성하는 프레임워크입니다.",
        "mcp": "MCP(Model Context Protocol)는 Anthropic이 설계한 LLM-도구 연결 표준 프로토콜입니다.",
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return f"'{query}'에 대한 정보를 찾지 못했습니다."


# --- 전문 에이전트 (서브그래프) ---

math_agent = create_react_agent(
    llm,
    [calculate],
    system_prompt=SystemMessage("당신은 수학 전문가입니다. 계산 도구를 활용해 정확히 계산하세요."),
)

research_agent = create_react_agent(
    llm,
    [search_knowledge],
    system_prompt=SystemMessage("당신은 리서치 전문가입니다. 지식 베이스를 검색해 정확한 정보를 제공하세요."),
)

# --- Supervisor 상태 ---

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str  # 다음 에이전트 or "FINISH"


SUPERVISOR_PROMPT = f"""당신은 팀 리더입니다. 사용자 요청을 분석해 적절한 팀원에게 위임하거나 직접 답변하세요.

팀원:
- math_agent: 수학 계산, 숫자 관련 문제
- research_agent: 사실 조회, 지식 검색

규칙:
- 팀원에게 위임할 때는 팀원 이름만 답하세요: {MEMBERS}
- 직접 답변할 수 있거나 팀원 결과로 충분하면: FINISH
"""


def supervisor_node(state: SupervisorState) -> SupervisorState:
    messages = [SystemMessage(SUPERVISOR_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    content = response.content.strip()

    # 팀원 이름 파싱
    next_agent = "FINISH"
    for member in MEMBERS:
        if member in content:
            next_agent = member
            break

    print(f"  [Supervisor] → {next_agent}")

    # FINISH: supervisor가 직접 답변 생성
    if next_agent == "FINISH":
        answer = llm.invoke([SystemMessage("당신은 친절하고 정확한 AI 어시스턴트입니다.")] + state["messages"])
        return {"next": next_agent, "messages": [HumanMessage(content=answer.content, name="supervisor")]}

    return {"next": next_agent}


def agent_node(agent, name: str):
    """전문 에이전트를 Supervisor State 노드로 래핑합니다."""
    def node(state: SupervisorState) -> SupervisorState:
        result = agent.invoke({"messages": state["messages"]})
        last = result["messages"][-1]
        print(f"  [{name}] {last.content[:80]}...")
        return {"messages": [HumanMessage(content=last.content, name=name)]}
    return node


def router(state: SupervisorState) -> Literal["math_agent", "research_agent", "__end__"]:
    n = state["next"]
    return "__end__" if n == "FINISH" else n


# --- 그래프 조립 ---

graph = (
    StateGraph(SupervisorState)
    .add_node("supervisor", supervisor_node)
    .add_node("math_agent", agent_node(math_agent, "math_agent"))
    .add_node("research_agent", agent_node(research_agent, "research_agent"))
    .add_edge(START, "supervisor")
    .add_conditional_edges("supervisor", router)
    # 전문 에이전트 완료 후 다시 supervisor로
    .add_edge("math_agent", "supervisor")
    .add_edge("research_agent", "supervisor")
    .compile()
)


def ask(question: str):
    print(f"\nQ: {question}")
    print("-" * 40)
    result = graph.invoke({"messages": [{"role": "user", "content": question}], "next": ""})
    # 마지막 에이전트(또는 supervisor 직접) 메시지 출력
    for msg in reversed(result["messages"]):
        if hasattr(msg, "name") and msg.name in (*MEMBERS, "supervisor") and msg.content:
            print(f"\n최종 답변: {msg.content}")
            break


if __name__ == "__main__":
    ask("1234 곱하기 5678은?")
    ask("LangGraph가 뭐야?")
    ask("파이썬을 만든 사람이 누구야?")
