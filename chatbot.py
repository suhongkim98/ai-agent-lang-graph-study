"""
RAG 챗봇 — 지금까지 배운 것을 하나로 + 문서 기반 정확한 답변
메모리 + 조건부 라우팅 + 도구 호출 + 스트리밍 + RAG + Reranker를 결합합니다.

사용자가 질문을 하면 그 질문만 보고 RAG 검색할지, MCP 도구를 호출할지 라우팅하는 단순한 구조입니다.

그래프 구조:
  START → router → tool_agent                          → END  (도구 필요 시)
                 → retrieve → rerank → rag_chat        → END  (일반 질문 → RAG + Rerank)

메모리  : MemorySaver 체크포인터로 thread_id별 메시지를 자동 관리합니다.
RAG     : InMemoryVectorStore + OllamaEmbeddings로 추가 패키지 없이 구동합니다.
Reranker: LLM으로 각 후보를 0~10점 점수화해 상위 RERANK_TOP_K개만 선별합니다.
"""

import re
from datetime import datetime
from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain.agents import create_agent as create_react_agent
from typing_extensions import TypedDict


# ── 유틸 ─────────────────────────────────

def _parse_score(text: str) -> float:
    """thinking 태그를 제거하고 첫 번째 숫자를 0~10으로 클램핑합니다."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", clean)
    if not match:
        return 0.0
    return min(10.0, max(0.0, float(match.group(1))))


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


def _format_tools(tools) -> str:
    """도구 이름과 설명(첫 줄)을 프롬프트용 문자열로 만듭니다."""
    return "\n".join(
        f"  · {t.name}: {t.description.strip().splitlines()[0]}" for t in tools
    )


TOOLS_DESC = _format_tools(TOOLS)

RETRIEVE_K = 6              # 벡터 검색 후보 수
RERANK_TOP_K = 3            # 리랭킹 후 선택 수
MIN_RERANK_SCORE = 5.0      # 이 점수 미만 문서는 컨텍스트에서 제외
_UNANSWERABLE_PREFIX = "문서에 없는 내용입니다"

# ── LLM & 임베딩 ──────────────────────────────────
# 임베딩 모델: ollama pull nomic-embed-text

llm = ChatOllama(model="qwen3:8b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)

# ── 지식 베이스 ────────────────────────────────────

_KNOWLEDGE = [
    Document(
        page_content=(
            "LangGraph는 LLM 기반 에이전트를 방향 그래프(DAG)로 구성하는 라이브러리입니다. "
            "State, Node, Edge 세 가지 개념으로 복잡한 워크플로를 표현합니다."
        ),
        metadata={"source": "langgraph_intro"},
    ),
    Document(
        page_content=(
            "LangGraph의 State는 그래프 전체에서 공유되는 데이터 구조입니다. "
            "TypedDict로 정의하고, 각 노드는 State를 받아 업데이트된 필드를 반환합니다."
        ),
        metadata={"source": "langgraph_state"},
    ),
    Document(
        page_content=(
            "RAG(Retrieval-Augmented Generation)는 외부 지식 베이스를 검색해 "
            "LLM 답변의 정확도와 최신성을 높이는 기법입니다. "
            "retrieve → generate 두 단계로 구성됩니다."
        ),
        metadata={"source": "rag_intro"},
    ),
    Document(
        page_content=(
            "Ollama는 로컬에서 LLM을 실행할 수 있는 도구입니다. "
            "qwen3, llama, mistral, gemma 등 다양한 오픈소스 모델을 지원합니다."
        ),
        metadata={"source": "ollama_intro"},
    ),
    Document(
        page_content=(
            "LangChain은 LLM 애플리케이션 개발 프레임워크입니다. "
            "체인(Chain), 프롬프트(Prompt), 도구(Tool), 메모리(Memory) 추상화를 제공합니다."
        ),
        metadata={"source": "langchain_intro"},
    ),
    Document(
        page_content=(
            "add_messages는 LangGraph의 리듀서로, 메시지를 덮어쓰지 않고 누적합니다. "
            "State에 Annotated[list, add_messages]로 선언해 사용합니다."
        ),
        metadata={"source": "add_messages"},
    ),
    Document(
        page_content=(
            "create_react_agent는 도구 선택 → 호출 → 결과 확인 → 재판단 루프를 "
            "자동으로 구성하는 LangGraph 유틸리티입니다."
        ),
        metadata={"source": "react_agent"},
    ),
    Document(
        page_content=(
            "Human-in-the-Loop(HITL)는 그래프 실행 도중 사람의 승인·수정을 끼워넣는 패턴입니다. "
            "interrupt()와 Command(resume=...)로 일시정지와 재개를 구현합니다."
        ),
        metadata={"source": "hitl"},
    ),
    Document(
        page_content=(
            "멀티 에이전트 패턴에서는 Supervisor가 질문을 분석해 전문 에이전트에게 위임합니다. "
            "각 전문 에이전트는 독립적인 create_react_agent 서브그래프로 구성됩니다."
        ),
        metadata={"source": "multi_agent"},
    ),
    Document(
        page_content=(
            "스트리밍 모드는 두 가지입니다. "
            "updates 모드는 노드 완료 시 State 변경분을 반환하고, "
            "messages 모드는 LLM 토큰을 실시간으로 반환합니다."
        ),
        metadata={"source": "streaming"},
    ),
]

vector_store.add_documents(_KNOWLEDGE)

# ── 상태 ───────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    use_tools: bool
    candidates: list[dict]      # retrieve가 반환한 후보 (content, source)
    context: list[str]          # rerank가 선별한 문서 내용
    sources: list[str]          # 선별 문서의 출처


# ── 노드 ───────────────────────────────────────────

ROUTE_PROMPT = """사용자 메시지에 다음 도구 중 하나라도 필요하면 'tools'라고만 답하고, 아니면 'chat'이라고만 답하세요.
{tools_desc}

메시지: {message}
분류:"""


def router_node(state: State) -> State:
    last = state["messages"][-1].content
    result = llm.invoke([{"role": "user", "content": ROUTE_PROMPT.format(
        message=last, tools_desc=TOOLS_DESC)}])
    use_tools = "tools" in result.content.lower()
    return {"use_tools": use_tools}


def retrieve_node(state: State) -> State:
    """벡터 유사도로 후보 RETRIEVE_K개를 검색합니다."""
    question = state["messages"][-1].content
    docs = vector_store.similarity_search(question, k=RETRIEVE_K)
    return {
        "candidates": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in docs
        ]
    }


def rerank_node(state: State) -> State:
    _SCORE_PROMPT = """\
    질문과 문서의 관련성을 0~10점으로 평가하세요.
    숫자 하나만 출력하세요.

    질문: {question}
    문서: {content}
    점수:"""

    """LLM으로 각 후보에 0~10점을 부여하고 상위 RERANK_TOP_K개를 선별합니다."""
    question = state["messages"][-1].content
    scored = []
    for cand in state["candidates"]:
        result = llm.invoke(
            _SCORE_PROMPT.format(question=question, content=cand["content"])
        )
        scored.append({**cand, "score": _parse_score(result.content)})

    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
    top = [d for d in ranked if d["score"] >= MIN_RERANK_SCORE][:RERANK_TOP_K]
    return {
        "context": [d["content"] for d in top],
        "sources": list(dict.fromkeys(d["source"] for d in top)),
    }


def rag_chat_node(state: State) -> State:
    RAG_SYSTEM = SystemMessage(
        "당신은 문서 기반 질의응답 전문가입니다. "
        "제공된 참고 문서와 대화 기록에 근거해서만 답하고, 그 밖의 내용은 추측하지 마세요. "
        "질문에 친절하고 명확한 한국어로 답변하세요."
    )

    """검색된 문서를 컨텍스트로 포함해 LLM이 답변합니다.
    문서가 없으면 대화 기록에서 답변을 시도하고, 기록에도 없으면 답변불가를 반환합니다."""
    last_question = state["messages"][-1].content
    history = list(state["messages"][:-1])

    # 문서가 없을 경우 → 대화 기록에서만 답변 시도 (일반 지식 사용 금지)
    if not state["context"]:
        history_prompt = (
            f"아래 대화 기록에 근거해 질문에 답하세요.\n"
            f'대화 기록에서 찾을 수 없으면 반드시 "{_UNANSWERABLE_PREFIX}"로 시작하세요.\n\n'
            f"[질문]\n{last_question}"
        )
        response = llm.invoke([RAG_SYSTEM] + history + [{"role": "user", "content": history_prompt}])
        return {"messages": [response]}

    # 검색된 문서가 있다면
    context_text = "\n\n".join(state["context"])
    rag_prompt = (
        f"다음 문서와 대화 기록을 참고해 질문에 답하세요.\n"
        f'문서와 대화 기록으로 답할 수 없으면 반드시 "{_UNANSWERABLE_PREFIX}"로 시작하세요.\n\n'
        f"[참고 문서]\n{context_text}\n\n"
        f"[질문]\n{last_question}"
    )
    response = llm.invoke([RAG_SYSTEM] + history + [{"role": "user", "content": rag_prompt}])
    return {"messages": [response]}


# 도구 에이전트 — create_react_agent를 노드로 래핑
_tool_agent = create_react_agent(llm, TOOLS, system_prompt=SystemMessage(
    "당신은 도구를 활용하는 유능한 어시스턴트입니다. "
    "도구 결과를 사실 그대로 사용하고 값을 임의로 지어내지 마세요."
))


def tool_agent_node(state: State) -> State:
    """ReAct 에이전트를 스트리밍으로 실행하며 도구 호출 과정을 출력합니다."""
    final_msg = None
    try:
        for chunk in _tool_agent.stream(
            {"messages": state["messages"]},
            stream_mode="updates",
            config={"recursion_limit": 10},
        ):
            for _node, delta in chunk.items():
                for msg in delta.get("messages", []):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        names = ", ".join(tc["name"] for tc in msg.tool_calls)
                        print(f"\n  → 도구 호출: {names}", flush=True)
                    elif hasattr(msg, "name") and msg.name:  # ToolMessage (도구 결과)
                        print(f"  → [{msg.name}] {msg.content}", flush=True)
                    elif msg.content:
                        final_msg = msg
    except GraphRecursionError:
        print("\n  → [경고] 도구 호출 반복 한도(10) 초과", flush=True)
        if final_msg is None:
            final_msg = AIMessage(content="도구 실행 중 반복 한도를 초과했습니다.")
    return {"messages": [final_msg] if final_msg else []}


def route(state: State) -> Literal["tool_agent_node", "retrieve_node"]:
    return "tool_agent_node" if state["use_tools"] else "retrieve_node"




# ── 그래프 ─────────────────────────────────────────

memory = MemorySaver()

graph = (
    StateGraph(State)
    .add_node("router_node", router_node)
    .add_node("tool_agent_node", tool_agent_node)
    .add_node("retrieve_node", retrieve_node)
    .add_node("rerank_node", rerank_node)
    .add_node("rag_chat_node", rag_chat_node)
    .add_edge(START, "router_node")
    .add_conditional_edges("router_node", route)
    .add_edge("retrieve_node", "rerank_node")
    .add_edge("rerank_node", "rag_chat_node")
    .add_edge("tool_agent_node", END)
    .add_edge("rag_chat_node", END)
    .compile(checkpointer=memory)
)

ANSWER_NODES = {"tool_agent_node", "rag_chat_node"}


# ── 실행 ───────────────────────────────────────────

def chat(thread_id: str = "user-1"):
    print("RAG 챗봇 시작 (종료: quit)\n")

    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("You: ").encode("utf-8", errors="replace").decode("utf-8").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        # stream_mode 리스트: (mode, chunk) 튜플로 yield됨
        print("AI", end="", flush=True)
        ai_chunk = None
        sources_captured: list[str] = []

        for mode, chunk in graph.stream(
            {
                "messages": [{"role": "user", "content": user_input}],
                "use_tools": False,
                "candidates": [],
                "context": [],
                "sources": [],
            },
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "updates":
                if "rerank_node" in chunk:
                    sources_captured = chunk["rerank_node"].get("sources", [])
                elif "tool_agent_node" in chunk:
                    msgs = chunk["tool_agent_node"].get("messages", [])
                    if msgs and msgs[-1].content:
                        ai_chunk = msgs[-1]
            elif mode == "messages":
                token, metadata = chunk
                if metadata.get("langgraph_node") in ANSWER_NODES and token.content:
                    print(token.content, end="", flush=True)
                    ai_chunk = token if ai_chunk is None else ai_chunk + token

        # 스트리밍된 답변이 _UNANSWERABLE_PREFIX로 시작하면 출처 표시 생략
        answer_text = ai_chunk.content.strip() if ai_chunk else ""
        if sources_captured and not answer_text.startswith(_UNANSWERABLE_PREFIX):
            print(f"\n출처: {' · '.join(sources_captured)}", end="", flush=True)

        print("\n")


if __name__ == "__main__":
    chat()