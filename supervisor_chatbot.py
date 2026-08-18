"""
Supervisor 챗봇 — supervisor가 워커를 오케스트레이션하는 구조
메모리 + Supervisor 라우팅 + 도구 호출 + 스트리밍 + RAG + Reranker를 결합합니다.

사용자가 질문하면 의도를 파악해 플랜을 세우고, 플랜을 기반으로 supervisor가 워커를 오케스트레이션하며 처리하는 구조입니다.

그래프 구조:
  START → intent_classification ─┐
                                 ↓ (계획의 첫 단계로 라우팅)
              supervisor ─┬→ retrieve → rerank → rag_chat ─┐
                          ├→ tool_agent ────────────────────┤
                          │        (결과를 supervisor로 반환) ┘
                          └→ final_answer → END

intent_classification: 사용자의 최초 요청을 분석해 어떤 작업(rag/tools)이 어떤
            순서로 필요한지 "계획"을 세웁니다 (예: tools → rag → tools).
supervisor: intent_classification이 세운 계획을 한 단계씩 실행하며 워커
            (rag_chat / tool_agent)의 결과를 받아 다음 단계로 넘기고,
            계획을 모두 마치면 final_answer로 보냅니다.
메모리  : Python dict로 thread_id별 메시지를 직접 관리합니다.
RAG     : InMemoryVectorStore + OllamaEmbeddings로 추가 패키지 없이 구동합니다.
Reranker: LLM으로 각 후보를 0~10점 점수화해 상위 RERANK_TOP_K개만 선별합니다.
"""

import operator
import re
from datetime import datetime
from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain.agents import create_agent as create_react_agent
from typing_extensions import TypedDict

# ── 유틸 ──────────────────────────────────────────


def _format_tools(tools) -> str:
  """도구 이름과 설명(첫 줄)을 프롬프트용 문자열로 만듭니다."""
  return "\n".join(
    f"  · {t.name}: {t.description.strip().splitlines()[0]}" for t in tools
  )


def _parse_score(text: str) -> float:
  """thinking 태그를 제거하고 첫 번째 숫자를 0~10으로 클램핑합니다."""
  clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
  match = re.search(r"\b(\d+(?:\.\d+)?)\b", clean)
  if not match:
    return 0.0
  return min(10.0, max(0.0, float(match.group(1))))


def _parse_intent(text: str) -> tuple[list[str], list[str]]:
  """intent 분류 결과에서 (계획, 단계별 정리된 요청)을 뽑습니다.

  LLM은 아래 형식으로 답합니다.
    PLAN: tools -> rag -> tools
    REQUEST: <1단계 지시문> -> <2단계 지시문> -> <3단계 지시문>
  PLAN과 REQUEST 모두 화살표(->)로 구분되며, 같은 순서끼리 대응합니다.
  (REQUEST[i]는 PLAN[i] 단계에서 워커에게 넘길 지시문)
  계획은 PLAN 줄에서만 rag/tools를 추출해, 정리된 요청 본문의 단어에 오염되지 않습니다.
  """
  clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

  plan_match = re.search(r"PLAN\s*[:：]\s*(.+)", clean, flags=re.IGNORECASE)
  plan_line = plan_match.group(1) if plan_match else ""
  plan = re.findall(r"\b(rag|tools)\b", plan_line.lower())

  req_match = re.search(r"REQUEST\s*[:：]\s*(.+)", clean, flags=re.IGNORECASE | re.DOTALL)
  req_line = req_match.group(1).strip() if req_match else ""
  requests = [r.strip() for r in req_line.split("->") if r.strip()]
  return plan, requests

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
  messages: Annotated[list, add_messages] # 주고받은 메시지들
  candidates: list[dict]              # retrieve가 반환한 후보 (content, source)
  context: list[str]                  # rerank가 선별한 문서 내용
  sources: list[str]                  # 선별 문서의 출처
  findings: Annotated[list[str], operator.add]  # 워커들이 수집한 정보 (누적)
  resolved_query: str                 # 현재 단계에서 워커에게 넘길 지시문 (supervisor가 매 단계 갱신)
  resolved_queries: list[str]         # 단계별 지시문 (plan과 1:1 대응, REQUEST를 -> 로 분해)
  plan: list[str]                     # 수행할 작업 순서 (예: ["tools", "rag", "tools"])
  plan_cursor: int                    # 다음에 실행할 plan 인덱스
  next_action: str                    # 라우팅 결정: "rag" | "tools" | "final"
  steps: int                          # supervisor 반복 횟수 (무한 루프 방지)


# ── 노드 ───────────────────────────────────────────

INTENT_PROMPT = """당신은 멀티 에이전트 파이프라인의 첫 번째 노드인 '의도 분석 라우터'입니다.
사용자의 최신 요청과 대화 기록을 분석해 두 가지를 출력하세요.
(1) PLAN : 답을 완성하기까지 필요한 작업의 실행 순서
(2) REQUEST : 대화 맥락을 반영해 다음 노드들이 이해하기 쉽게 다시 쓴 요청 순서

[사용할 수 있는 작업]
- rag  : 지식 베이스에서 문서를 검색한다.
        → 개념·정의·사용법 설명, 사실 확인 등 "새로운 지식이 있어야 답할 수 있는" 요청에 사용
- tools: 아래 도구 중 하나를 실행한다.
        → 계산, 저장, 조회, 외부 동작 등 "무언가를 실행해야 완료되는" 요청에 사용
{tools_desc}

[판단 절차 — 아래 순서로 생각하되, 결과만 출력하세요]
1. 요청 복원: "그거", "방금 그 값", "이거" 같은 지시어와 생략된 목적어를
   대화 기록에서 찾아 채우고, 사용자가 진짜 원하는 것을 한 문장으로 정리한다.
2. 필요 작업 판별:
   - 답하는 데 새로운 지식·문서가 필요한가? → rag
   - 위 도구 목록에 있는 도구를 실행해야 하는가? → tools
   - 둘 다 필요한가? → 의존 관계에 맞는 순서로 나열 (예: 검색 결과를 도구에 넘겨야 하면 rag -> tools)
   - 인사·잡담이거나, 직전 답변의 요약·번역·형식 변경이거나,
     대화 기록만으로 답할 수 있는가? → none
3. 최소화 검증: 각 단계를 빼도 답이 완성되는지 되묻고, 불필요한 단계는 제거한다.
   확신이 없는 사실 질문은 추측으로 답하지 말고 rag를 넣는다.
   도구 목록에 없는 기능은 절대 tools로 계획하지 않는다.

[PLAN 규칙]
- 실제로 필요한 작업만, 수행할 순서대로 화살표(->)로 나열하세요.
- 같은 작업이 여러 번 필요하면 반복해서 적으세요. (예: tools -> tools)
- 아무 작업도 필요 없으면 none 이라고만 적으세요.
- 작업 이름(rag, tools, none) 외의 단어를 쓰지 마세요.

[REQUEST 규칙]
- PLAN의 각 단계와 1:1로 대응하는 지시문을 같은 순서로 " -> "로 구분해 나열하세요.
  PLAN의 단계 개수와 REQUEST의 지시문 개수는 반드시 같아야 합니다.
- 단, PLAN이 none이면 REQUEST는 최종 답변을 위한 지시문 한 개만 쓰세요.
- 각 지시문은 해당 단계 하나에서 할 일만 쓰세요. 여러 단계의 일을 한 지시문에 합치지 마세요.
- 지시어("그거", "이거" 등)를 모두 실제 내용으로 치환해, 대화 기록 없이 읽어도 수행할 수 있게 쓰세요.
  이전 단계의 결과를 이어받을 때는 "앞 단계의 검색 결과"처럼 명시적으로 지칭하세요.
- 각 지시문에 무엇을(검색어·대상), 어떻게(사용할 도구 이름과 인자)가 드러나게 쓰세요.
- 지시문 본문 안에는 화살표(->)를 쓰지 마세요. 화살표는 단계 구분에만 사용합니다.
- 사용자의 원래 의도를 추가·축소·변경하지 말고, 명확하게 풀어서만 쓰세요.
- 질문에 대한 답을 직접 쓰지 마세요. REQUEST는 '지시문'입니다.

[예시]
- 요청: "LangGraph가 뭐야?"
  PLAN: rag
  REQUEST: LangGraph가 무엇인지 지식 베이스에서 검색해 개념과 용도를 설명한다.
- 요청: "그거 검색해서 기억해 줘" (직전 대화 주제: LangChain, 장기 기억 도구가 있는 경우)
  PLAN: rag -> tools
  REQUEST: LangChain이 무엇인지 지식 베이스에서 검색한다. -> 검색 결과 요약을 장기 기억 도구로 저장한다.
- 요청: "고마워! 방금 답변 세 줄로 줄여줘"
  PLAN: none
  REQUEST: 직전 답변 내용을 세 줄로 요약해 다시 전달한다.
- 요청: "1300달러 원화로 얼마야?" (환율 조회 도구가 있는 경우)
  PLAN: tools
  REQUEST: 환율 조회 도구를 사용해 1300 USD를 KRW로 환산한 금액을 알려준다.
- 요청: "어제 매출 조회해서 팀 채널에 공유해 줘" (조회 도구와 메시지 전송 도구가 있는 경우)
  PLAN: tools -> tools
  REQUEST: 매출 조회 도구로 어제(날짜 명시) 매출을 조회한다. -> 앞 단계의 조회 결과를 메시지 전송 도구로 팀 채널에 공유한다.
  
[사용자 요청]
{question}

이제 아래 형식으로만, 다른 설명 없이 정확히 두 줄로 답하세요.
PLAN: <작업 순서 또는 none>
REQUEST: <다시 쓴 요청>"""


def intent_classification_node(state: State) -> State:
  """최초 요청의 의도를 분석해 (1) 작업 순서(계획)를 세우고,
  (2) 대화 맥락을 반영해 요청을 명확히 다시 써서 다음 노드들이 이해하도록 만듭니다.
  그 뒤 계획의 첫 단계로 라우팅합니다."""
  question = state["messages"][-1].content
  history = list(state["messages"][:-1])   # 현재 요청을 제외한 이전 대화 기록
  prompt = INTENT_PROMPT.format(question=question, tools_desc=TOOLS_DESC)
  # 대화 기록을 앞에 붙여 후속 질문("그거", "방금 그 값" 등)의 맥락을 파악합니다
  result = llm.invoke([SystemMessage(
    "당신은 사용자 요청의 의도를 분석하는 플래너입니다. "
    "요청을 완수하는 데 필요한 작업(rag/tools) 순서를 계획으로 세우고, "
    "대화 맥락을 반영해 요청을 명확한 지시문으로 다시 쓰는 것이 임무입니다. "
    "직접 답을 작성하지 말고, 지정된 형식(PLAN/REQUEST)으로만 출력하세요."
  )] + history + [{"role": "user", "content": prompt}])

  plan, requests = _parse_intent(result.content)
  # 단계별 지시문을 못 뽑은 경우 원본 질문 하나로 폴백
  resolved_queries = requests or [question]
  intent = " -> ".join(plan) if plan else "none (대화 기록만으로 답변)"
  first_action = plan[0] if plan else "final"
  # 첫 단계(plan[0])는 여기서 바로 라우팅하므로 대응하는 지시문을 활성화
  first_query = resolved_queries[0]

  print(f"\n  [intent] 의도 분류: {intent}", flush=True)
  print(f"  [intent] 의도 분류 단계별 요청 문구: {' -> '.join(resolved_queries)}", flush=True)
  return {
    "resolved_query": first_query,
    "resolved_queries": resolved_queries,
    "plan": plan,
    "plan_cursor": 1,          # 0번은 지금 라우팅하므로 다음 실행 위치는 1
    "next_action": first_action,
    "steps": 1,
  }


def supervisor_node(state: State) -> State:
  MAX_SUPERVISOR_STEPS = 10    # supervisor가 워커를 호출할 수 있는 최대 횟수

  """intent_classification이 세운 계획을 한 단계씩 실행하고, 다 마치면 final로 보냅니다.

  각 단계로 라우팅할 때 그 단계에 대응하는 지시문(resolved_queries[cursor])을
  resolved_query로 활성화해, 워커가 자기 단계에 맞는 요청만 보고 실행하도록 합니다.
  """
  plan = state.get("plan", [])
  cursor = state.get("plan_cursor", 0)
  steps = state.get("steps", 0)
  requests = state.get("resolved_queries", [])

  # 무한 루프 방지: 최대 횟수 도달 시 강제로 최종 답변으로
  if steps >= MAX_SUPERVISOR_STEPS:
    print(f"\n  [supervisor] 최대 반복({MAX_SUPERVISOR_STEPS}) 도달 → final", flush=True)
    return {"next_action": "final", "steps": steps + 1}

  # 계획에 남은 단계가 있으면 다음 작업으로, 없으면 최종 답변으로
  if cursor < len(plan):
    action = plan[cursor]
    # 이 단계에 대응하는 지시문을 활성화 (개수가 안 맞으면 마지막 지시문/직전 값으로 폴백)
    if cursor < len(requests):
      resolved_query = requests[cursor]
    elif requests:
      resolved_query = requests[-1]
    else:
      resolved_query = state.get("resolved_query", "")
    print(f"\n  [supervisor] 계획 {cursor + 1}/{len(plan)}단계 → {action}", flush=True)
    print(f"  [supervisor] 단계 요청 → {resolved_query}", flush=True)
    return {
      "next_action": action,
      "plan_cursor": cursor + 1,
      "resolved_query": resolved_query,
      "steps": steps + 1,
    }

  print("\n  [supervisor] 계획 완료 → final", flush=True)
  return {"next_action": "final", "steps": steps + 1}


def retrieve_node(state: State) -> State:
  """벡터 유사도로 후보 RETRIEVE_K개를 검색합니다."""
  # intent가 정리한 요청을 검색어로 사용 (지시어가 실제 내용으로 치환된 문장)
  question = state.get("resolved_query") or state["messages"][-1].content
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
  question = state.get("resolved_query") or state["messages"][-1].content
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
  """검색된 문서를 근거로 정보를 정리해 findings에 담아 supervisor로 반환합니다."""
  last_question = state.get("resolved_query") or state["messages"][-1].content

  # 문서를 찾지 못한 경우 → supervisor가 다른 행동을 판단하도록 그대로 알림
  if not state["context"]:
    print("  [rag] 관련 문서를 찾지 못함", flush=True)
    return {"findings": [f"[RAG 결과] '{last_question}'에 대한 관련 문서를 찾지 못했습니다."]}

  context_text = "\n\n".join(state["context"])
  rag_prompt = (
    f"다음 문서를 근거로 질문에 관련된 사실만 간결히 정리하세요.\n"
    f'문서로 답할 수 없으면 반드시 "{_UNANSWERABLE_PREFIX}"로 시작하세요.\n\n'
    f"[참고 문서]\n{context_text}\n\n"
    f"[질문]\n{last_question}"
  )
  response = llm.invoke([SystemMessage(
    "당신은 문서 기반 정보 정리 전문가입니다. "
    "제공된 참고 문서에 근거해서만 답하고, 문서에 없는 내용은 추측하지 마세요. "
    "질문과 관련된 사실을 간결하게 정리하세요."
  ), {"role": "user", "content": rag_prompt}])
  print("  [rag] 문서 기반 정보 수집 완료", flush=True)
  return {"findings": [f"[RAG 결과]\n{response.content}"]}


# 도구 에이전트 — create_react_agent를 노드로 래핑
_tool_agent = create_react_agent(llm, TOOLS, system_prompt=SystemMessage(
  "당신은 도구를 활용하는 유능한 어시스턴트입니다. "
  "도구 결과를 사실 그대로 사용하고 값을 임의로 지어내지 마세요."
))


def tool_agent_node(state: State) -> State:
  """ReAct 에이전트를 실행하고 그 결과를 findings에 담아 supervisor로 반환합니다."""
  # intent가 정리한 요청을 지시문으로 사용하고, 지금까지 수집한 정보를 함께 전달합니다.
  # (예: "tools → rag → tools"에서 마지막 tools가 앞선 RAG 결과를 기억할 수 있도록)
  query = state.get("resolved_query") or state["messages"][-1].content
  findings = state.get("findings", [])
  if findings:
    query = f"{query}\n\n[지금까지 수집한 정보]\n" + "\n\n".join(findings)
  agent_input = list(state["messages"][:-1]) + [HumanMessage(content=query)]

  final_msg = None
  try:
    for chunk in _tool_agent.stream(
        {"messages": agent_input},
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
  content = final_msg.content if final_msg else "도구 실행 결과가 없습니다."
  print("  [tools] 도구 실행 완료", flush=True)
  return {"findings": [f"[도구 결과]\n{content}"]}


def final_answer_node(state: State) -> State:
  lang = "한국어"
  """supervisor가 모은 findings와 대화 기록을 종합해 최종 답변을 생성합니다.

  findings가 있으면(RAG·도구를 거친 경우) 그 근거 중심으로 답하고,
  findings가 비어 있으면(대화형·메타 질문) 문서 거부 프레이밍 없이 대화 기록만으로 답합니다.
  """
  question = state["messages"][-1].content
  history = list(state["messages"][:-1])
  findings = state.get("findings", [])

  if findings:
    # RAG·도구 근거가 있는 경우: 수집한 정보 + 대화 기록을 종합
    findings_text = "\n\n".join(findings)
    final_prompt = (
      f'대화 기록과 수집한 정보 어디에도 근거가 없을 때만 "{_UNANSWERABLE_PREFIX}"로 시작하세요.\n\n'
      f"[수집한 정보]\n{findings_text}\n\n"
      f"[질문]\n{question}"
    )
    system_msg = SystemMessage(
      "당신은 친절하고 유능한 AI 어시스턴트입니다. "
      f'수집된 정보와 대화 기록을 모두 근거로 종합해 명확하고 자연스러운 "{lang}"로 최종 답변을 작성하세요. '
      "이전 질문·답변 등 대화 기록에 있는 내용도 유효한 근거이니 적극 활용하세요. "
      "대화 기록과 수집된 정보 어디에도 없는 내용만 지어내지 않으면 됩니다."
    )
  else:
    # 대화형·메타 질문: 문서 거부 프레이밍 없이 대화 기록만으로 답변
    final_prompt = (
      f'위 대화 기록을 근거로 아래 질문에 자연스럽고 친절한 "{lang}"로 답하세요.\n'
      f"이전 질문·답변 등 대화 기록에 있는 내용을 적극 활용하세요.\n\n"
      f"[질문]\n{question}"
    )
    system_msg = SystemMessage(
      "당신은 친절하고 유능한 AI 어시스턴트입니다. "
      f'위 대화 기록을 근거로 사용자의 질문에 자연스러운 "{lang}"로 답하세요. '
      "이전 질문·답변 등 대화 기록에 있는 내용도 유효한 근거이니 적극 활용하세요."
      "대화 기록 어디에도 없는 내용만 지어내지 않으면 됩니다."
    )

  response = llm.invoke([system_msg] + history + [{"role": "user", "content": final_prompt}])
  return {"messages": [response]}


def supervisor_route(
    state: State,
) -> Literal["retrieve_node", "tool_agent_node", "final_answer_node"]:
  action = state["next_action"]
  if action == "tools":
    return "tool_agent_node"
  if action == "rag":
    return "retrieve_node"
  return "final_answer_node"




# ── 대화 저장소 ────────────────────────────────────
# { thread_id: [HumanMessage, AIMessage, ...] }
store: dict[str, list[BaseMessage]] = {}

# ── 그래프 ─────────────────────────────────────────

graph = (
  StateGraph(State)
  .add_node("intent_classification_node", intent_classification_node)
  .add_node("supervisor_node", supervisor_node)
  .add_node("tool_agent_node", tool_agent_node)
  .add_node("retrieve_node", retrieve_node)
  .add_node("rerank_node", rerank_node)
  .add_node("rag_chat_node", rag_chat_node)
  .add_node("final_answer_node", final_answer_node)
  .add_edge(START, "intent_classification_node")
  # intent 분류 후 계획의 첫 단계로 라우팅 (supervisor_route 재사용)
  .add_conditional_edges("intent_classification_node", supervisor_route)
  .add_conditional_edges("supervisor_node", supervisor_route)
  .add_edge("retrieve_node", "rerank_node")
  .add_edge("rerank_node", "rag_chat_node")
  .add_edge("rag_chat_node", "supervisor_node")   # 워커 결과를 supervisor로 반환
  .add_edge("tool_agent_node", "supervisor_node")  # 워커 결과를 supervisor로 반환
  .add_edge("final_answer_node", END)
  .compile()
)

# ── 실행 ───────────────────────────────────────────

def chat(thread_id: str = "user-1"):
  print("Supervisor 챗봇 시작 (종료: quit)\n")

  while True:
    user_input = input("You: ").encode("utf-8", errors="replace").decode("utf-8").strip()
    if user_input.lower() in ("quit", "exit", "q"):
      break
    if not user_input:
      continue

    history = store.setdefault(thread_id, [])

    # stream_mode 리스트: (mode, chunk) 튜플로 yield됨
    print("AI", end="", flush=True)
    ai_chunk = None
    sources_captured: list[str] = []

    for mode, chunk in graph.stream(
        {
          "messages": history + [{"role": "user", "content": user_input}],
          "candidates": [],
          "context": [],
          "sources": [],
          "findings": [],
          "resolved_query": "",
          "resolved_queries": [],
          "plan": [],
          "plan_cursor": 0,
          "next_action": "",
          "steps": 0,
        },
        stream_mode=["messages", "updates"],
    ):
      if mode == "updates":
        # rerank가 여러 번 돌 수 있으므로 출처는 누적합니다
        if "rerank_node" in chunk:
          for src in chunk["rerank_node"].get("sources", []):
            if src not in sources_captured:
              sources_captured.append(src)
      elif mode == "messages":
        token, metadata = chunk
        # {"final_answer_node"} -> 최종 답변을 스트리밍으로 출력할 노드
        if metadata.get("langgraph_node") in {"final_answer_node"} and token.content:
          print(token.content, end="", flush=True)
          ai_chunk = token if ai_chunk is None else ai_chunk + token

    # 스트리밍된 답변이 _UNANSWERABLE_PREFIX로 시작하면 출처 표시 생략
    answer_text = ai_chunk.content.strip() if ai_chunk else ""
    if sources_captured and not answer_text.startswith(_UNANSWERABLE_PREFIX):
      print(f"\n출처: {' · '.join(sources_captured)}", end="", flush=True)

    if ai_chunk is not None:
      store[thread_id] = history + [HumanMessage(content=user_input), ai_chunk]

    print("\n")


if __name__ == "__main__":
  chat()
