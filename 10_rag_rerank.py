"""
10. RAG + Reranker — 검색 → 재정렬 → 생성
벡터 검색으로 후보를 넓게 뽑은 뒤 LLM이 관련성을 점수화해 상위 문서만 선별합니다.
rerank 후 통과 문서가 없으면 LLM 호출 없이 즉시 답변불가를 반환

그래프 구조:
  START → retrieve → rerank → generate → END

  retrieve : 벡터 유사도로 후보 RETRIEVE_K(6)개 검색
  rerank   : LLM이 각 후보에 0~10점 부여 → 상위 RERANK_TOP_K(3)개 선별
  generate : 선별된 문서로 답변 + 출처 표시

[프로덕션 참고]
  실제 서비스에서는 LLM 대신 전용 cross-encoder 모델을 reranker로 씁니다.
  예) cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers)
  이 예제는 추가 패키지 없이 개념을 익히는 용도로 LLM reranker를 사용합니다.

State:
  question  : str         — 사용자 질문
  candidates: list[dict]  — retrieve가 반환한 후보 (content, source)
  documents : list[str]   — rerank가 선별한 최종 문서
  sources   : list[str]   — 선별 문서의 출처
  answer    : str         — 최종 답변 (출처 포함)
"""

import re

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

RETRIEVE_K = 6          # 벡터 검색 후보 수 (넓게)
RERANK_TOP_K = 3        # 리랭킹 후 선택 수 (좁게)
MIN_RERANK_SCORE = 5.0  # 이 점수 미만 문서는 컨텍스트에서 제외


# ── State ─────────────────────────────────────────────
class State(TypedDict):
    question: str
    candidates: list[dict]   # {"content": str, "source": str}
    documents: list[str]     # rerank 후 선별된 문서 본문
    sources: list[str]       # 선별 문서의 출처
    answer: str


# ── 모델 초기화 ────────────────────────────────────────
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)
llm = ChatOllama(model="qwen3:8b", temperature=0)


# ── 샘플 문서 ─────────────────────────────────────────
_DOCS = [
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
            "InMemoryVectorStore는 langchain-core에 내장된 벡터 스토어입니다. "
            "별도 외부 서비스 없이 임베딩 기반 유사도 검색을 사용할 수 있습니다."
        ),
        metadata={"source": "vectorstore_intro"},
    ),
    Document(
        page_content=(
            "OllamaEmbeddings는 Ollama가 제공하는 임베딩 모델을 LangChain에서 사용하는 클래스입니다. "
            "nomic-embed-text, mxbai-embed-large 등의 모델과 함께 사용합니다."
        ),
        metadata={"source": "ollama_embeddings"},
    ),
]

vector_store.add_documents(_DOCS)


# ── 리랭킹 프롬프트 ────────────────────────────────────
_SCORE_PROMPT = """\
질문과 문서의 관련성을 0~10점으로 평가하세요.
숫자 하나만 출력하세요.

질문: {question}
문서: {content}
점수:"""


def _parse_score(text: str) -> float:
    """LLM 응답에서 점수를 추출해 0~10으로 클램핑합니다.
    qwen3 같은 thinking 모델은 <think>...</think>를 먼저 제거합니다."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", clean)
    if not match:
        return 0.0
    return min(10.0, max(0.0, float(match.group(1))))


# ── 노드 정의 ─────────────────────────────────────────

def retrieve(state: State) -> State:
    """벡터 유사도로 후보 RETRIEVE_K개를 검색합니다."""
    results = vector_store.similarity_search(state["question"], k=RETRIEVE_K)
    return {
        "candidates": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in results
        ]
    }


def rerank(state: State) -> State:
    """LLM으로 각 후보에 0~10점을 부여하고 상위 RERANK_TOP_K개를 선별합니다."""
    scored = []
    print(f"\n[rerank] 후보 {len(state['candidates'])}개 점수화 중...")

    for cand in state["candidates"]:
        result = llm.invoke(
            _SCORE_PROMPT.format(
                question=state["question"],
                content=cand["content"],
            )
        )
        score = _parse_score(result.content)
        scored.append({**cand, "score": score})

    # 점수 내림차순 정렬 → MIN_RERANK_SCORE 이상만 필터링 → 상위 k개 선택
    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
    filtered = [d for d in ranked if d["score"] >= MIN_RERANK_SCORE]
    top = filtered[:RERANK_TOP_K]

    print("[rerank] 점수 결과:")
    for item in ranked:
        if item["score"] >= MIN_RERANK_SCORE and item in top:
            marker = "✓"
        elif item["score"] < MIN_RERANK_SCORE:
            marker = "✗"  # 임계값 미달
        else:
            marker = " "
        print(f"  {marker} {item['score']:4.1f}점  {item['source']}")

    return {
        "documents": [d["content"] for d in top],
        "sources": list(dict.fromkeys(d["source"] for d in top)),
    }


_UNANSWERABLE_PREFIX = "문서에 없는 내용입니다"


def generate(state: State) -> State:
    """선별된 문서로 답변을 생성합니다.
    rerank 후 통과 문서가 없으면 LLM 호출 없이 즉시 답변불가를 반환합니다."""
    if not state["documents"]:
        return {"answer": f"{_UNANSWERABLE_PREFIX}."}
    context = "\n\n".join(state["documents"])
    prompt = (
        "다음 문서를 참고해 질문에 답하세요.\n"
        f'문서로 답할 수 없으면 반드시 "{_UNANSWERABLE_PREFIX}"로 시작하세요.\n\n'
        f"[문서]\n{context}\n\n"
        f"[질문]\n{state['question']}"
    )
    response = llm.invoke(prompt)
    can_answer = not response.content.strip().startswith(_UNANSWERABLE_PREFIX)
    suffix = f"\n\n출처: {' · '.join(state['sources'])}" if can_answer else ""
    return {"answer": f"{response.content}{suffix}"}


# ── 그래프 조립 ─────────────────────────────────────────
graph = (
    StateGraph(State)
    .add_node("retrieve", retrieve)
    .add_node("rerank", rerank)
    .add_node("generate", generate)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "rerank")
    .add_edge("rerank", "generate")
    .add_edge("generate", END)
    .compile()
)


# ── 공개 API ──────────────────────────────────────────

def ask(question: str) -> str:
    result = graph.invoke({"question": question})
    return result["answer"]


# ── 데모 ─────────────────────────────────────────────

def demo():
    questions = [
        "RAG란 무엇인가요?",
        "LangGraph의 State를 어떻게 정의하나요?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
        print("-" * 60)


def chat():
    """대화 루프 (종료: quit)"""
    print("RAG + Reranker 채팅 시작 (종료: quit)\n")
    while True:
        question = input("Q: ").encode("utf-8", errors="replace").decode("utf-8").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        print(f"A: {ask(question)}\n")


if __name__ == "__main__":
    demo()
