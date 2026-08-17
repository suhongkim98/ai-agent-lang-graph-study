"""
09. RAG 기반 답변 — LangGraph로 Retrieval-Augmented Generation 구현
문서를 InMemoryVectorStore에 저장하고, 검색된 컨텍스트를 기반으로 LLM이 답변합니다.
추가 패키지 없이 langchain-core + langchain-ollama만으로 동작합니다.

그래프 구조:
  START → retrieve → generate → END

State:
  question : str           — 사용자 질문
  documents: list[str]     — 검색된 문서 내용
  sources  : list[str]     — 검색된 문서의 출처 (metadata["source"])
  answer   : str           — 최종 답변 (출처 포함)
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── State ─────────────────────────────────────────────
class State(TypedDict):
    question: str
    documents: list[str]
    sources: list[str]
    answer: str


# ── 모델 초기화 ────────────────────────────────────────
# 임베딩 모델: ollama pull nomic-embed-text 로 받아야 합니다.
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


# ── 노드 정의 ─────────────────────────────────────────

def retrieve(state: State) -> State:
    """질문과 유사한 문서를 벡터 스토어에서 가져옵니다."""
    results = vector_store.similarity_search(state["question"], k=3)
    # dict.fromkeys로 순서를 유지하면서 중복 제거
    return {
        "documents": [doc.page_content for doc in results],
        "sources": list(dict.fromkeys(
            doc.metadata.get("source", "unknown") for doc in results
        )),
    }


_UNANSWERABLE_PREFIX = "문서에 없는 내용입니다"


def generate(state: State) -> State:
    """검색된 문서를 컨텍스트로 LLM에 전달해 답변을 생성합니다.
    문서로 답할 수 없으면 출처를 표시하지 않습니다."""
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
    .add_node("generate", generate)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "generate")
    .add_edge("generate", END)
    .compile()
)


# ── 공개 API ──────────────────────────────────────────

def ask(question: str) -> str:
    """질문을 받아 RAG 파이프라인을 실행하고 답변을 반환합니다."""
    result = graph.invoke({"question": question})
    return result["answer"]


# ── 데모 ─────────────────────────────────────────────

def demo():
    questions = [
        "LangGraph란 무엇인가요?",
        "RAG는 어떤 기법인가요?",
        "InMemoryVectorStore를 쓰는 이유가 뭔가요?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask(q)}")
        print()


def chat():
    """대화 루프 (종료: quit)"""
    print("RAG 채팅 시작 (종료: quit)\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        print(f"A: {ask(question)}\n")


if __name__ == "__main__":
    chat()
