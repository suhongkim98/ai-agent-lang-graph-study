# LangGraph 로컬 스터디

API 키 없이 로컬 LLM(Ollama)으로 LangGraph의 핵심 개념을 단계별로 학습합니다.

---

## 환경 구성

### 필요 조건

| 도구 | 버전 | 역할 |
|------|------|------|
| Python | 3.14+ | 런타임 |
| [uv](https://docs.astral.sh/uv/) | 최신 | 패키지 관리 / 가상환경 |
| [Ollama](https://ollama.com) | 최신 | 로컬 LLM 서버 |

### 1단계 — Ollama 설치 및 모델 다운로드

```bash
# macOS
brew install ollama

# LLM 모델 (도구 호출 지원)
ollama pull qwen3:8b

# 임베딩 모델 (RAG 예제에서 사용)
ollama pull nomic-embed-text
```

### 2단계 — uv 설치 및 프로젝트 패키지 설치

```bash
# uv 설치 (이미 있으면 건너뜀)
pip install uv

# 프로젝트 의존성 설치 (.venv 자동 생성)
uv sync
```

### 3단계 — Ollama 서버 실행

예제를 실행하기 전에 Ollama 서버가 실행 중이어야 합니다.

```bash
ollama serve
```

> 터미널을 별도로 열어두거나, 백그라운드로 실행하세요. macOS에서는 앱을 설치하면 자동 실행됩니다.

### 설치된 패키지

| 패키지 | 역할 |
|--------|------|
| `langgraph` | 그래프 기반 LLM 오케스트레이션 |
| `langchain-ollama` | Ollama ↔ LangChain 연결 |
| `langchain-mcp-adapters` | MCP 서버 도구를 LangChain으로 변환 |
| `mcp` | MCP 서버 구현 |
| `numpy` | `InMemoryVectorStore` 코사인 유사도 계산에 필요 |

---

## 예제 파일

### 01. 기본 채팅 — `01_basic_chat.py`

LangGraph의 가장 단순한 형태. 상태(State), 노드(Node), 그래프(Graph)의 3요소를 익힙니다.

```
START → chatbot → END
```

**핵심 개념**
- `State`: 그래프 전체가 공유하는 데이터 구조 (여기선 메시지 목록)
- `add_messages`: 메시지를 덮어쓰지 않고 누적하는 리듀서
- `StateGraph`: 노드와 엣지를 연결해 그래프를 조립하는 빌더

```bash
uv run python 01_basic_chat.py
```

대화 루프가 실행됩니다. `quit` 입력 시 종료.

---

### 02. 커스텀 도구 + ReAct 에이전트 — `02_tools.py`

`@tool` 데코레이터로 Python 함수를 LLM이 호출할 수 있는 도구로 등록하고,
`create_react_agent`로 "도구 선택 → 호출 → 결과 확인 → 재판단" 루프를 구성합니다.

**등록된 도구**
- `get_current_time`: 현재 시각 반환
- `calculator`: 수식 계산
- `word_counter`: 단어/글자 수 계산

```bash
uv run python 02_tools.py
```

3개의 쿼리를 자동으로 실행하며 도구 호출 흐름을 출력합니다.

---

### 03. MCP 서버 / 클라이언트 — `03_mcp_server.py` + `03_mcp_client.py`

MCP(Model Context Protocol)는 LLM에 도구를 제공하는 표준 프로토콜입니다.
서버와 클라이언트를 분리해, 서버가 어떤 언어로 만들어졌든 연결할 수 있습니다.

**`03_mcp_server.py`** — stdio 방식 MCP 서버 (직접 실행하지 않음)
- `get_time`: 현재 시각
- `list_files`: 디렉토리 파일 목록
- `read_file`: 파일 내용 읽기

**`03_mcp_client.py`** — LangGraph 에이전트 + MCP 연결
- `MultiServerMCPClient`가 서버를 subprocess로 띄워 stdio로 통신
- MCP 도구를 LangChain Tool 형식으로 자동 변환

```bash
# 클라이언트만 실행하면 서버가 자동으로 subprocess로 실행됨
uv run python 03_mcp_client.py
```

---

### 04. 멀티턴 메모리 — `04_memory.py`

`MemorySaver` 대신 Python `dict`로 `thread_id`별 메시지 목록을 직접 관리합니다.
그래프는 체크포인터 없이 컴파일하고, 호출 전에 이전 메시지를 직접 주입합니다.

**핵심 개념**
- `store: dict[str, list[BaseMessage]]`: thread_id → 메시지 목록 저장소
- `thread_id`: 대화 세션 식별자. 다른 ID면 독립된 대화
- `invoke(thread_id, message)`: 히스토리를 꺼내 주입하고, 결과를 다시 저장

**MemorySaver 대비 장점**
- `store` dict를 직접 접근해 기록 조회(`get_history`) / 삭제(`clear_history`) 가능
- JSON 직렬화 등 외부 저장소 연동 시 확장이 쉬움

```bash
uv run python 04_memory.py
```

`thread-A`에서 이름을 기억하고, `thread-B`에서는 모르는 것을 확인합니다.

---

### 05. 조건부 분기 — `05_conditional.py`

LLM이 질문 유형을 판단해 다른 노드로 라우팅합니다.

```
START → classify → (math_node | search_node | chat_node) → END
```

**핵심 개념**
- `add_conditional_edges`: 라우터 함수 결과에 따라 다음 노드를 동적으로 결정
- 각 노드는 서로 다른 시스템 프롬프트를 가진 LLM을 실행

```bash
uv run python 05_conditional.py
```

수학 계산 / 사실 조회 / 감정 대화 3가지 케이스를 자동으로 분기합니다.

---

### 06. 스트리밍 — `06_streaming.py`

LangGraph의 두 가지 스트리밍 모드를 시연합니다.

| 모드 | 단위 | 용도 |
|------|------|------|
| `updates` | 노드 완료 시 State 변경분 | 실행 흐름 모니터링 |
| `messages` | LLM 토큰 | 타이핑 효과 UI |

```bash
uv run python 06_streaming.py
```

---

### 07. Human-in-the-Loop — `07_human_in_loop.py`

그래프 실행 도중 사람의 승인/수정을 끼워넣는 패턴입니다.

```
START → draft_node → [INTERRUPT] → (approve → finalize | 수정 지시 → revise_node → [INTERRUPT] → ...)
```

**핵심 개념**
- `interrupt(value)`: 노드 실행을 일시 정지하고 값을 외부에 노출
- `Command(resume=value)`: 정지된 그래프를 재개
- `interrupt_before=[...]`: 특정 노드 진입 직전에 자동으로 정지
- MemorySaver 필수: 정지 상태를 thread_id에 저장해야 재개 가능

```bash
uv run python 07_human_in_loop.py
```

이메일 주제를 입력하면 LLM이 초안을 작성합니다.
- `approve` 입력 → 최종 확정
- 다른 입력 → 수정 지시로 처리 후 재작성

---

### 08. 멀티 에이전트 — `08_multi_agent.py`

Supervisor가 질문을 분석해 전문 에이전트에게 위임하는 패턴입니다.

```
START → supervisor → (math_agent | research_agent | FINISH) → supervisor → ... → END
```

**에이전트 역할**

| 에이전트 | 도구 | 담당 |
|----------|------|------|
| `supervisor` | 없음 | 라우팅 + 직접 답변 |
| `math_agent` | `calculate` | 수식 계산 |
| `research_agent` | `search_knowledge` | 지식 조회 |

**핵심 개념**
- 각 전문 에이전트는 독립적인 `create_react_agent` 서브그래프
- 전문 에이전트 완료 후 다시 supervisor로 복귀해 추가 위임 여부 판단

```bash
uv run python 08_multi_agent.py
```

---

### 09. RAG 기반 답변 — `09_rag.py`

문서를 벡터 스토어에 저장하고, 질문과 유사한 문서를 검색해 LLM이 정확하게 답변합니다.
추가 패키지 없이 `langchain-core` + `langchain-ollama`만으로 동작합니다.

```
START → retrieve → generate → END
```

**핵심 개념**
- `InMemoryVectorStore`: `langchain-core` 내장 벡터 스토어, 외부 서비스 불필요
- `OllamaEmbeddings`: 로컬 임베딩 모델(`nomic-embed-text`)로 문서 인덱싱
- `similarity_search(query, k=3)`: 질문과 가장 유사한 문서 k개 반환
- RAG 패턴: `retrieve` 노드(문서 검색) → `generate` 노드(컨텍스트 기반 답변)

```bash
uv run python 09_rag.py
```

대화 루프가 실행됩니다. `quit` 입력 시 종료.

---

### 10. RAG + Reranker — `10_rag_rerank.py`

벡터 검색으로 후보를 넓게 뽑은 뒤, LLM이 관련성을 점수화해 상위 문서만 선별합니다.

```
START → retrieve(k=6) → rerank → generate → END
```

**핵심 개념**
- `retrieve`: 임베딩 유사도로 후보 6개를 넓게 검색
- `rerank`: LLM이 각 후보에 0~10점 부여 → 상위 3개 선별
- `generate`: 선별된 3개 문서로 답변 생성 + 출처 표시
- 점수 결과가 터미널에 출력되어 선별 과정이 눈에 보임

| 단계 | 기준 | 특징 |
|------|------|------|
| retrieve | 임베딩 각도(cosine) | 단어 유사성 기반, 빠름 |
| rerank | LLM 의미 이해 | 질문 의도 파악, 정확함 |

> 프로덕션에서는 LLM 대신 `cross-encoder/ms-marco-MiniLM-L-6-v2` 같은 전용 cross-encoder 모델을 씁니다.

```bash
uv run python 10_rag_rerank.py
```

---

### 종합 챗봇 (RAG + Reranker) — `chatbot.py`

지금까지 배운 모든 개념을 결합한 챗봇입니다. 일반 질문은 Reranker를 거쳐 가장 관련성 높은 문서로 답변합니다.

```
START → router → tool_agent                       → END  (도구 필요 시)
               → retrieve → rerank → rag_chat    → END  (일반 질문 → RAG + Rerank)
```

**결합된 기능**

| 기능 | 구현 |
|------|------|
| 메모리 | Python `dict` + `thread_id` |
| 도구 자동 선택 | `router_node`가 LLM으로 판단 |
| 도구 실행 | `tool_agent_node` (ReAct) |
| RAG + Rerank | `retrieve_node(k=6)` → `rerank_node` → `rag_chat_node` |
| 토큰 스트리밍 | `stream_mode=["messages", "updates"]` |
| 출처 표시 | rerank_node 완료 시 sources 캡처 → 스트림 후 출력 |

**등록된 도구**
- `get_time`: 현재 시각
- `calculate`: 수식 계산
- `remember_fact`: 사실 기억

```bash
uv run python chatbot.py
```

대화 루프가 실행됩니다. `quit` 입력 시 종료.

---

## 학습 순서

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → chatbot
기초   도구  MCP  메모리 분기  스트림 HITL  멀티   RAG  Rerank  종합
```

## 문제 해결

**Ollama 연결 오류**
```bash
ollama serve  # 서버 실행
ollama list   # 모델 확인
```

**모듈을 찾을 수 없음**
```bash
uv sync  # 패키지 재설치
```

**모델이 너무 느림**
- `qwen3:8b` → `qwen3:1.7b` 또는 `llama3.2:1b` 로 교체 후 각 파일의 `model=` 변경
