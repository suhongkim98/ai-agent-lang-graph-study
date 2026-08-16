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

# 모델 다운로드 (도구 호출 지원 모델)
ollama pull qwen3:8b
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

### 09. 종합 챗봇 — `09_chatbot.py`

지금까지 배운 개념을 하나로 결합한 실용적인 챗봇입니다.

```
START → router_node → (tool_agent_node | direct_chat_node) → END
```

**결합된 기능**

| 기능 | 구현 |
|------|------|
| 메모리 | Python `dict` + `thread_id` (04와 동일 방식) |
| 도구 자동 선택 | `router_node`가 LLM으로 판단 |
| 도구 실행 | `tool_agent_node` (ReAct) |
| 토큰 스트리밍 | `stream_mode="messages"` + `langgraph_node` 필터 |

스트리밍 중 `AIMessageChunk`를 누적(`chunk + chunk`)해 저장소에 반영하므로 LLM을 한 번만 호출합니다.

**등록된 도구**
- `get_time`: 현재 시각
- `calculate`: 수식 계산
- `remember_fact`: 사실 기억

```bash
uv run python 09_chatbot.py
```

대화 루프가 실행됩니다. `quit` 입력 시 종료.

---

```bash
uv run python 10_skills.py
```

---

## 학습 순서

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10
기초   도구  MCP  메모리 분기  스트림 HITL  멀티   종합  스킬
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
