"""
07. Human-in-the-Loop
그래프 실행 도중 사람의 승인/수정을 끼워넣는 패턴입니다.

흐름:
  START → draft → [INTERRUPT: 사람 검토] → revise → END

핵심 API:
  - interrupt(value): 실행을 일시 정지하고 값을 노출
  - Command(resume=value): 정지된 그래프를 재개
  - MemorySaver: 정지 상태를 thread_id에 저장 (재개에 필수)
"""

from typing import Annotated
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str       # LLM이 초안 작성
    approved: bool   # 사람의 승인 여부


llm = ChatOllama(model="qwen3:8b", temperature=0)
checkpointer = MemorySaver()


# --- 노드 ---

def draft_node(state: State) -> State:
    """LLM이 이메일 초안을 작성합니다."""
    prompt = f"다음 요청에 대한 이메일 초안을 작성해주세요:\n\n{state['messages'][-1].content}"
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"draft": response.content}


def human_review_node(state: State) -> Command:
    """
    interrupt()로 실행을 멈추고 사람에게 초안을 보여줍니다.
    사람이 'approve' 또는 수정 내용을 입력하면 재개됩니다.
    """
    print("\n" + "=" * 50)
    print("📝 LLM 초안:")
    print(state["draft"])
    print("=" * 50)

    # interrupt: 실행 일시 정지 → 반환값은 Command(resume=...)로 전달됨
    human_input = interrupt("초안을 검토하세요. 'approve' 입력 시 승인, 그 외 입력 시 수정 지시:")

    if human_input.strip().lower() == "approve":
        return Command(goto="finalize_node", update={"approved": True})
    else:
        # 수정 지시를 메시지로 추가하고 다시 draft로
        return Command(
            goto="revise_node",
            update={
                "approved": False,
                "messages": [{"role": "user", "content": f"수정 요청: {human_input}"}],
            },
        )


def revise_node(state: State) -> State:
    """수정 지시를 반영해 초안을 다시 작성합니다."""
    print("\n✏️  수정 중...")
    messages = [
        {"role": "assistant", "content": state["draft"]},
        state["messages"][-1],  # 수정 지시
    ]
    response = llm.invoke(messages)
    return {"draft": response.content}


def finalize_node(state: State) -> State:
    """승인된 초안을 최종 메시지로 저장합니다."""
    print("\n✅ 최종 승인 완료")
    return {"messages": [{"role": "assistant", "content": state["draft"]}]}


# --- 그래프 ---

graph = (
    StateGraph(State)
    .add_node("draft_node", draft_node)
    .add_node("human_review_node", human_review_node)
    .add_node("revise_node", revise_node)
    .add_node("finalize_node", finalize_node)
    .add_edge(START, "draft_node")
    .add_edge("draft_node", "human_review_node")
    .add_edge("revise_node", "human_review_node")  # 수정 후 다시 검토
    .add_edge("finalize_node", END)
    .compile(checkpointer=checkpointer, interrupt_before=["human_review_node"])
)


def run():
    config = {"configurable": {"thread_id": "email-1"}}
    topic = input("이메일 주제를 입력하세요: ").strip()

    # 1단계: draft까지 실행 → human_review_node 직전에서 멈춤
    graph.invoke(
        {
            "messages": [{"role": "user", "content": topic}],
            "draft": "",
            "approved": False,
        },
        config=config,
    )

    # 2단계: 사람 검토 루프
    while True:
        # 현재 상태 확인
        current = graph.get_state(config)
        if not current.next:  # 더 이상 실행할 노드 없으면 종료
            break

        human_input = input("\n> ").strip()
        result = graph.invoke(Command(resume=human_input), config=config)

        # finalize까지 완료됐으면 종료
        current = graph.get_state(config)
        if not current.next:
            print(f"\n📧 최종 이메일:\n{result['messages'][-1].content}")
            break


if __name__ == "__main__":
    run()
