"""
02. 커스텀 도구 + ReAct 에이전트
LLM이 스스로 도구를 선택하고 호출하는 패턴을 익힙니다.
"""

from datetime import datetime
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent as create_react_agent


# 1. 도구 정의 — @tool 데코레이터로 LLM이 호출할 수 있는 함수를 등록
@tool
def get_current_time() -> str:
    """현재 날짜와 시간을 반환합니다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """
    수식을 계산합니다.

    Args:
        expression: 계산할 수식 문자열 (예: "2 + 3 * 4")
    """
    try:
        # eval 대신 안전한 계산만 허용
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "허용되지 않는 문자가 포함되어 있습니다."
        result = eval(expression)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def word_counter(text: str) -> str:
    """
    텍스트의 단어 수와 글자 수를 셉니다.

    Args:
        text: 분석할 텍스트
    """
    words = len(text.split())
    chars = len(text)
    return f"단어 수: {words}, 글자 수: {chars}"


# 2. LLM + 도구 연결
llm = ChatOllama(model="qwen3:8b", temperature=0)
tools = [get_current_time, calculator, word_counter]

# create_react_agent: LangGraph의 내장 ReAct 에이전트
# 내부적으로 "도구 호출 → 결과 확인 → 다시 LLM" 루프를 자동으로 구성
agent = create_react_agent(llm, tools)


# 3. 실행
def run(query: str):
    print(f"\nQuery: {query}")
    print("-" * 40)

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # 메시지 흐름 출력
    for msg in result["messages"]:
        role = msg.__class__.__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"[{role}] 도구 호출: {[tc['name'] for tc in msg.tool_calls]}")
        elif hasattr(msg, "name") and msg.name:  # ToolMessage
            print(f"[Tool:{msg.name}] {msg.content}")
        else:
            print(f"[{role}] {msg.content}")


if __name__ == "__main__":
    run("지금 몇 시야?")
    run("1234 * 5678 계산해줘")
    run("'안녕하세요 반갑습니다 오늘 날씨가 좋네요' 이 문장의 단어 수 알려줘")
