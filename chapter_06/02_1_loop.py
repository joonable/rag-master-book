import operator
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 1. State 정의: messages 필드는 Annotated와 operator.add를 사용하여 누적되도록 설정
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 2. LLM 설정 (gpt-4o-mini)
llm = ChatOpenAI(model="gpt-4o-mini")

# 3. Chatbot 노드 정의
def chatbot(state: State):
    # state["messages"]에 담긴 대화 기록을 기반으로 LLM 호출
    response = llm.invoke(state["messages"])
    # 새로운 응답을 messages 필드에 추가 (리듀서에 의해 기존 리스트와 합쳐짐)
    return {"messages": [response]}

# 4. 그래프 구성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("chatbot", chatbot)

# 에지 설정: START -> chatbot -> END
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# 그래프 컴파일
graph = workflow.compile()

def run_chat():
    print("나만의 랭그래프 챗봇 (종료를 원하시면 'quit', 'exit', 'q'를 입력하세요)")
    
    # 초기 상태 (빈 메시지 리스트)
    # 실제 루프에서는 이전 상태를 유지하며 대화를 이어갈 수 있습니다.
    # 여기서는 간단히 사용자 입력을 받아 계속해서 그래프를 실행하는 구조로 만듭니다.
    messages = []
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("대화를 종료합니다.")
            break
        
        # 사용자 입력을 HumanMessage로 변환하여 리스트에 추가
        new_message = HumanMessage(content=user_input)
        messages.append(new_message)
        
        # graph.stream()을 사용하여 각 단계의 이벤트를 처리
        # 입력으로 전체 메시지 리스트를 전달 (State는 누적되지만, 매 호출마다 새로 시작하는 경우 대비)
        # 랭그래프의 State가 누적되도록 설계되었으므로, 현재는 매번 전체 messages를 보냅니다.
        # (참고: persistence를 사용하면 이전 messages를 수동으로 관리하지 않아도 됩니다.)
        for event in graph.stream({"messages": [new_message]}):
            # event는 노드 이름을 키로 하고 노드의 반환값을 값으로 하는 딕셔너리
            for value in event.values():
                # chatbot 노드에서 반환한 최신 응답 출력
                print(f"Chatbot: {value['messages'][-1].content}")
                # 누적을 위해 전체 리스트에 응답 추가
                messages.append(value['messages'][-1])

if __name__ == "__main__":
    run_chat()
