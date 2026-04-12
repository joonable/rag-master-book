import operator
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition # 2) 코드 간소화 도구
from langgraph.checkpoint.memory import MemorySaver      # 1) 메모리 저장소
from dotenv import load_dotenv

load_dotenv()

# 1. 도구 및 상태 정의
tools = [TavilySearch(max_results=2)]

class State(TypedDict):
    # Annotated와 operator.add는 내부적으로 상태를 합칠 때 사용됩니다.
    messages: Annotated[List[BaseMessage], operator.add]

# 2. LLM 설정 및 도구 바인딩
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# 3. 노드 정의
def chatbot(state: State):
    # MemorySaver가 이전 대화 내용을 자동으로 state["messages"]에 채워줍니다.
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 4. 그래프 구성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode(tools)) # Prebuilt ToolNode 사용

# 에지 설정
workflow.add_edge(START, "chatbot")

# 조건부 에지 추가
# tools_condition은 LLM의 tool_calls 유무에 따라 "tools" 노드 또는 END로 자동 라우팅합니다.
workflow.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# 도구 실행 후 다시 챗봇으로
workflow.add_edge("tools", "chatbot")

# 5. 상태 저장소(Memory) 추가 및 컴파일
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory) # 체크포인터 연결

def run_chat():
    print("🧠 랭그래프 상태 저장 실습 (MemorySaver)")
    
    # 3) thread_id를 포함한 config 정의
    # 이 ID를 기반으로 대화 기록이 식별됩니다.
    config = {"configurable": {"thread_id": "my_conversation_1"}}
    
    while True:
        user_input = input("\nUser: ")
        
        # 5) 종료 시 상태 출력
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n--- 최종 그래프 상태(State) 확인 ---")
            # 현재 thread_id에 저장된 모든 상태를 가져옵니다.
            state_snapshot = graph.get_state(config)
            messages = state_snapshot.values.get("messages", [])
            print(f"현재 저장된 총 메시지 수: {len(messages)}개")
            for i, msg in enumerate(messages):
                role = "Bot" if not isinstance(msg, HumanMessage) else "User"
                content = msg.content if msg.content else f"(Tool Call/Result: {getattr(msg, 'tool_calls', 'Result')})"
                print(f"{i+1}. [{role}]: {content[:70]}...")
            break
        
        new_msg = HumanMessage(content=user_input)
        
        # 4) stream에 config 추가
        # 이제 외부 리스트 관리 없이 새 메시지만 보내도 맥락이 유지됩니다.
        for event in graph.stream({"messages": [new_msg]}, config=config, stream_mode="updates"):
            for node_name, value in event.items():
                print(f"\n[Node: {node_name}]")
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    if last_msg.content:
                        print(f"Chatbot: {last_msg.content}")
                    elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"Action: {last_msg.tool_calls[0]['name']} 도구 호출 준비 중...")

if __name__ == "__main__":
    run_chat()
