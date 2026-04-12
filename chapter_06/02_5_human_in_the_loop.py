import operator
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# 1. 도구 및 상태 정의
tools = [TavilySearch(max_results=2)]

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 2. LLM 설정 및 도구 바인딩
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# 3. 노드 정의
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 4. 그래프 구성
workflow = StateGraph(State)

workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")

# 5. [핵심] 체크포인터와 인터럽트 설정
memory = MemorySaver()

# interrupt_before=["tools"] 설정을 통해 'tools' 노드 진입 직전에 실행을 일시 중단합니다.
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"] 
)

def run_chat():
    print("🙋 랭그래프 개입 실습 (Human-in-the-loop)")
    config = {"configurable": {"thread_id": "hitl_session_1"}}
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]: break
        
        new_msg = HumanMessage(content=user_input)
        
        # 1차 실행: chatbot 노드까지 실행되고 'tools' 직전에 멈춤
        for event in graph.stream({"messages": [new_msg]}, config=config, stream_mode="updates"):
            for node_name, value in event.items():
                print(f"\n[Node: {node_name}]")
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    if last_msg.content:
                        print(f"Chatbot: {last_msg.content}")
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"⚠️ 승인 대기 중: {last_msg.tool_calls[0]['name']} 도구를 실행하려고 합니다.")

        # 현재 그래프의 상태를 확인하여 인터럽트 되었는지 체크
        snapshot = graph.get_state(config)
        
        # snapshot.next에 다음 실행될 노드(tools)가 있다면 중단된 상태입니다.
        if snapshot.next:
            print(f"\n>>> 다음 노드 실행 대기 중: {snapshot.next}")
            user_approval = input(">>> 도구 실행을 승인하시겠습니까? (y/n): ")
            
            if user_approval.lower() == "y":
                # 입력값을 None으로 주면 중단된 지점부터 다시 실행을 이어갑니다.
                print(">>> 실행을 승인했습니다. 도구를 실행합니다...")
                for event in graph.stream(None, config=config, stream_mode="updates"):
                    for node_name, value in event.items():
                        print(f"\n[Node: {node_name}]")
                        if "messages" in value:
                            print(f"Chatbot: {value['messages'][-1].content}")
            else:
                print(">>> 도구 실행을 취소했습니다. 다음 입력을 기다립니다.")
                # 실행을 취소한 경우 상태는 그대로 머물러 있게 됩니다.

if __name__ == "__main__":
    run_chat()
