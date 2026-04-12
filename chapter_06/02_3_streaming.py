import operator
import json
from typing import Annotated, TypedDict, List, Union
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 1. Tavily 검색엔진을 tool로 정의 (max_results = 2)
tools = [TavilySearch(max_results=2)]

# 2. State 정의 (messages 누적)
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 3. LLM 설정 (gpt-4o-mini) 및 도구 바인딩
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# 4. chatbot 노드 정의
def chatbot(state: State):
    # LLM을 호출하고 결과를 반환 (메시지 리스트에 추가됨)
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 5. BasicToolNode 클래스 정의
class BasicToolNode:
    """도구 실행을 담당하는 사용자 정의 노드"""
    def __init__(self, tools: list):
        # 도구 이름을 키로 하고 도구 자체를 값으로 하는 딕셔너리 생성
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State):
        messages = state.get("messages", [])
        last_message = messages[-1]
        
        # 마지막 메시지에 tool_calls가 없으면 에러 (라우터에서 걸러야 함)
        if not last_message.tool_calls:
            raise ValueError("No tool calls found in the last message.")
        
        outputs = []
        # 모든 도구 호출을 순차적으로 실행
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool = self.tools_by_name[tool_name]
            # 도구 실행 결과 생성
            observation = tool.invoke(tool_call["args"])
            # ToolMessage 객체로 반환 (도구 이름과 실행 결과를 명시)
            outputs.append(
                ToolMessage(
                    content=json.dumps(observation),
                    tool_call_id=tool_call["id"],
                    name=tool_name # 도구 이름 추가
                )
            )
        return {"messages": outputs}

# 6. tool_node 생성 및 노드 추가 준비
tool_node = BasicToolNode(tools)

# 7. route_tools: 다음에 갈 노드를 결정하는 라우터 함수
def route_tools(state: State):
    """
    가장 최근 메시지에 tool_calls 속성이 있다면 'tools' 노드를, 
    그렇지 않으면 종료 지점(END)을 반환합니다.
    """
    # 상태값의 가장 최근 메시지 정의
    last_message = state["messages"][-1]
    
    # 도구 호출(tool_calls)이 존재하는지 확인
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("--- [Router] LLM이 도구 사용을 결정했습니다. 'tools' 노드로 이동합니다. ---")
        return "tools"
    # 없으면 종료
    print("--- [Router] 도구 호출이 없습니다. 종료 지점(END)으로 이동합니다. ---")
    return END

# 8. 그래프 구성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", tool_node)

# 에지 설정
workflow.add_edge(START, "chatbot")

# chatbot 노드 다음에 어디로 갈지 결정하는 조건부 에지 추가
workflow.add_conditional_edges(
    "chatbot",
    route_tools,
    # route_tools의 반환값에 따라 매핑될 노드 지정
    {"tools": "tools", END: END}
)

# 도구 실행 후에는 다시 chatbot으로 돌아가서 결과를 해석하게 함
workflow.add_edge("tools", "chatbot")

# 그래프 컴파일
graph = workflow.compile()

def run_chat():
    print("랭그래프 스트리밍 실습 (stream_mode='updates')")
    messages = []
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        new_msg = HumanMessage(content=user_input)
        messages.append(new_msg)
        
        # graph.stream()을 호출할 때 stream_mode="updates"를 지정합니다.
        # 이 모드는 각 노드가 상태를 '업데이트'할 때마다 이벤트를 발생시킵니다.
        for event in graph.stream({"messages": [new_msg]}, stream_mode="updates"):
            for node_name, value in event.items():
                print(f"\n[Node Update: {node_name}]")
                # 해당 노드가 추가/변경한 메시지들 출력
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    if last_msg.content:
                        print(f"Content: {last_msg.content}")
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"Tool Calls: {last_msg.tool_calls}")
                    
                    # 대화 기록 누적 (다음 호출을 위해)
                    messages.append(last_msg)

if __name__ == "__main__":
    run_chat()
