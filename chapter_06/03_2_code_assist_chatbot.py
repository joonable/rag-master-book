import os
from dotenv import load_dotenv
from typing import Optional, List, TypedDict, Annotated

# .env 파일로부터 환경 변수를 로드합니다.
load_dotenv()
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

# 캐시 파일 경로 및 URL 설정
CACHE_DIR = "data"
CACHE_FILE = os.path.join(CACHE_DIR, "lcel_docs.txt")
URL = "https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel"

def get_lcel_context(url=URL, cache_file=CACHE_FILE):
    # 캐시 디렉토리가 없으면 생성
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # 1. 캐시 파일이 존재하는지 확인
    if os.path.exists(cache_file):
        print(f"[시스템] 캐시 파일 '{cache_file}'에서 컨텍스트를 로드합니다...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    # 2. 캐시 파일이 없으면 크롤링 수행
    print(f"[시스템] '{url}'에서 문서를 크롤링합니다...")
    
    # RecursiveUrlLoader 설정 (단순 텍스트 추출을 위해 BeautifulSoup 사용)
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=2,  # 필요에 따라 깊이 조절
        extractor=lambda x: Soup(x, "html.parser").text
    )
    
    docs = loader.load()
    
    # 3. 'source' 메타데이터 기준으로 정렬
    docs.sort(key=lambda x: x.metadata.get('source', ''))
    
    # 4. "\n\n\n --- \n\n\n" 기준으로 조인
    lcel_context = "\n\n\n --- \n\n\n".join([doc.page_content for doc in docs])
    
    # 5. 결과를 캐시 파일로 저장
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(lcel_context)
    
    print(f"[시스템] 크롤링 완료 및 '{cache_file}'에 저장되었습니다.")
    return lcel_context

from langgraph.graph import END, StateGraph, START

# 1. Pydantic 기반 code 클래스 생성
class Code(BaseModel):
    """생성된 코드의 구조를 정의합니다."""
    prefix: str = Field(description="코드 생성에 앞선 간단한 설명 또는 도입부")
    imports: str = Field(description="필요한 모든 import 문")
    code: str = Field(description="실행 가능한 파이썬 코드")
    description: str = Field(description="코드의 작동 방식에 대한 상세 설명")

# 2. LLM 설정 (gpt-4o-mini)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. 프롬프트 설정 (LCEL 전문가 롤)
code_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 LangChain Expression Language (LCEL)의 전문가 코딩 어시스턴트입니다.
제공된 컨텍스트(LCEL 문서)를 기반으로 사용자의 질문에 답변하고 코드를 생성하십시오.
최신 LCEL 문법과 모범 사례를 준수해야 합니다.

<context>
{context}
</context>"""),
    ("human", "{question}")
])

# 4. 최종 형태 code_gen_chain 구성
code_gen_chain = code_gen_prompt | llm.with_structured_output(Code)

# 5. 그래프 상태 정의
class GraphState(TypedDict):
    """그래프 상태를 정의합니다."""
    error: str
    messages: List[BaseMessage]
    generation: Optional[Code]
    iterations: int

# 전역 플래그 선언
flag = True

# 6. 노드 정의
def generate(state: GraphState):
    """코드를 생성하는 노드입니다."""
    iterations = state["iterations"]
    error = state["error"]
    print(f"\n{'='*20} [단계: {iterations + 1}차 코드 생성] {'='*20}")
    
    messages = state["messages"]
    
    # 에러가 있으면 재시도 메시지 추가
    if error != "no" and error != "":
        print(f"[정보] 이전 에러({error})를 해결하기 위한 수정을 시도합니다.")
        messages.append(HumanMessage(content=f"이전 코드에서 다음 오류가 발생했습니다: {error}. 오류를 수정하고 다시 결과를 구조화하여 코드를 생성해주세요."))
    
    # 컨텍스트 로드
    lcel_context = get_lcel_context()
    question = messages[0].content
    
    # 코드 생성 (code_gen_chain 활용)
    print("[진행] LLM이 LCEL 코드를 생성 중입니다...")
    code_solution = code_gen_chain.invoke({"context": lcel_context, "question": question})
    
    # 메시지에 누적
    messages.append(AIMessage(content=f"{code_solution.prefix}\n\n{code_solution.imports}\n\n{code_solution.code}\n\n{code_solution.description}"))
    
    print(f"[완료] {iterations + 1}차 코드 생성 완료")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations + 1
    }

def code_check(state: GraphState):
    """생성된 코드를 실행하여 확인하는 노드입니다."""
    print(f"\n{'-'*20} [단계: 코드 실행 검증] {'-'*20}")
    generation = state["generation"]
    
    try:
        # 1. Imports 확인
        print("[검증 1/2] Import 문 확인 중...")
        exec(generation.imports)
    except Exception as e:
        error_msg = f"Import Error: {str(e)}"
        print(f"[실패] Import 검증 단계에서 오류 발생: {error_msg}")
        return {"error": error_msg}
        
    try:
        # 2. Code 확인 (imports + code)
        print("[검증 2/2] 메인 코드 실행 확인 중...")
        # input() 등이 포함된 경우 비대화형 환경에서 EOFError가 발생할 수 있음을 알림
        exec(generation.imports + "\n" + generation.code)
    except EOFError:
        error_msg = "Code Execution Error: input() function requested but no input was provided (EOFError)."
        print(f"[주의] {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Code Execution Error: {str(e)}"
        print(f"[실패] 코드 실행 단계에서 오류 발생: {error_msg}")
        return {"error": error_msg}
        
    print("[성공] 모든 검증 단계를 통과했습니다.")
    return {"error": "no"}

def reflect(state: GraphState):
    """오류를 code_gen_chain을 이용해서 코드의 오류를 잡는 노드입니다."""
    global flag
    error = state["error"]
    print(f"\n{'*'*20} [단계: 오류 분석 및 반성(Reflect)] {'*'*20}")
    print(f"[분석] 발생한 오류: {error}")
    
    messages = state["messages"]
    lcel_context = get_lcel_context()
    question = messages[0].content
    
    # 오류 수정을 위한 메시지 추가
    messages.append(HumanMessage(content=f"코드를 실행하는 중 오류가 발생했습니다: {error}\n이 오류를 해결한 수정된 코드를 생성해주세요."))
    
    # code_gen_chain을 활용하여 오류가 반영된 코드 생성
    print("[진행] 오류 원인을 파악하여 코드를 재구성 중입니다...")
    code_solution = code_gen_chain.invoke({"context": lcel_context, "question": question})
    
    # 메시지에 누적
    messages.append(AIMessage(content=f"{code_solution.prefix}\n\n{code_solution.imports}\n\n{code_solution.code}\n\n{code_solution.description}"))
    
    # 플래그 업데이트 (한 번만 reflect 하도록 제어하는 예시 로직)
    flag = False
    
    print("[완료] 수정 제안 생성이 완료되었습니다. 다시 검증 단계로 이동합니다.")
    return {
        "generation": code_solution,
        "messages": messages,
        "error": "yes"  # 다음 확인을 위해 에러 상태 유지
    }

# 7. 엣지 정의 (경로 판단 로직)
def decide_to_finish(state: GraphState):
    """종료 여부를 결정하는 엣지 로직입니다."""
    error = state["error"]
    iterations = state["iterations"]
    
    print(f"\n{'#'*20} [단계: 다음 경로 결정] {'#'*20}")
    print(f"[상태] 현재 반복: {iterations}, 에러 상태: {error}")

    if error == "no" or iterations >= 3:
        if error == "no":
            print("[결정] 검증이 완료되어 작업을 성공적으로 종료합니다.")
        else:
            print("[결정] 최대 반복 횟수(3회)에 도달하여 작업을 종료합니다.")
        return "end"
    else:
        # flag가 True이면 reflect, 아니면 generate 리턴
        if flag:
            print("[결정] 오류 수정을 위해 'reflect' 노드로 이동합니다.")
            return "reflect"
        else:
            print("[결정] 코드를 처음부터 다시 구성하기 위해 'generate' 노드로 이동합니다.")
            return "generate"

# 8. 워크플로우 구성
workflow = StateGraph(GraphState)

# --- 노드 추가 ---
# 각 함수를 그래프 내에서 사용할 노드 이름으로 등록합니다.
workflow.add_node("generate", generate)
workflow.add_node("code_check", code_check)
workflow.add_node("reflect", reflect)

# --- 엣지(흐름) 연결 ---

# 1) 진입점 설정: 그래프가 시작(START)되면 가장 먼저 'generate' 노드를 실행합니다.
workflow.add_edge(START, "generate")

# 2) 일반 엣지: 'generate' 노드의 작업이 끝나면 무조건 'code_check' 노드로 이동합니다.
workflow.add_edge("generate", "code_check")

# 3) 조건부 엣지: 'code_check'의 실행 결과에 따라 다음 노드를 유동적으로 선택합니다.
# decide_to_finish 함수가 리턴하는 문자열("end", "reflect", "generate")에 매칭되는 경로로 분기합니다.
workflow.add_conditional_edges(
    "code_check",            # 시작 노드
    decide_to_finish,        # 다음 경로를 판단할 함수 (리턴값 기반)
    {
        "end": END,          # 리턴값이 "end"인 경우 -> 그래프 종료 지점(END)으로 이동
        "reflect": "reflect",# 리턴값이 "reflect"인 경우 -> 'reflect' 노드로 이동
        "generate": "generate" # 리턴값이 "generate"인 경우 -> 'generate' 노드로 이동
    }
)

# 4) 순환 엣지(Loop): 'reflect' 노드에서 수정 제안이 생성되면 다시 'code_check' 노드로 보내서 검증합니다.
# 이를 통해 "검증 -> 실패 -> 수정 -> 재검증"의 자기 교정(Self-Correction) 사이클이 완성됩니다.
workflow.add_edge("reflect", "code_check")

# --- 그래프 컴파일 ---
# 정의된 노드와 엣지를 바탕으로 실행 가능한 애플리케이션 객체를 생성합니다.
app = workflow.compile()

if __name__ == "__main__":
    # 그래프 시각화 및 저장
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open("code_assist_chatbot_graph.png", "wb") as f:
            f.write(png_data)
        print("\n--- 워크플로우 그래프가 'code_assist_chatbot_graph.png'로 저장되었습니다 ---\n")
    except Exception as e:
        print(f"\n그래프 시각화 실패: {e}")

    # 질문 입력 받기
    print("="*50)
    print("LCEL 코딩 어시스턴트입니다. 질문을 입력해주세요.")
    print("예시: 두 숫자를 더하는 간단한 LCEL 체인을 만들어줘.")
    print("="*50)
    user_question = input("\n질문 입력: ")
    
    if not user_question.strip():
        user_question = "사용자로부터 입력을 받아 텍스트를 대문자로 변환한 뒤 출력하는 간단한 LCEL 체인을 만들어줘."
        print(f"\n[알림] 입력이 없어 기본 질문으로 진행합니다: {user_question}")
    
    # 초기 상태 설정
    initial_state = {
        "messages": [HumanMessage(content=user_question)],
        "error": "",
        "iterations": 0,
        "generation": None
    }
    
    # 그래프 실행
    final_state = app.invoke(initial_state)
    
    # 최종 결과 출력
    result = final_state["generation"]
    print("\n" + "!"*30 + " 최종 결과물 " + "!"*30)
    print(f"\n[설명]\n{result.prefix}\n")
    print(f"[임포트 문]\n{result.imports}\n")
    print(f"[코드]\n{result.code}\n")
    print(f"[상세 설명]\n{result.description}")
    print("!"*73)
