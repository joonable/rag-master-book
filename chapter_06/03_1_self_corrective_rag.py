"""
이 코드는 LangGraph를 활용한 Self-Corrective RAG (CRAG) 시스템의 구현 예제입니다.

주요 목적 및 기능:
1. 자가 교정(Self-Correction): 검색된 문서의 관련성을 LLM이 스스로 평가합니다.
2. 동적 경로 제어: 검색 결과가 부적절할 경우, 질문을 재작성(Rewriting)하고 웹 검색(Tavily)을 통해 외부 지식을 동적으로 보완합니다.
3. 그래프 워크플로우: LangGraph를 사용하여 '검색 -> 평가 -> (필요시 웹 검색) -> 답변 생성'의 복잡한 로직을 상태 기반의 유향 그래프로 관리합니다.
"""
import os
from typing import List, TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START

# 1. 환경 변수 로드 (.env)
load_dotenv()

# --- 1. Retriever 생성 (문서 크롤링 및 인덱싱) ---
persist_directory = "./data/chroma_db"
collection_name = "google_style_guides"

embeddings = OpenAIEmbeddings()

vector_db = Chroma(
    collection_name=collection_name,
    persist_directory=persist_directory,
    embedding_function=embeddings
)

if vector_db._collection.count() == 0:
    print("--- 기존 인덱스가 없습니다. 문서를 크롤링하고 인덱싱을 시작합니다. ---")
    urls = [
        "https://google.github.io/styleguide/pyguide.html",
        "https://google.github.io/styleguide/javaguide.html",
        "https://google.github.io/styleguide/jsguide.html",
    ]
    loader = WebBaseLoader(urls)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs)
    vector_db = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print("--- 인덱싱이 완료되었습니다. ---")
else:
    print("--- 기존 인덱스를 발견했습니다. 저장된 데이터를 로드합니다. ---")

retriever = vector_db.as_retriever()

# --- 2. 문서 관련성 평가 (Document Grader) ---
class GradeDocuments(BaseModel):
    """문서가 질문과 관련이 있는지 평가합니다."""
    binary_score: Literal["예", "아니오"] = Field(description="문서가 질문과 관련이 있는지 여부, '예' 또는 '아니오'")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 검색된 문서가 사용자의 질문과 관련이 있는지 평가하는 평가자(Grader)입니다.\n"
                   "문서에 사용자의 질문과 관련된 키워드나 의미가 포함되어 있다면 '관련 있음'으로 간주합니다.\n"
                   "평가 결과는 질문과 관련 여부에 따라 '예' 또는 '아니오'로만 답해주세요."),
        ("human", "검색된 문서: \n\n {document} \n\n 사용자 질문: {question}"),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader

# --- 3. 답변 생성 (RAG Chain) ---
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 질문-답변 과제를 위한 보조 전문가입니다.\n"
                   "검색된 다음의 문맥(context)을 사용하여 질문(question)에 답하세요.\n"
                   "답을 모른다면 모른다고 말하고, 직접 답변을 지어내지 마세요.\n"
                   "최대한 세 문장 내외로 간결하게 답변하세요."),
        ("human", "문맥: \n\n {context} \n\n 사용자 질문: {question}"),
    ]
)
rag_chain = rag_prompt | llm | StrOutputParser()

# --- 4. 질문 재작성 (Question Rewriter) ---
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 사용자 질문을 웹 검색에 최적화된 형태로 변환하는 질문 재작성 전문가입니다.\n"
                   "원래 질문의 의도를 유지하면서, 검색 엔진이 더 정확한 결과를 찾을 수 있도록 구체적이고 명확한 키워드로 재구성하세요.\n"
                   "출력은 재작성된 질문 문장만 반환하세요."),
        ("human", "원래 질문: {question}"),
    ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()

# --- 5. 웹 검색 도구 (Web Search Tool) ---
web_search_tool = TavilySearchResults(k=3)

# --- 6. 그래프 상태 (Graph State) 정의 ---
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]

# --- 7. 노드(Nodes) 및 엣지(Edges) 구현 ---

def retrieve(state):
    print("\n[Node: Retrieve] 문서 검색 중...")
    question = state["question"]
    documents = retriever.invoke(question)
    print(f"-> 검색된 문서 개수: {len(documents)}")
    return {"documents": documents, "question": question}

def generate(state):
    print("\n[Node: Generate] 답변 생성 중...")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}

def grade_documents(state):
    print("\n[Node: Grade Documents] 문서 관련성 평가 중...")
    question = state["question"]
    documents = state["documents"]
    
    filtered_documents = []
    web_search = "아니오"
    
    for i, d in enumerate(documents):
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "예":
            print(f"-> 문서 {i+1}: 관련 있음")
            filtered_documents.append(d)
        else:
            print(f"-> 문서 {i+1}: 관련 없음")
            web_search = "예"
            
    return {"documents": filtered_documents, "question": question, "web_search": web_search}

def transform_query(state):
    print("\n[Node: Transform Query] 질문 재작성 중...")
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    print(f"-> 재작성된 질문: {better_question}")
    return {"question": better_question}

def web_search(state):
    print("\n[Node: Web Search] 웹 검색 수행 중...")
    question = state["question"]
    documents = state.get("documents", [])
    
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    print("-> 웹 검색 결과가 문서 리스트에 추가되었습니다.")
    
    return {"documents": documents, "question": question}

def decide_to_generate(state):
    print("\n[Edge: Decision] 다음 단계 결정 중...")
    if state["web_search"] == "예":
        print("-> 결과: 관련 없는 문서가 포함됨. 웹 검색 단계로 이동합니다.")
        return "transform_query"
    else:
        print("-> 결과: 모든 문서가 관련됨. 바로 답변을 생성합니다.")
        return "generate"

# --- 8. 그래프 구성 (Workflow Graph) ---
workflow = StateGraph(GraphState)

# 노드 정의 (Nodes)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

# 그래프 연결 (Edges)

# 1. 시작점 설정: 사용자의 질문이 들어오면 가장 먼저 'retrieve' 노드를 실행하여 관련 문서를 검색합니다.
workflow.add_edge(START, "retrieve")

# 2. 검색 직후 평가: 검색된 문서들이 실제로 질문과 관련이 있는지 검증하기 위해 'grade_documents' 노드로 이동합니다.
workflow.add_edge("retrieve", "grade_documents")

# 3. 자가 교정 로직 (Conditional Edges):
# 'grade_documents'에서 판별한 결과(web_search 여부)에 따라 경로를 분기합니다.
# - 모든 문서가 관련 있다면: 즉시 'generate'로 이동하여 답변을 생성합니다.
# - 관련 없는 문서가 섞여 있다면: 'transform_query'로 이동하여 외부 지식을 찾을 준비를 합니다.
workflow.add_conditional_edges(
    "grade_documents",      # 조건부 로직이 시작되는 노드
    decide_to_generate,     # 다음 노드를 결정할 판단 함수
    {
        "transform_query": "transform_query",  # 관련성 부족 시: 질문 재작성 노드로 분기
        "generate": "generate",                # 관련성 충분 시: 답변 생성 노드로 분기
    },
)

# 4. 외부 지식 보완 (Fallback Path): 
# 내부 문서로 부족할 경우, 검색 최적화를 위해 질문을 재구성한 후 웹 검색을 수행합니다.
workflow.add_edge("transform_query", "web_search")

# 5. 지식 통합: 웹 검색을 통해 얻은 새로운 정보를 기존 문서 리스트에 통합하고 최종 답변 생성을 위해 'generate'로 이동합니다.
workflow.add_edge("web_search", "generate")

# 6. 종료: 최종 생성된 답변을 사용자에게 반환하며 프로세스를 마칩니다.
workflow.add_edge("generate", END)

app = workflow.compile()

# --- 9. 실행 (Running the App) ---

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Self-Corrective RAG 시스템 (종료하려면 'exit' 또는 'q' 입력)")
    print("="*50)

    while True:
        user_input = input("\n질문을 입력하세요: ").strip()
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("프로그램을 종료합니다.")
            break
        
        if not user_input:
            continue

        inputs = {"question": user_input}
        
        print(f"\n{'#'*10} 워크플로우 시작 {'#'*10}")
        
        final_answer = ""
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"--- 노드 '{key}' 실행 완료 ---")
                if "generation" in value:
                    final_answer = value["generation"]
        
        print(f"\n{'#'*10} 최종 답변 {'#'*10}")
        print(final_answer)
        print(f"{'#'*32}")
