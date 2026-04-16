import os
# 0. 최상단에서 환경 변수 설정 (ChromaDB 텔레메트리 비활성화)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 로그 레벨 조정
logging.getLogger('chromadb').setLevel(logging.ERROR)

# 1. 환경 변수 로드
load_dotenv()

def format_docs(docs):
    """메타데이터를 제외하고 순수한 문서 내용만 추출하는 유틸리티 함수"""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    # 설정값 (02_1_multi_query_retriever.py에서 생성된 DB 경로와 컬렉션 이름 사용)
    persist_directory = "data/chroma_db"
    collection_name = "multi_query_invest"
    
    # 2. 모델 및 임베딩 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    embeddings = OpenAIEmbeddings()

    # 3. 벡터 데이터베이스(Chroma) 로드
    # 인덱싱 로직(TextLoader, Splitter 등)은 모두 제거되었습니다.
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # 4. 데이터 존재 확인 (인덱싱을 새로 하지 않음)
    collection_count = vectorstore._collection.count()
    if collection_count == 0:
        print(f"❌ [{collection_name}] 컬렉션이 비어있습니다.")
        print("먼저 chapter_04/02_1_multi_query_retriever.py를 실행하여 데이터를 인덱싱해 주세요.")
        return
    else:
        print(f"✅ [{collection_name}] 기존 {collection_count}개 데이터를 사용합니다.")

    # 5. 가상 문서 생성 체인 (HyDE 1단계)
    hyde_system = "당신은 전문적인 금융 상담가입니다. 사용자의 질문에 대해 답변이 될 수 있는 가상의 문서를 작성해 주세요."
    hyde_user = "{question}"
    hyde_prompt = ChatPromptTemplate.from_messages([
        ("system", hyde_system),
        ("user", hyde_user)
    ])
    
    hyde_chain = hyde_prompt | llm | StrOutputParser()

    # 6. 최종 응답 생성 체인 (HyDE 2~3단계)
    qa_prompt = ChatPromptTemplate.from_template("""다음 제공된 문맥(Context)을 사용하여 질문에 답하세요. 
문맥에 없는 내용은 답하지 마세요. 최대한 간결하게 답변하세요.

[Context]
{context}

[Question]
{question}

[Helpful Answer]:""")

    # 전체 RAG 파이프라인 구성
    # hyde_chain -> retriever -> format_docs 순으로 데이터가 흐릅니다.
    retrieval_chain = hyde_chain | vectorstore.as_retriever() | format_docs

    final_chain = (
        {"context": retrieval_chain, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    print("\n" + "="*50)
    print("🚀 HyDE (Hypothetical Document Embeddings) 상세 분석 모드")
    print("="*50)

    while True:
        query = input("\n❓ 질문 (종료: q): ").strip()
        if query.lower() in ['q', 'quit', 'exit']: break
        if not query: continue

        # [1단계] 가상 문서 생성
        print(f"\n📡 [1단계] 가상 문서 생성 중...")
        hypothetical_doc = hyde_chain.invoke({"question": query})
        print(f"--- 가상 문서 샘플 ---\n{hypothetical_doc[:300]}...\n------------------")

        # [2단계] 가상 문서를 기반으로 실제 문서 검색
        print(f"\n🔍 [2단계] 가상 문서를 기반으로 실제 문서 검색 중...")
        context = retrieval_chain.invoke({"question": query})
        
        print("\n--- [검색된 실제 문맥(Context)] ---")
        print(context if context else "검색된 결과가 없습니다.")
        print("---------------------------------")

        # [3단계] 최종 답변 생성 시 사용되는 프롬프트 전문 출력
        print(f"\n📝 [3단계] 최종 프롬프트 전문")
        full_prompt = qa_prompt.format(context=context, question=query)
        print("-" * 30)
        print(full_prompt)
        print("-" * 30)
        
        # [4단계] 최종 답변 생성
        print(f"\n✨ [4단계] 최종 답변 생성")
        response = final_chain.invoke(query)
        
        print("-" * 30)
        print(response)
        print("-" * 30)

if __name__ == "__main__":
    main()
