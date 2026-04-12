import os
# 0. 최상단에서 환경 변수 설정 (ChromaDB 텔레메트리 비활성화)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 로그 레벨 조정
logging.getLogger('chromadb').setLevel(logging.ERROR)

# 1. 환경 변수 로드
load_dotenv()

def main():
    # 설정값
    file_path = "chapter_04/data/How_to_invest_money.txt"
    persist_directory = "./.chroma"
    collection_name = "multi_query_invest"
    
    # 2. 모델 및 임베딩 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    embeddings = OpenAIEmbeddings()

    # 3. 벡터 데이터베이스(Chroma) 설정
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # 4. 데이터 인덱싱 여부 확인 및 처리
    collection_count = vectorstore._collection.count()
    if collection_count == 0:
        print(f"[{collection_name}] 인덱싱 시작...")
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore.add_documents(documents=splits)
    else:
        print(f"[{collection_name}] 기존 {collection_count}개 데이터를 사용합니다.")

    # 5. MultiQueryRetriever 설정
    # from_llm은 내부적으로 LineListOutputParser를 사용하여 결과를 List[str]로 반환합니다.
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), 
        llm=llm
    )

    # 6. QA 프롬프트 템플릿 설정
    prompt_template = """다음 제공된 문맥(Context)을 사용하여 질문에 답하세요. 
최대한 간결하게 답변하세요.

[Context]
{context}

[Question]
{question}

[Helpful Answer]:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    # 7. RetrievalQA 체인 구성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    print("\n" + "="*50)
    print("🔎 Multi-Query Retriever 상세 분석 모드")
    print("="*50)

    while True:
        query = input("\n❓ 질문 (종료: q): ").strip()
        if query.lower() in ['q', 'quit', 'exit']: break
        if not query: continue

        print(f"\n📡 [1단계] 질문 확장")
        # MultiQueryRetriever의 llm_chain은 LineListOutputParser를 통해 List[str]을 반환합니다.
        generated_queries = retriever.llm_chain.invoke({"question": query})
        
        print(f"원래 질문: {query}")
        for i, g_query in enumerate(generated_queries[:3]):
            print(f"  확장 {i+1}: {g_query}")

        print(f"\n🔍 [2단계] 문서 검색")
        docs = retriever.invoke(query)
        print(f"검색된 문서 조각 수: {len(docs)}개")

        print(f"\n📝 [3단계] 최종 프롬프트 샘플")
        context_text = "\n\n".join([doc.page_content for doc in docs])
        full_prompt = QA_CHAIN_PROMPT.format(context=context_text, question=query)
        print(f"{full_prompt[:400]}...\n(...생략...)")

        print(f"\n✨ [4단계] 최종 답변")
        result = qa_chain.invoke({"query": query})
        print("-" * 30)
        print(result["result"])
        print("-" * 30)

if __name__ == "__main__":
    main()
