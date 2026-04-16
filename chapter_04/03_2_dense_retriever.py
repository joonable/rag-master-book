import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# 1. 환경 변수 로드
load_dotenv()

def main():
    # 2. 문서 로드 또는 캐시된 데이터 불러오기
    file_path = "data/투자설명서.pdf"
    save_path = "data/투자설명서_chunks.pkl"
    faiss_index_path = "data/faiss_index"
    
    # 2.1 분할된 문서 로드
    docs = []
    if os.path.exists(save_path):
        print(f"재사용: [{save_path}]에서 기존 분할된 문서를 로드합니다.")
        with open(save_path, "rb") as f:
            docs = pickle.load(f)
    else:
        # PDF 파일이 존재하지 않는 경우 에러 처리 (chunks.pkl도 없는 경우)
        if not os.path.exists(file_path):
            print(f"에러: {file_path} 또는 {save_path} 파일이 존재하지 않습니다.")
            return

        print(f"신규 로드: [{file_path}] PDF 로드 및 분할 중...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # RecursiveCharacterTextSplitter를 이용한 분할 (03_1과 동일하게 설정)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        # 분할된 문서 저장
        with open(save_path, "wb") as f:
            pickle.dump(docs, f)
        print(f"문서 분할 및 저장 완료 ({save_path}): {len(docs)} 개의 청크 생성")

    # 3. 임베딩 모델 및 FAISS 인덱싱
    # 저렴하고 효율적인 text-embedding-3-small 사용
    # chunk_size를 설정하여 한 번의 API 요청에 포함될 문서 수를 제한 (토큰 제한 오류 방지)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=100)

    if os.path.exists(faiss_index_path):
        print(f"재사용: [{faiss_index_path}]에서 기존 FAISS 인덱스를 로드합니다.")
        vectorstore = FAISS.load_local(
            faiss_index_path, 
            embeddings, 
            allow_dangerous_deserialization=True  # 로컬에 저장된 데이터이므로 True 설정
        )
    else:
        print("신규 FAISS 인덱싱 시작 (OpenAI Embeddings 사용)...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        # 인덱스 저장 (나중에 재사용 가능하도록)
        vectorstore.save_local(faiss_index_path)
        print(f"FAISS 인덱싱 및 저장 완료 ({faiss_index_path})")

    # retriever 생성 (k=2)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print("FAISSRetriever 설정 완료 (k=2)")

    # 4. LLM 및 RetrievalQA 설정
    # gpt-4o 모델 사용, 정확도를 위해 temperature=0 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_retriever,
        return_source_documents=True
    )

    # 5. 질문 및 답변 테스트 (사용자 입력)
    print("\n" + "="*50)
    print("질문을 입력하세요. (종료하려면 'exit' 또는 'quit' 입력)")
    print("="*50)

    while True:
        query = input("\n질문: ").strip()
        
        if query.lower() in ["exit", "quit"]:
            print("프로그램을 종료합니다.")
            break
        
        if not query:
            continue

        print(f"\n--- 질문 처리 중: {query} ---")
        
        # RetrievalQA 실행
        response = qa_chain.invoke({"query": query})
        
        print(f"\n답변:\n{response['result']}")
        
        print("\n--- 참조 문서 정보 ---")
        for i, doc in enumerate(response['source_documents']):
            source = doc.metadata.get('source', 'N/A')
            page = doc.metadata.get('page', 'N/A')
            print(f"[{i+1}] 출처: {source} (페이지: {page})")
            # 내용 일부 출력
            content_preview = doc.page_content.replace('\n', ' ')[:150]
            print(f"   내용 요약: {content_preview}...")
            print("-" * 50)

if __name__ == "__main__":
    main()
