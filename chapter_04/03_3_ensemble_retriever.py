import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from kiwipiepy import Kiwi

# 1. 환경 변수 로드
load_dotenv()

# Kiwi 형태소 분석기 초기화 (BM25 토큰화용)
kiwi = Kiwi()

def kiwi_tokenize(text):
    """Kiwi를 사용하여 텍스트를 토큰화하는 함수"""
    return [token.form for token in kiwi.tokenize(text)]

def main():
    # 2. 문서 및 경로 설정
    file_path = "data/투자설명서.pdf"
    save_path = "data/투자설명서_chunks.pkl"
    faiss_index_path = "data/faiss_index"
    
    # 2.1 분할된 문서 로드 (기존 데이터 재사용)
    docs = []
    if os.path.exists(save_path):
        print(f"재사용: [{save_path}]에서 기존 분할된 문서를 로드합니다.")
        with open(save_path, "rb") as f:
            docs = pickle.load(f)
    else:
        if not os.path.exists(file_path):
             print(f"에러: {file_path} 파일이 존재하지 않습니다.")
             return
        print(f"신규 로드: [{file_path}] PDF 로드 및 분할 중...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        with open(save_path, "wb") as f:
            pickle.dump(docs, f)

    # 3. 각 리트리버 초기화
    
    # 3.1 Sparse Retriever (BM25)
    print("BM25Retriever 설정 중 (Kiwi 토큰화)...")
    bm25_retriever = BM25Retriever.from_documents(
        docs, 
        preprocess_func=kiwi_tokenize
    )
    bm25_retriever.k = 2

    # 3.2 Dense Retriever (FAISS)
    print("FAISSRetriever 로드 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(
            faiss_index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        print("신규 FAISS 인덱싱 진행...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(faiss_index_path)
    
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 4. Ensemble Retriever 구성
    # BM25와 FAISS 결과를 5:5 비율로 결합 (RRF 방식)
    print("EnsembleRetriever 구성 중 (Weight [0.5, 0.5])...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    # 5. LLM 및 QA 체인 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # RetrievalQA에서 ensemble_retriever 사용
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever,
        return_source_documents=True
    )

    # 6. 질문 및 답변 테스트
    print("\n" + "="*50)
    print("질문을 입력하세요. (종료하려면 'exit' 또는 'quit' 입력)")
    print("="*50)

    while True:
        query = input("\n질문: ").strip()
        
        if query.lower() in ["exit", "quit"]:
            break
        
        if not query:
            continue

        print(f"\n--- 앙상블 검색 및 처리 중: {query} ---")
        
        # QA 체인 실행 (내부적으로 EnsembleRetriever 호출)
        response = qa_chain.invoke({"query": query})
        
        print(f"\n답변:\n{response['result']}")
        
        print("\n--- 앙상블 검색된 참조 문서 (RRF 결합 결과) ---")
        for i, doc in enumerate(response['source_documents']):
            source = doc.metadata.get('source', 'N/A')
            page = doc.metadata.get('page', 'N/A')
            print(f"[{i+1}] 출처: {source} (페이지: {page})")
            content_preview = doc.page_content.replace('\n', ' ')[:150]
            print(f"   내용 요약: {content_preview}...")
            print("-" * 50)

if __name__ == "__main__":
    main()
