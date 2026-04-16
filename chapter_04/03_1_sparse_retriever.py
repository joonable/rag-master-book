import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from kiwipiepy import Kiwi

# 1. 환경 변수 로드
load_dotenv()

# Kiwi 형태소 분석기 초기화
kiwi = Kiwi()

def kiwi_tokenize(text):
    """Kiwi를 사용하여 텍스트를 토큰화하는 함수"""
    return [token.form for token in kiwi.tokenize(text)]

def main():
    # 2. 문서 로드 또는 캐시된 데이터 불러오기
    file_path = "data/투자설명서.pdf"
    save_path = "data/투자설명서_chunks.pkl"
    
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

        # RecursiveCharacterTextSplitter를 이용한 분할 (chunk_size=2000, chunk_overlap=200)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        # 분할된 문서 저장
        with open(save_path, "wb") as f:
            pickle.dump(docs, f)
        print(f"문서 분할 및 저장 완료 ({save_path}): {len(docs)} 개의 청크 생성")

    # 3. Kiwi 토큰화 함수를 preprocess_func로 전달하여 BM25Retriever 생성
    print("BM25Retriever 인덱싱 시작 (Kiwi 토큰화 사용)...")
    bm25_retriever = BM25Retriever.from_documents(
        docs, 
        preprocess_func=kiwi_tokenize
    )
    # k값 2로 설정
    bm25_retriever.k = 2
    print("BM25Retriever 설정 완료 (k=2)")

    # 4. LLM 및 RetrievalQA 설정
    # gpt-4o 모델 사용, 정확도를 위해 temperature=0 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=bm25_retriever,
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
