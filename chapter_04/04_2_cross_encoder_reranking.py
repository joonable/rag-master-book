import os
import pickle
from typing import List, Any
from dotenv import load_dotenv

# PyTorch 관련 라이브러리 경고 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

# 1. 환경 변수 로드
load_dotenv()

# [Step 1] 커스텀 리트리버 클래스 정의 (Cross-Encoder 활용)
class RetrieverWithCrossEncoder(BaseRetriever):
    """
    1차로 VectorDB에서 검색된 문서들을 
    Cross-Encoder 모델을 통해 정교하게 리랭킹하는 커스텀 리트리버
    """
    vectorstore: Any
    cross_encoder: Any
    k_initial: int = 4
    k_final: int = 2

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. 벡터 검색을 통해 1차 후보 문서 추출 (k=4)
        print(f"\n🔍 [1단계] 벡터 DB 검색 수행 (k={self.k_initial})")
        initial_docs = self.vectorstore.similarity_search(query, k=self.k_initial)
        
        if not initial_docs:
            return []

        # 2. Cross-Encoder 리랭킹 (질문과 문서 쌍 생성)
        print(f"📡 [2단계] Cross-Encoder 리랭킹 수행 ({self.k_initial}개 -> {self.k_final}개 선별)")
        
        # (query, context) 리스트 생성
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        # Cross-Encoder 점수 예측
        scores = self.cross_encoder.predict(pairs)
        
        # 문서와 점수를 매칭하여 리스트 생성
        scored_docs = []
        for i, score in enumerate(scores):
            doc = initial_docs[i]
            doc.metadata["cross_encoder_score"] = float(score)
            scored_docs.append((score, doc))
            print(f"   - 후보 {i+1} 점수: {score:.4f} | 내용: {doc.page_content[:40]}...")

        # 점수 기준 내림차순 정렬
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 k_final(2개) 추출
        final_docs = [doc for _, doc in scored_docs[:self.k_final]]
        
        print(f"   ✅ 상위 {self.k_final}개 문서 선별 완료")
        return final_docs

def main():
    # 2. 경로 및 설정
    faiss_index_path = "chapter_04/data/faiss_index"
    
    if not os.path.exists(faiss_index_path):
        print("❌ 기존 FAISS 인덱스가 없습니다. chapter_04/03_2_dense_retriever.py를 먼저 실행해 주세요.")
        return

    # 3. 모델 로드
    print("🛠️ 모델 로드 중 (OpenAI Embeddings, Cross-Encoder)...")
    
    # 임베딩 (FAISS 로드용)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # FAISS 로드
    vectorstore = FAISS.load_local(
        faiss_index_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # LLM (최종 답변용)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Cross-Encoder (리랭킹용)
    # ms-marco-MiniLM-L-12-v2 모델은 질문과 문서의 관련성 점수 계산에 널리 사용됨
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    print("✅ 모델 로드 완료")

    # 4. 커스텀 리트리버 생성
    rerank_retriever = RetrieverWithCrossEncoder(
        vectorstore=vectorstore, 
        cross_encoder=cross_encoder_model,
        k_initial=4,
        k_final=2
    )

    # 5. 최종 QA 체인 설정
    # 리랭킹된 결과를 기반으로 답변을 생성하기 위한 프롬프트
    final_qa_prompt = PromptTemplate.from_template("""다음 제공된 문맥(Context)을 사용하여 질문에 답하세요. 
문맥에 없는 내용은 답하지 마세요. 최대한 정중하고 상세하게 답변하세요.

[Context]
{context}

[Question]
{question}

[Helpful Answer]:""")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=rerank_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": final_qa_prompt}
    )

    print("\n" + "="*60)
    print("🚀 4.2 Cross-Encoder 기반 리랭킹 실습")
    print("="*60)

    # 6. 사용자 질의응답 루프
    while True:
        query = input("\n❓ 질문 입력 (종료: q): ").strip()
        if query.lower() in ['q', 'quit', 'exit']: 
            break
        if not query: 
            continue

        # RetrievalQA 실행 (내부에서 커스텀 리트리버가 리랭킹 수행)
        response = qa_chain.invoke({"query": query})
        
        # 최종 답변 및 결과 출력
        print(f"\n✨ [3단계] 최종 답변")
        print("=" * 50)
        print(response['result'])
        print("=" * 50)

        # 참조 문서 정보 출력
        print("\n📚 [참조 문서 (리랭킹 결과)]")
        for i, doc in enumerate(response["source_documents"]):
            score = doc.metadata.get("cross_encoder_score", 0.0)
            page = doc.metadata.get("page", "N/A")
            print(f"[{i+1}] (Score: {score:.4f}) Page: {page} | {doc.page_content[:100]}...")

if __name__ == "__main__":
    main()
