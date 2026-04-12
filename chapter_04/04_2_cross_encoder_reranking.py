import os
from typing import List, Any, Optional
from dotenv import load_dotenv

# PyTorch 관련 라이브러리 경고 방지 (중복 로드 문제 해결)
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

class RetrieverWithCrossEncoder(BaseRetriever):
    """
    1차로 VectorDB에서 검색된 후보 문서들을 
    Cross-Encoder 모델을 통해 정교하게 리랭킹(Re-ranking)하는 커스텀 리트리버입니다.
    
    원리:
    1. Vector Search: 질문과 문서의 임베딩 유사도 기반으로 후보군 추출 (빠름)
    2. Cross-Encoder: 질문과 문서를 동시에 입력받아 실제 관련성을 정밀 계산 (정확함)
    """
    vectorstore: Any
    cross_encoder: Any
    k_initial: int = 4  # 1차로 가져올 문서 수
    k_final: int = 2    # 리랭킹 후 최종 선택할 문서 수

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # [1단계] 벡터 검색을 통해 1차 후보 문서 추출
        print(f"\n🔍 [1단계] 벡터 DB에서 '{query}' 관련 후보 {self.k_initial}개 검색")
        initial_docs = self.vectorstore.similarity_search(query, k=self.k_initial)
        
        if not initial_docs:
            return []

        # [2단계] Cross-Encoder를 사용한 리랭킹 수행
        print(f"📡 [2단계] Cross-Encoder 리랭킹 수행 ({self.k_initial}개 -> {self.k_final}개 선별)")
        return self._rerank_documents(query, initial_docs)

    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """질문과 문서 리스트를 받아 점수를 계산하고 정렬하여 반환합니다."""
        # (질문, 문서내용) 쌍의 리스트 생성
        pairs = [[query, doc.page_content] for doc in docs]
        
        # Cross-Encoder 점수 예측 (관련성이 높을수록 높은 점수)
        scores = self.cross_encoder.predict(pairs)
        
        # 문서와 점수를 매칭하여 리스트 생성 및 정렬
        scored_docs = []
        for i, score in enumerate(scores):
            doc = docs[i]
            doc.metadata["cross_encoder_score"] = float(score) # 메타데이터에 점수 저장
            scored_docs.append((score, doc))
            print(f"   - 후보 {i+1} 점수: {score:.4f} | 내용: {doc.page_content[:40]}...")

        # 점수 기준 내림차순 정렬 (높은 점수가 위로)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 k_final개만 선별
        final_docs = [doc for _, doc in scored_docs[:self.k_final]]
        print(f"   ✅ 최종 {self.k_final}개 문서 선정 완료")
        
        return final_docs

class CrossEncoderRAG:
    """리랭킹 파이프라인의 설정 및 실행을 관리하는 클래스입니다."""
    
    def __init__(self, faiss_index_path: str):
        self.faiss_path = faiss_index_path
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.cross_encoder = None
        self.qa_chain = None

    def initialize_models(self):
        """필요한 모든 모델 및 벡터 스토어를 로드합니다."""
        print("🛠️ 모델 및 인덱스 로드 중...")
        
        # 1. 임베딩 모델 (벡터 DB 로드용)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 2. FAISS 인덱스 로드
        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError(f"인덱스 경로를 찾을 수 없습니다: {self.faiss_path}")
            
        self.vectorstore = FAISS.load_local(
            self.faiss_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # 3. LLM (최종 답변 생성용)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # 4. Cross-Encoder (리랭킹용 전문 모델)
        # ms-marco-MiniLM 모델은 질문-문서 관련성 판단에 최적화되어 있습니다.
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        print("✅ 모든 모델 로드 완료")

    def build_qa_chain(self):
        """커스텀 리트리버를 포함한 QA 체인을 구성합니다."""
        # 리랭킹 리트리버 초기화
        rerank_retriever = RetrieverWithCrossEncoder(
            vectorstore=self.vectorstore, 
            cross_encoder=self.cross_encoder,
            k_initial=5,  # 5개를 먼저 뽑아서
            k_final=3     # 가장 우수한 3개만 사용
        )

        # 답변 생성을 위한 프롬프트 정의
        prompt = PromptTemplate.from_template("""다음 문맥(Context)만을 사용하여 질문에 답하세요. 
모르는 내용이라면 답변을 지어내지 말고 모른다고 답하세요.

[Context]
{context}

[Question]
{question}

[Answer]:""")

        # RetrievalQA 체인 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=rerank_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def run(self):
        """사용자 대화 루프를 실행합니다."""
        print("\n" + "="*60)
        print("🚀 Cross-Encoder 리랭킹 기반 RAG 시스템 가동")
        print("="*60)

        while True:
            query = input("\n❓ 질문 (종료: q): ").strip()
            if query.lower() in ['q', 'quit', 'exit']: break
            if not query: continue

            # 체인 실행
            result = self.qa_chain.invoke({"query": query})
            
            # 결과 출력
            print(f"\n✨ [3단계] 최종 답변")
            print("-" * 50)
            print(result['result'])
            print("-" * 50)

            # 참조된 문서 정보 확인
            print("\n📚 [참조 문서 및 리랭킹 점수]")
            for i, doc in enumerate(result["source_documents"]):
                score = doc.metadata.get("cross_encoder_score", 0.0)
                print(f"[{i+1}] Score: {score:.4f} | {doc.page_content[:80]}...")

def main():
    faiss_index_path = "chapter_04/data/faiss_index"
    
    try:
        rag = CrossEncoderRAG(faiss_index_path)
        rag.initialize_models()
        rag.build_qa_chain()
        rag.run()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        if "faiss_index" in str(e):
            print("💡 03_2_dense_retriever.py를 먼저 실행하여 인덱스를 생성해주세요.")

if __name__ == "__main__":
    main()
