import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

# 1. 환경 변수 로드
load_dotenv()

# [Step 1] Pydantic 기반 RelevanceScore 클래스 생성
class RelevanceScore(BaseModel):
    score: float = Field(description="문서와 질문의 관련성 점수 (1~10점)")
    reason: str = Field(description="해당 점수를 부여한 짧은 이유")

# [Step 2] 리랭킹 로직 정의
def reranking_documents(query: str, docs: List[Document], llm: ChatOpenAI) -> List[Document]:
    parser = JsonOutputParser(pydantic_object=RelevanceScore)
    
    rerank_prompt = PromptTemplate(
        template="""당신은 전문 리랭커입니다. 질문과 검색된 문서의 관련성을 평가해 주세요.
반드시 제공된 JSON 형식을 지켜주세요.

[질문]
{query}

[문서 내용]
{context}

{format_instructions}""",
        input_variables=["query", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    rerank_chain = rerank_prompt | llm | parser
    
    scored_docs = []
    print(f"\n📡 [2단계] LLM 리랭킹 수행 (4개 -> 2개 선별)")
    
    for i, doc in enumerate(docs):
        try:
            result = rerank_chain.invoke({"query": query, "context": doc.page_content})
            score = result.get("score", 0.0)
            reason = result.get("reason", "")
            
            doc.metadata["relevance_score"] = score
            scored_docs.append((score, doc))
            print(f"   - 후보 {i+1} 점수: {score}점 | 이유: {reason[:40]}...")
        except Exception:
            scored_docs.append((0.0, doc))
    
    # 점수 기준 내림차순 정렬 후 상위 2개 추출
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_2_docs = [doc for _, doc in scored_docs[:2]]
    
    print(f"   ✅ 상위 2개 문서 선별 완료")
    return top_2_docs

# [Step 3] 커스텀 리트리버 클래스
class LLMRerankRetriever(BaseRetriever):
    vectorstore: FAISS
    llm: ChatOpenAI
    k_initial: int = 4 

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. FAISS 기반 1차 검색 (4개 추출)
        print(f"\n🔍 [1단계] 초기 벡터 검색 수행 (k={self.k_initial})")
        initial_docs = self.vectorstore.similarity_search(query, k=self.k_initial)
        
        # 2. LLM 기반 리랭킹 (최종 2개 선택)
        final_docs = reranking_documents(query, initial_docs, self.llm)
        
        return final_docs

def main():
    # 설정 및 데이터 로드
    faiss_index_path = "data/faiss_index"
    if not os.path.exists(faiss_index_path):
        print("❌ 기존 FAISS 인덱스가 없습니다. 03_2 스크립트를 먼저 실행해 주세요.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 커스텀 리트리버 및 최종 답변용 프롬프트 설정
    rerank_retriever = LLMRerankRetriever(vectorstore=vectorstore, llm=llm)

    final_qa_prompt = PromptTemplate.from_template("""다음 제공된 문맥(Context)을 사용하여 질문에 답하세요. 
문맥에 없는 내용은 답하지 마세요. 최대한 정중하고 상세하게 답변하세요.

[Context]
{context}

[Question]
{question}

[Helpful Answer]:""")

    # [Step 4] 최종 RetrievalQA 구성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=rerank_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": final_qa_prompt}
    )

    print("\n" + "="*60)
    print("🚀 4.1 고성능 LLM 기반 리랭킹 실습 (Step-by-Step)")
    print("="*60)

    while True:
        query = input("\n❓ 질문 입력 (종료: q): ").strip()
        if query.lower() in ['q', 'quit', 'exit']: break
        if not query: continue

        # 1~2단계는 qa_chain.invoke 내부의 리트리버에서 수행됨 (출력은 리트리버 내 print문이 담당)
        response = qa_chain.invoke({"query": query})
        
        # [3단계] 최종 프롬프트 출력 (선별된 2개 문서가 합쳐진 상태 시각화)
        print(f"\n📝 [3단계] 최종 프롬프트 구성 (선별된 2개 문서 포함)")
        context_text = "\n\n".join([doc.page_content for doc in response["source_documents"]])
        full_prompt = final_qa_prompt.format(context=context_text, question=query)
        print("-" * 50)
        print(f"{full_prompt[:500]}...\n(중략)\n...{full_prompt[-100:]}")
        print("-" * 50)

        # [4단계] 최종 답변 출력
        print(f"\n✨ [4단계] 최종 답변")
        print("=" * 50)
        print(response['result'])
        print("=" * 50)

if __name__ == "__main__":
    main()
