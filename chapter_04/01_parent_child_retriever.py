import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import BaseStore

# 1. 환경 변수 로드
load_dotenv()

# Document 객체를 바이트로 변환하기 위한 어댑터 클래스
class PersistentDocStore(BaseStore[str, bytes]):
    def __init__(self, path):
        self.store = LocalFileStore(path)

    def mget(self, keys):
        docs = []
        for key in keys:
            value = self.store.mget([key])[0]
            if value:
                docs.append(pickle.loads(value))
            else:
                docs.append(None)
        return docs

    def mset(self, key_value_pairs):
        serialized_pairs = [(k, pickle.dumps(v)) for k, v in key_value_pairs]
        self.store.mset(serialized_pairs)

    def mdelete(self, keys):
        self.store.mdelete(keys)

    def yield_keys(self, prefix=None):
        yield from self.store.yield_keys(prefix)

def get_parent_child_retriever():
    persist_directory = "./.chroma"
    parent_store_path = "data/parent_store"
    
    embeddings = OpenAIEmbeddings()

    # 2. 벡터 데이터베이스(Chroma) 설정
    vectorstore = Chroma(
        collection_name="invest_guide",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # 3. 부모 문서 저장소 설정
    store = PersistentDocStore(parent_store_path)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # 4. 리트리버 초기화
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    return retriever, vectorstore

def main():
    file_path = "data/How_to_invest_money.txt"
    retriever, vectorstore = get_parent_child_retriever()
    
    # 5. 파일명 기준 인덱싱 여부 확인
    existing_docs = vectorstore.get(where={"source": file_path})
    
    if not existing_docs or len(existing_docs['ids']) == 0:
        print(f"[{file_path}] 인덱싱을 시작합니다...")
        loader = TextLoader(file_path)
        docs = loader.load()
        retriever.add_documents(docs, ids=None)
        print(f"[{file_path}] 인덱싱 완료.")
    else:
        print(f"[{file_path}]는 이미 인덱싱되어 있습니다. (건너뜀)")

    print("\n" + "⭐"*25)
    print("  부모-자식 리트리버 실습 모드")
    print("  (종료하려면 'q'를 입력하세요)")
    print("⭐"*25)

    # 6. 대화형 질의 루프
    while True:
        try:
            # 첫 번째 입력: 질문
            query = input("\n💬 질문 입력: ").strip()
            
            if query.lower() in ['exit', 'q', 'quit']:
                print("\n실습을 종료합니다. 즐거운 학습 되세요!")
                break
            
            if not query:
                continue

            # 두 번째 입력: k값
            k_input = input("🔢 검색 개수(k) 입력 (기본값: 1): ").strip()
            try:
                k = int(k_input) if k_input else 1
            except ValueError:
                print("  ⚠️ 숫자를 입력해주세요. 기본값 1로 설정합니다.")
                k = 1

            print(f"\n🔍 검색 중: '{query}' (k={k})...")
            
            # 지정된 k만큼 자식 조각 검색
            child_docs = vectorstore.similarity_search(query, k=k)
            
            if not child_docs:
                print("  ❌ 연관된 문서를 찾지 못했습니다.")
                continue

            print(f"\n" + "🚀" + " ="*25)
            print(f"  [검색 결과 보고서] 총 {len(child_docs)}개의 자식 조각 발견")
            print(" " + " ="*25)
            
            seen_parent_ids = set()
            
            for i, child in enumerate(child_docs):
                print(f"\n({i+1}) --------------------------------------------------")
                print(f"  🔍 [검색된 자식 조각 (Child)]")
                print(f"  > 내용: {child.page_content[:200].replace(os.linesep, ' ')}...")
                
                parent_id = child.metadata.get("doc_id")
                if parent_id:
                    print(f"  |")
                    print(f"  └── 🏠 [연결된 부모 문서 (Parent)]")
                    
                    if parent_id not in seen_parent_ids:
                        seen_parent_ids.add(parent_id)
                        parent_docs = retriever.docstore.mget([parent_id])
                        if parent_docs and parent_docs[0]:
                            parent_doc = parent_docs[0]
                            # 부모 내용은 들여쓰기를 더 주어 구분
                            content = parent_doc.page_content.replace('\n', '\n      ')
                            print(f"      ID: {parent_id}")
                            print(f"      내용:\n      {content[:600]}...")
                    else:
                        print(f"      (위에서 이미 출력된 부모 문서와 동일합니다.)")

            print(f"\n" + "🚀" + " ="*25 + "\n")

        except KeyboardInterrupt:
            print("\n\n실습을 종료합니다.")
            break

if __name__ == "__main__":
    main()
