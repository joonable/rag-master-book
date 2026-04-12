import os
import pickle
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from dotenv import load_dotenv

load_dotenv()

def visualize_hierarchy():
    persist_directory = "chapter_04/data/chroma_db"
    parent_store_path = "chapter_04/data/parent_store"
    
    # 1. 저장소 로드
    vectorstore = Chroma(
        collection_name="invest_guide",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    store = LocalFileStore(parent_store_path)
    
    # 2. 부모 ID 하나 선택
    parent_keys = list(store.yield_keys())
    if not parent_keys:
        print("데이터가 없습니다. 먼저 인덱싱을 수행하세요.")
        return

    target_parent_id = parent_keys[0]
    
    # 3. 해당 부모의 내용 가져오기
    parent_bytes = store.mget([target_parent_id])[0]
    parent_doc = pickle.loads(parent_bytes)
    
    # 4. 해당 부모에 속한 자식들 모두 찾기 (Chroma에서 doc_id로 필터링)
    # Chroma의 get 메서드는 where 절을 지원합니다.
    children = vectorstore.get(where={"doc_id": target_parent_id})
    
    print("\n" + "="*80)
    print(f"🏠 [PARENT DOCUMENT] - ID: {target_parent_id}")
    print("="*80)
    print(parent_doc.page_content)
    print("-" * 80)
    
    print(f"\n👶 [CHILD DOCUMENTS] - 총 {len(children['ids'])}개의 조각으로 나뉨")
    print("="*80)
    
    for i, content in enumerate(children['documents']):
        print(f"👉 자식 #{i+1} (검색 단위):")
        # 자식 문서는 짧으므로 전체 출력
        print(f"   \"{content.replace(os.linesep, ' ')}\"")
        print("-" * 40)

if __name__ == "__main__":
    visualize_hierarchy()
