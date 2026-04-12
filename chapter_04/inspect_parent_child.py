import os
import pickle
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from dotenv import load_dotenv

load_dotenv()

def inspect_storage():
    persist_directory = "./.chroma"
    parent_store_path = "./parent_store"
    
    # 1. 자식 문서(Chroma) 확인
    vectorstore = Chroma(
        collection_name="invest_guide",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    
    # Chroma에서 모든 데이터 가져오기 (샘플 2개만 출력)
    all_children = vectorstore.get()
    print("\n" + "="*50)
    print(f"검색용 자식 문서 (총 {len(all_children['ids'])}개 중 샘플 2개)")
    print("="*50)
    
    for i in range(min(2, len(all_children['ids']))):
        print(f"[자식 #{i+1}]")
        print(f"ID: {all_children['ids'][i]}")
        print(f"내용 (약 200자): {all_children['documents'][i][:200]}...")
        print(f"메타데이터: {all_children['metadatas'][i]}")
        print("-" * 20)

    # 2. 부모 문서(Docstore) 확인
    store = LocalFileStore(parent_store_path)
    # 저장된 키(부모 ID)들 가져오기
    parent_keys = list(store.yield_keys())
    
    print("\n" + "="*50)
    print(f"LLM 전달용 부모 문서 (총 {len(parent_keys)}개 중 샘플 1개)")
    print("="*50)
    
    if parent_keys:
        parent_id = parent_keys[0]
        parent_bytes = store.mget([parent_id])[0]
        if parent_bytes:
            parent_doc = pickle.loads(parent_bytes)
            print(f"[부모 ID: {parent_id}]")
            print(f"내용 (약 1000자):\n{parent_doc.page_content}")
            print(f"메타데이터: {parent_doc.metadata}")
    else:
        print("저장된 부모 문서가 없습니다.")

if __name__ == "__main__":
    inspect_storage()
