# Project Context: rag-master-book

본 프로젝트는 **'RAG Master'** 책의 튜토리얼을 실습하기 위한 저장소입니다. 사용자는 각 챕터별 내용을 바탕으로 Gemini를 활용하여 필요한 기능을 구현하고 학습합니다.

## 프로젝트 개요

- **목적:** 검색 증강 생성(RAG)의 기초부터 고도화 전략, 멀티모달, 그래프 RAG, 파인튜닝까지 실습을 통해 마스터함.
- **주요 기술:** Python, LangChain, LangGraph, Vector Databases (Chroma, FAISS), Neo4j, LLM Fine-tuning.
- **의존성 관리:** `uv`를 사용하여 빠르고 일관된 환경을 관리함.

## 실습 목차 (Chapters)

📖 **Chapter 1. 랭체인 살펴보기**
- 개요, LLM 파라미터, LCEL, 프롬프트, 출력 파서, 메모리 관리

🔍 **Chapter 2. 검색 증강 생성 기초와 실습**
- 임베딩, 문서 로더, 텍스트 분할, 벡터 DB(Chroma, FAISS), Streamlit UI

🧩 **Chapter 3. 멀티모달 RAG를 활용한 복합 데이터 처리**
- 멀티모달 전략 및 구현, 멀티-벡터 검색기

🚀 **Chapter 4. 검색과 응답을 최적화하는 RAG 고도화 전략**
- 부모-자식 분할, 질의 변형(Multi-query, HyDE), 앙상블 검색, 리랭킹, Self-RAG

🕸️ **Chapter 5. 지식 그래프를 활용한 그래프 RAG**
- 지식 그래프 개요, Neo4j 구축 및 통합, GraphRAG 질의

⚙️ **Chapter 6. 랭그래프로 설계하는 RAG 파이프라인**
- 랭그래프 구성요소(노드, 에지, 상태), 제어 흐름(루프, 조건문), 자체교정-RAG

🤖 **Chapter 7. 리액트 에이전트를 활용한 RAG**
- Chain of Thought, 에이전트 도구 및 프롬프트 설정, 에이전트 RAG 실습

🧠 **Chapter 8. RAG 성능을 높이는 LLM 파인튜닝**
- RAFT 방법론, Qwen 모델 파인튜닝(LoRA), 데이터 전처리 및 학습 환경(RunPod)

🎯 **Chapter 9. 임베딩 모델 파인튜닝**
- 대조 학습, MNR Loss, 하드 네거티브 선정, 합성 데이터 생성 및 평가

## 개발 규칙 및 컨벤션

- **의존성 추가:** 새로운 패키지가 필요할 경우 `uv add <package>`를 사용함.
- **구현 방식:** 세부 챕터별로 관심 있는 내용을 Gemini CLI를 활용하여 단계적으로 구현함.
- **중간 과정 출력:** 작동 과정을 직관적으로 이해할 수 있도록 체인의 중간 결과나 처리 단계별 정보를 상세하게 출력함.
- **언어:** 모든 코드 주석 및 문서는 한글을 기본으로 함.
- **테스트:** 구현된 RAG 파이프라인의 응답 정확성을 검증하는 테스트 코드를 포함함.

## 실행 및 설치

- **환경 설정:**
  ```bash
  uv venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate
  uv sync
  ```
- **실습 실행:** 각 챕터별 디렉토리에서 관련 스크립트 실행.
