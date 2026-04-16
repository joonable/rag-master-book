import os
import json
import random
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 로드 (.env 파일에 OPENAI_API_KEY 필요)
load_dotenv()

class RAFTDatasetGenerator:
    """
    RAFT(Retrieval-Augmented Fine-Tuning) 데이터셋 생성 파이프라인
    
    [데이터 처리 5단계 공정]
    1단계: 지식 쪼개기 (Chunking)
       - 방대한 문서를 모델이 학습하기 좋은 크기(예: 1,000자)로 자릅니다. 
       - 이 조각들이 나중에 '정답 문서' 혹은 '방해 문서'의 재료가 됩니다.
       
    2단계: 정답지 추출 (Oracle & CoT Generation)
       - 특정 조각(Oracle)을 GPT-4o에게 주고, 여기서만 풀 수 있는 '질문'과 
         논리적 근거가 담긴 '생각의 사슬(CoT) 답변'을 생성합니다.
         
    3단계: 독약 주입 (Negative Sampling / Distractors)
       - 정답 조각 외에 아무 상관 없는 다른 조각들(Distractors)을 무작위로 선택합니다.
       - 이는 모델에게 '관련 없는 정보에 낚이지 않는 법'을 가르치는 핵심 과정입니다.
       
    4단계: 정보 뒤섞기 (Shuffling & Noise Injection)
       - 정답 문서와 방해 문서들을 한데 섞고 순서를 무작위로 바꿉니다.
       - 모델이 "첫 번째 문서가 항상 정답이야"라는 식의 편법을 배우지 못하게 합니다.
       
    5단계: 훈련용 포맷팅 (Structuring for SFT)
       - 질문, 뒤섞인 문서들, CoT 답변을 하나의 JSONL 라인으로 구성하여 
         학습 코드(04_xxx.py)가 바로 읽을 수 있는 형태로 저장합니다.

    [실제 데이터 구성 예시 (JSONL 한 줄)]
    {
      "instruction": "복리 효과의 핵심 원리는 무엇인가요?",
      "context": [
        "분산 투자는 위험을 낮추는 방법입니다.", // Distractor (방해 문서)
        "복리는 이자에 이자가 붙는 원리로, 시간이 갈수록 자산이 기하급수적으로 늘어납니다.", // Oracle (정답 문서)
        "채권은 정부나 기업에 돈을 빌려주고 이자를 받는 증권입니다." // Distractor (방해 문서)
      ],
      "oracle_context": "복리는 이자에 이자가 붙는 원리로...",
      "cot_answer": "[근거 추출] 문서 2에 따르면 '복리는 이자에 이자가 붙는 원리'라고 명시되어 있습니다. [추론] 이는 초기 원금뿐만 아니라 발생한 이자에서도 추가 수익이 발생함을 의미하며, 시간이 지남에 따라 성장 속도가 가속화됩니다. [최종 답변] 복리 효과의 핵심 원리는 발생한 수익이 다시 재투자되어 추가 수익을 만들어내는 '수익의 재투자' 과정입니다."
    }
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )

    def load_and_split(self, file_path: str) -> List[str]:
        """[1단계] 문서를 로드하고 의미 단위의 청크로 분할합니다."""
        loader = TextLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return [chunk.page_content for chunk in chunks]

    def generate_qna_and_cot(self, context: str) -> Dict:
        """[2단계] 선택된 청크(Oracle)를 기반으로 고품질 Q&A와 추론 과정을 생성합니다."""
        prompt = ChatPromptTemplate.from_template("""
        당신은 RAFT 학습 데이터를 만드는 전문가입니다. 
        주어진 [Context]를 바탕으로 다음 작업을 수행하세요.

        1. 질문 생성: 이 컨텍스트에서만 답을 찾을 수 있는 구체적인 질문을 만드세요.
        2. CoT 답변 생성: 
           - 먼저 컨텍스트에서 정답의 근거가 되는 문장을 직접 추출하세요.
           - 그 근거를 바탕으로 논리적인 추론 과정을 서술하세요.
           - 마지막으로 최종 답변을 작성하세요.

        형식:
        질문: <질문 내용>
        생각의사슬: <[근거 추출] -> [추론] -> [최종 답변] 형식의 상세한 답변>

        [Context]
        {context}
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context})
        
        lines = response.split("\n")
        question = lines[0].replace("질문: ", "").strip()
        cot_answer = "\n".join(lines[1:]).replace("생각의사슬: ", "").strip()
        
        return {"question": question, "answer": cot_answer}

    def create_dataset(self, file_path: str, num_distractors: int = 3) -> List[Dict]:
        """전체 공정을 통합하여 RAFT 데이터셋을 구축합니다."""
        print(f"[*] 1단계 시작: 문서 로드 및 분할 중...")
        chunks = self.load_and_split(file_path)
        dataset = []

        for i, oracle_chunk in enumerate(chunks):
            print(f"[*] {i+1}/{len(chunks)} 번째 데이터 포인트 공정 진행 중...")
            
            # [2단계] 질문 및 CoT 답변 생성
            qna = self.generate_qna_and_cot(oracle_chunk)
            
            # [3단계] 방해 요소(Distractors) 무작위 추출
            distractors = random.sample(
                [c for j, c in enumerate(chunks) if i != j], 
                min(num_distractors, len(chunks) - 1)
            )
            
            # [4단계] 데이터 결합 및 순서 뒤섞기 (Noise Injection)
            combined_context = [oracle_chunk] + distractors
            random.shuffle(combined_context)
            
            # [5단계] 최종 구조화
            data_point = {
                "instruction": qna["question"],
                "context": combined_context,
                "oracle_context": oracle_chunk,
                "cot_answer": qna["answer"]
            }
            dataset.append(data_point)
            
        return dataset

    def save_to_jsonl(self, dataset: List[Dict], output_file: str):
        """[결과 저장] 완성된 문제집을 JSONL 파일로 출력합니다."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[!] 데이터셋 저장 완료: {output_file}")

if __name__ == "__main__":
    input_file = "data/How_to_invest_money.txt"
    output_file = "chapter_08/raft_training_data.jsonl"
    
    if os.path.exists(input_file):
        generator = RAFTDatasetGenerator()
        raft_data = generator.create_dataset(input_file)
        generator.save_to_jsonl(raft_data, output_file)
    else:
        print(f"[ERROR] 입력 파일({input_file})이 없습니다. data 디렉토리를 확인하세요.")
