import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# [핵심: Input과 Output의 학습 구조]
# ----------------------------------------------------------------------------------
# SFT(Supervised Fine-Tuning)에서 모델은 전체 문장을 읽지만, 
# 'Assistant'가 말하는 시점부터의 토큰들을 맞추는 데 집중합니다.
#
# 1. [INPUT: 학습의 조건 (Prompt)]
#    - <|im_start|>system\n...<|im_end|>
#    - <|im_start|>user\n참고 문서:\n[문서 1]...\n질문:...\n<|im_end|>
#    => 여기까지는 모델이 '이해'해야 하는 상황 설명입니다. (Loss 계산 제외 가능)
#
# 2. [OUTPUT: 학습의 목표 (Completion/Label)]
#    - <|im_start|>assistant\n[근거 추출]... [추론]... [최종 답변]...<|im_end|>
#    => 여기서부터가 모델이 '똑같이 따라 해야 하는' 진짜 공부 내용입니다.
#    => 모델은 앞선 [INPUT]이 주어졌을 때, 이 [OUTPUT]을 출력할 확률을 극대화하도록 학습됩니다.
# ----------------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = "chapter_08/raft_training_data.jsonl"
OUTPUT_DIR = "chapter_08/qwen-raft-lora"

def formatting_prompts_func(examples):
    """
    [03번 데이터셋 -> 04번 학습용 프롬프트 변환 로직]
    
    매핑 관계:
    1. examples["instruction"] (03번 생성) -> 변수 'inst' -> 모델의 '질문' 섹션
    2. examples["context"]     (03번 생성) -> 변수 'ctx'  -> 모델의 '참고 문서' 섹션
    3. examples["cot_answer"]  (03번 생성) -> 변수 'ans'  -> 모델이 맞추어야 할 '정답(Output)'
    """
    instructions = examples["instruction"]
    contexts = examples["context"]
    answers = examples["cot_answer"]
    
    texts = []
    for inst, ctx, ans in zip(instructions, contexts, answers):
        # [과정 1] 03번의 리스트형 context를 번호가 붙은 하나의 문자열로 합침
        combined_ctx = "\n\n".join([f"[문서 {i+1}] {c}" for i, c in enumerate(ctx)])
        
        # [과정 2] 모델이 학습할 '전체 문장(Input+Output)'을 완성
        full_text = (
            f"<|im_start|>system\n당신은 제공된 문서를 바탕으로 추론하여 답변하는 AI입니다.<|im_end|>\n"
            f"<|im_start|>user\n참고 문서:\n{combined_ctx}\n\n질문: {inst}<|im_end|>\n" # <-- 여기까지가 Input
            f"<|im_start|>assistant\n{ans}<|im_end|>" # <-- 여기가 학습해야 할 Output(정답)
        )
        texts.append(full_text)
        
    return { "text": texts }

def train():
    # 양자화 및 모델 로드 (메모리 최적화)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA 설정: 어느 부위를 튜닝할지 결정
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # 데이터셋 로드 및 변환
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 학습 하이퍼파라미터
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=True, 
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )

    # SFTTrainer: Input과 Output이 합쳐진 'text' 필드를 감시하며 학습 진행
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text", # 위에서 만든 full_text가 들어있는 필드
        max_seq_length=2048,
        args=training_args,
        peft_config=peft_config,
    )

    # [학습의 핵심 메커니즘]
    # SFTTrainer는 'text' 내부의 <|im_start|>assistant 뒤에 나오는 텍스트들을 
    # 모델이 생성하도록(Next Token Prediction) 강력하게 유도합니다.
    # 이를 통해 모델은 '참고 문서'를 어떻게 요리해서 'CoT 답변'을 내놓을지 배우게 됩니다.

    print("[*] 학습 시작: 질문(Input)에 대해 정답(Output)을 생성하도록 모델을 최적화합니다.")
    trainer.train()

    # 최종 LoRA 아답터 저장
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[!] 학습 완료! 'Input->Output' 매핑 능력이 탑재된 모델이 {OUTPUT_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    if os.path.exists(DATASET_PATH):
        train()
    else:
        print(f"[ERROR] '{DATASET_PATH}' 파일을 찾을 수 없습니다. 03번 스크립트를 먼저 실행하세요.")
