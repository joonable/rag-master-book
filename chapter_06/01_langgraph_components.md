# 🧩 1. 랭그래프의 구성요소 (Components of LangGraph)

기존의 LangChain(LCEL)이 단방향 유향 비순환 그래프(DAG) 구조로 설계되어 '루프(Loop)'나 '반복'을 구현하기 어려웠던 한계를 극복하기 위해 등장한 것이 **LangGraph**입니다. 랭그래프는 복잡한 에이전트의 사고 과정을 상태(State) 기반의 순환(Cycle) 구조로 설계할 수 있게 해주는 핵심 아키텍처를 제공합니다.

---

### 1.1 그래프 (Graph): 워크플로우의 뼈대
전체 프로세스를 정의하는 컨테이너이자 도화지입니다.
*   **StateGraph**: 랭그래프의 심장입니다. 사용할 상태(State)의 타입을 정의하고 노드와 에지를 추가하는 기본 클래스입니다.
*   **Compiled Graph**: `graph.compile()`을 통해 생성된 실행 가능한 객체입니다. 컴파일 과정에서 그래프의 정당성(모든 노드가 연결되었는지 등)을 검증하며, 체크포인터(Checkpointer)를 붙여 '중단 후 재개' 기능을 활성화할 수도 있습니다.

### 1.2 상태 (State): 전역 공유 메모리
모든 노드가 읽고 쓰는 **중앙 집중식 저장소**입니다.
*   **구조 정의**: Python의 `TypedDict`나 `Pydantic`을 사용하여 엄격하게 정의합니다.
*   **업데이트 방식: 덮어쓰기(Overwrite) vs 리듀서(Reducer)**
    상태는 노드가 반환하는 값에 의해 업데이트됩니다. 이때 필드별로 어떻게 업데이트할지 정의할 수 있습니다.

    1.  **기본 방식 (Overwrite)**: 별도의 설정이 없으면 새로운 노드가 반환한 값이 기존 값을 완전히 대체합니다.
    ```python
    from typing import TypedDict

    class MyState(TypedDict):
        # 새로운 노드가 값을 반환하면 기존 문자열을 지우고 새로 씀
        current_step: str 
    ```

    2.  **리듀서 방식 (Reducer)**: `Annotated`와 `operator.add` 등을 사용하여 기존 데이터에 새로운 데이터를 **합치는(Merge/Append)** 방식입니다. 대화 기록(Messages)이나 병렬 작업 결과를 모을 때 필수적입니다.
    ```python
    import operator
    from typing import Annotated, TypedDict, List

    class MyState(TypedDict):
        # operator.add는 리스트의 경우 .append()처럼 기존 리스트 뒤에 새 요소를 붙임
        messages: Annotated[List[str], operator.add]
        
        # 사용자 정의 함수를 리듀서로 사용할 수도 있음
        # (기존 값, 새로운 값)을 입력받아 최종 상태를 반환
        count: Annotated[int, lambda old, new: old + new]
    ```

    *   **왜 중요한가요?**: 랭그래프의 노드는 상태 전체가 아니라 **업데이트하고 싶은 일부 필드**만 반환합니다. 리듀서가 설정된 필드는 "추가"되고, 그렇지 않은 필드는 "교체"되므로 데이터 흐름을 정교하게 제어할 수 있습니다.

---

### 1.3 노드 (Node): 실제 작업의 단위
노드는 현재의 `State`를 입력받아 작업을 수행한 뒤, 업데이트할 `State`의 일부를 반환하는 함수입니다.

| 노드 유형 | 설명 | 상세 특징 |
| :--- | :--- | :--- |
| **작업 노드 (Function Node)** | 사용자가 정의한 일반 파이썬 함수 | LLM 호출, 데이터 가공, API 요청 등 모든 로직이 여기서 일어납니다. |
| **도구 노드 (Tool Node)** | 에이전트가 도구를 실행하도록 돕는 특수 노드 | `langgraph.prebuilt.ToolNode`를 사용하여 LLM이 선택한 도구를 실제로 실행하고 그 결과를 상태에 반영합니다. |
| **시스템 노드 (START/END)** | 그래프의 시작과 끝을 알리는 특수 마커 | `START`는 입력을 처음 받는 지점이며, `END`는 도달 시 실행을 완전히 종료하는 지점입니다. |

---

### 1.4 에지 (Edge): 흐름과 제어권

에지는 노드와 노드 사이를 잇는 통로로, 그래프 내에서 데이터(State)가 흐르는 방향을 결정합니다. 단순히 선을 긋는 것을 넘어, 특정 조건에 따라 경로를 바꾸거나 반복(Loop)을 만드는 지능적인 라우팅 역할을 수행합니다.

#### 1) 일반 에지 (Normal Edge)
가장 기본적인 연결 방식으로, 한 노드의 작업이 끝나면 **무조건 지정된 다음 노드**로 이동합니다. 흐름이 결정론적(Deterministic)일 때 사용합니다.

```python
# 'research' 노드가 끝나면 무조건 'writer' 노드로 이동
workflow.add_edge("research", "writer")
```

#### 2) 조건부 에지 (Conditional Edge)
상태(State)를 분석하여 다음에 갈 노드를 **동적으로 선택**합니다. '라우터(Router)' 함수를 통해 논리적인 분기(If-Else)를 구현할 때 필수적입니다.

```python
def should_continue(state: MyState):
    # 논리에 따라 다음 목적지의 이름을 문자열로 반환
    if state["is_accurate"]:
        return "finish"
    return "re_search"

# 'validate' 노드 이후, should_continue 함수의 결과에 따라 분기
workflow.add_conditional_edges(
    "validate",
    should_continue,
    {
        "finish": END,
        "re_search": "research"
    }
)
```

#### 3) 진입점(START) 및 종료(END) 에지
그래프가 어디서 시작하고 어디서 끝나는지를 정의하는 특수 에지입니다.

```python
from langgraph.graph import START, END

# 그래프의 시작점을 'agent' 노드로 설정
workflow.add_edge(START, "agent")

# 'final_answer' 노드가 끝나면 그래프 실행 종료
workflow.add_edge("final_answer", END)
```

#### 4) 순환 에지 (Cycle / Recursive Edge)
에지가 자기 자신이나 이전 단계의 노드를 가리키게 하여 **반복(Loop)** 구조를 만듭니다. "답변이 만족스러울 때까지 재시도"하는 자가교정(Self-Correction) 로직을 구현할 때 핵심입니다.

```python
# 조건부 에지를 활용한 순환 구조 예시
workflow.add_conditional_edges(
    "generate_answer",
    check_quality,
    {
        "pass": END,
        "fail": "generate_answer"  # 자기 자신으로 돌아가서 다시 생성 (Loop)
    }
)
```

---

### 💡 Insight: 왜 LangGraph인가?

기존 LCEL 체인 방식은 다음과 같은 문제점이 있었습니다:
1.  **데이터 전달의 복잡성**: 단계가 많아질수록 필요한 변수를 뒤로 전달하기 위해 딕셔너리를 계속 가공해야 했습니다.
2.  **순환 로직의 부재**: "답변이 틀리면 다시 시도하라"는 루프를 체인 내부에서 우아하게 표현할 방법이 없었습니다.

**LangGraph**는 **`State`**라는 공용 바구니를 모든 노드가 공유하게 함으로써 데이터 관리를 단순화합니다. 노드는 자기가 할 일만 하고 결과만 바구니에 던지면 됩니다. 에지는 그 바구니의 상태를 보고 다음에 누구에게 일을 시킬지 결정합니다. 이 구조 덕분에 복잡한 자가교정(Self-Correction) RAG나 멀티 에이전트 시스템을 매우 직관적으로 구현할 수 있습니다.

---

### 요약: 구성요소 간의 관계
> **"상태(State)**라는 바구니에 데이터를 담아, **그래프(Graph)**라는 지도 위에서, **노드(Node)**라는 작업자들이 전문적인 일을 수행하고, **에지(Edge)**라는 지능적인 길을 따라 데이터가 흐른다."
