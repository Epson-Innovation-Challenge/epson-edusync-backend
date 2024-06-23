# FastAPI

![로고](https://tech.osci.kr/wp-content/uploads/2023/09/image-20230920-051212.png)
## 개요

이 FastAPI 애플리케이션은 교육 문서와 학생 답안을 처리하는 두 가지 주요 엔드포인트를 제공합니다. Dropbox와 연동하여 PDF 파일을 다운로드하고 이를 처리하여 점수 보고서와 암기장, 오답노트, 유사문제를 생성합니다.

## 설치
 **필요한 패키지 설치:**
   ```bash
   pip install fastapi uvicorn dropbox pymupdf pydantic
   ```

## 애플리케이션 실행

FastAPI 서버를 시작하려면 다음 명령어를 실행하세요:
```bash
uvicorn main:app --reload
```

## API 엔드포인트

### 1. `POST /MakeStudentInfoScoreCommentary`

#### 설명:
이 엔드포인트는 제공된 PDF 문서와 제출된 답안을 기반으로 학생의 점수 해설을 생성합니다.

#### 요청 본문:
- `document` (str): Dropbox 내 PDF 문서 경로.
- `submit` (List[Answer]): 학생이 제출한 답안 리스트.

#### 예시 요청:
```json
{
  "document": "path/to/document.pdf",
  "submit": [
    {"question": 1, "answer": 2},
    {"question": 2, "answer": 3}
  ]
}
```

#### 응답:
다음 내용을 포함하는 JSON 객체:
- 문제 번호
- 학생이 제출한 답안
- 실제 정답
- 정답 여부
- 각 문제에 대한 요약 해설

### 2. `POST /CreateMemorizationBookAddCommentary`

#### 설명:
이 엔드포인트는 다음과 같은 다양한 학습 자료를 생성합니다:
- 오답 노트 (Output_path1)
- 정답 및 해설이 포함된 암기장 (Output_path2)
- 유사 문제 문서 (Output_path3)

#### 요청 매개변수:
- `Output_path1` (str): 오답 노트 문서 경로 (기본값: `/study/오답노트.docx`).
- `Output_path2` (str): 암기장 문서 경로 (기본값: `/memory/암기장.docx`).
- `Output_path3` (str): 유사 문제 문서 경로 (기본값: `/gen/유사문제.docx`).

#### 예시 요청:
```json
{
  "Output_path1": "/study/error_note.docx",
  "Output_path2": "/memory/memorization_book.docx",
  "Output_path3": "/gen/similar_problems.docx"
}
```

## 함수 설명

### `MakeStudentInfoScoreCommentary`
이 함수는 제공된 PDF에서 질문과 학생 답안을 추출하고 각 답안의 정답 여부를 확인하며 요약 해설을 생성합니다.

### `CreateMemorizationBook`
이 함수는 원본 질문이 포함된 학습 문서를 생성합니다.

### `CreateMemorizationBookAddCommentary`
이 함수는 원본 질문, 정답 및 상세 해설이 포함된 학습 문서를 생성합니다.

### `CreateCorrectAnswerNote`
이 함수는 틀린 답안에 대한 유사 문제를 포함하는 문서를 생성합니다.

---

### 주요 기능

#### 1. `MakeStudentInfo`

PDF 파일에서 텍스트를 추출하고 문제 세부 사항 및 학생 답안을 포함하는 JSON 객체를 생성합니다.

**매개변수:**
- `Question_pdf`: 시험 문제가 포함된 PDF 파일
- `Answer_json`: 학생의 답안을 포함하는 JSON 데이터

**반환값:**
- 문제 세부 사항 및 학생 답안을 포함하는 JSON 객체

#### 2. `MakeScoreCommentary`

학생 답안과 정답을 바탕으로 점수 해설을 포함하는 JSON 객체를 생성합니다.

**매개변수:**
- `QuestionAnswerJson`: 문제 세부 사항 및 학생 답안을 포함하는 JSON 데이터

**반환값:**
- 점수 해설 및 요약을 포함하는 JSON 객체

#### 3. `CreateMemorizationBookAddCommentary`

해설이 추가된 암기장을 생성합니다.

**매개변수:**
- `QuestionAnswer_json`: 문제 세부 사항 및 학생 답안을 포함하는 JSON 데이터
- `AnswerCommentary_json`: 점수 해설 및 요약을 포함하는 JSON 데이터
- `Output_path`: 출력 문서가 저장될 경로

#### 4. `CreateCorrectAnswerNote`

관련 문제와 해설이 포함된 정답 노트를 생성합니다.

**매개변수:**
- `QuestionAnswer_json`: 문제 세부 사항 및 학생 답안을 포함하는 JSON 데이터
- `AnswerCommentary_json`: 점수 해설 및 요약을 포함하는 JSON 데이터
- `Output_path`: 출력 문서가 저장될 경로

### 예제 사용법

다음은 이러한 함수를 사용하는 예제입니다:

```python
from your_module import MakeStudentInfo, MakeScoreCommentary, CreateMemorizationBookAddCommentary, CreateCorrectAnswerNote

# 1단계: PDF에서 학생 정보와 답안을 추출
response1 = MakeStudentInfo(Question_pdf="path/to/your/Question.pdf",
                            Answer_json="path/to/your/answers.json")

# 2단계: 추출된 데이터를 바탕으로 점수 해설 생성
response2 = MakeScoreCommentary(QuestionAnswerJson=response1)

# 3단계: 해설이 추가된 암기장 생성
CreateMemorizationBookAddCommentary(QuestionAnswer_json=response1,
                                    AnswerCommentary_json=response2,
                                    Output_path="path/to/your/output_memorization_book.docx")

# 4단계: 정답 노트 생성
CreateCorrectAnswerNote(QuestionAnswer_json=response1,
                        AnswerCommentary_json=response2,
                        Output_path="path/to/your/output_correct_answer_note.docx")
```

### 코드 실행

Python 스크립트를 실행하여 코드를 실행할 수 있습니다. 필요한 파일이 준비되어 있고 Dropbox 및 OpenAI API 키가 `.env` 파일에 올바르게 설정되어 있는지 확인하십시오.


## 결론

이 프로젝트는 시험 문제와 학생 답안을 처리하고, 유용한 보고서를 생성하며, 클라우드 스토리지와 통합하는 포괄적인 워크플로를 제공합니다. 특정 사용 사례에 맞게 기능을 커스터마이즈하고 확장할 수 있습니다.



