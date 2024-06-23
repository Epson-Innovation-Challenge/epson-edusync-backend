import os
from typing import Union, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import dropbox
import fitz

from EpsonFunc import *

app = FastAPI()
# Set your Dropbox access token and folder path
ACCESS_TOKEN = "EpsonAPI"

# Initialize Dropbox client
dbx = dropbox.Dropbox(ACCESS_TOKEN)

# 문제번호 + 문제 + 첨부된 지문 + 보기 + 학생이 낸 정답 추출
# RAG 이용: 정답 여부 + 해설 추출

class Answer(BaseModel):
    question: int
    answer: int

class Submit_pydantic(BaseModel):
    document: str
    submit: List[Answer]
# Pydantic 모델 정의
class CommentaryResponse(BaseModel):
    Number: int = Field(description="문제 번호")
    StudentAnswer: int = Field(description="학생이 제출한 정답")
    CorrectAnswer: int = Field(description="실제 정답")
    IsCorrect: bool = Field(description="정답 여부")
    CommentarySummarize: List[str] = Field(description="문제 해설 3줄 요약")

@app.post("/MakeStudentInfoScoreCommentary")
def PostMakeStudentInfoScoreCommentary(data: dict):
    data = Submit_pydantic(**data)
    
    #file_path = f"/test/{data['document']}"
    file_path = f"/test/{data.document}"
    _, res = dbx.files_download(file_path)

    with open("temp.pdf", "wb") as f:
        f.write(res.content)
    doc = fitz.open("temp.pdf")
    
    #summary = finetuned_summarize(input_text.text)
    result = MakeStudentInfoScoreCommentary(Question_pdf=doc,
                                            Answer_json=data.submit)
    
    return result

@app.post("/CreateMemorizationBookAddCommentary")
def PostCreateMemorizationBookAddCommentary(Output_path1="/study/오답노트.docx", 
                                            Output_path2="/memory/암기장.docx",
                                            Output_path3="/gen/유사문제.docx"):
    
    CreateMemorizationBook(QuestionAnswer_json="./QuestionAnswer.json",
                       AnswerCommentary_json="./AnswerCommentary.json",
                       Output_path=Output_path1) # 원문제만
    
    CreateMemorizationBookAddCommentary(QuestionAnswer_json="./QuestionAnswer.json",
                       AnswerCommentary_json="./AnswerCommentary.json",
                       Output_path=Output_path2) # 원문제 + 정답 + 해설
    
    CreateCorrectAnswerNote(QuestionAnswer_json="./QuestionAnswer.json",
                       AnswerCommentary_json="./AnswerCommentary.json",
                       Output_path=Output_path3) # 유사문제 생성