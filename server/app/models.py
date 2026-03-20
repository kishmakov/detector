from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    word_count: int
