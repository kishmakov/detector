from fastapi import APIRouter

from app.models import AnalyzeRequest, AnalyzeResponse
from app.services.text_analysis import count_words

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return AnalyzeResponse(word_count=count_words(request.text))
