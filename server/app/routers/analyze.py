import datetime
import os
import pathlib

from fastapi import APIRouter

from app.models import AnalyzeRequest, AnalyzeResponse
from app.services.text_analysis import count_words

router = APIRouter()

_log_path = os.environ.get("LOG_FILE")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    word_count = count_words(request.text)
    if _log_path:
        log = pathlib.Path(_log_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            snippet = request.text[:80].replace("\n", " ")
            f.write(f'{datetime.datetime.utcnow().isoformat()} | words={word_count} | text="{snippet}"\n')
    return AnalyzeResponse(word_count=word_count)
