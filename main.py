import os
import re
import time
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

GROQ_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("MODEL", "openai/gpt-oss-120b")

RATE_LIMIT = 20
WINDOW = 3600
_hits: dict[str, list[float]] = defaultdict(list)

FILLER_TAIL = re.compile(
    r"(могут?\s+быть\s+полезны?|пригодятся|помог(ут|ло|ает)|"
    r"внести\s+вклад|соответству\w+\s+требовани)",
    re.IGNORECASE,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class GenRequest(BaseModel):
    vacancy: str = Field(min_length=20, max_length=8000)
    profile: str = Field(min_length=10, max_length=4000)
    tone: str = "живой, по делу, без канцелярита"


def check_limit(ip: str) -> None:
    now = time.time()
    window = [t for t in _hits[ip] if now - t < WINDOW]
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, "Лимит генераций исчерпан, попробуй позже.")
    window.append(now)
    _hits[ip] = window


def strip_filler_tail(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) > 1 and FILLER_TAIL.search(parts[-1]):
        parts = parts[:-1]
    return " ".join(parts).strip()


def build_prompt(vacancy: str, profile: str, tone: str) -> str:
    return (
        "Напиши короткое сопроводительное к отклику на hh.ru. От первого лица, по-русски.\n\n"
        "ГЛАВНОЕ ПРАВИЛО — НИКАКИХ ВЫДУМОК:\n"
        "Используй ТОЛЬКО факты из профиля ниже. Не придумывай результаты и метрики, "
        "если их нет в профиле.\n\n"
        "КАК ЗАКОНЧИТЬ ПИСЬМО:\n"
        "Последнее предложение — про конкретный проект или про готовность показать код на GitHub. "
        "Не заканчивай общими фразами про пользу навыков.\n\n"
        "ЗАПРЕТЫ:\n"
        "- Не начинай с 'Я заинтересован', 'Меня привлекает'.\n"
        "- Не используй канцелярит.\n"
        "- Не пересказывай профиль списком.\n\n"
        "КАК НАДО:\n"
        "- 3-4 живых предложения.\n"
        "- 1-2 конкретных проекта из профиля, релевантных вакансии.\n"
        "- Простой разговорный язык. Без markdown и эмодзи.\n\n"
        "=== ВАКАНСИЯ ===\n"
        + vacancy
        + "\n\n=== ПРОФИЛЬ ===\n"
        + profile
        + "\n\nВерни только текст письма, без пояснений."
    )


@app.post("/generate")
async def generate(req: GenRequest, request: Request):
    if not GROQ_KEY:
        raise HTTPException(503, "На сервере не задан GROQ_API_KEY.")

    check_limit(request.client.host)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": build_prompt(req.vacancy, req.profile, req.tone),
            }
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=40) as client:
            r = await client.post(GROQ_URL, json=payload, headers=headers)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Не удалось подключиться к Groq: {e}") from e

    if r.status_code != 200:
        raise HTTPException(502, f"Ошибка Groq: {r.status_code} {r.text[:300]}")

    data = r.json()
    letter = data["choices"][0]["message"]["content"].strip()
    return {"letter": strip_filler_tail(letter)}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "groq_configured": bool(GROQ_KEY)}
