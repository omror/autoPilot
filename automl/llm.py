"""LLM cagrilari icin tek arayuz.

Sistemin geri kalani buraya bakar: anahtar yoksa, paket kurulu degilse
ya da cagri patlarsa hicbir istisna disari sizmaz, sadece None doner.
Boylece LLM katmani her zaman opsiyonel kalir.
"""
import json
import os
from typing import Optional, TypeVar

from pydantic import BaseModel

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 8192

T = TypeVar("T", bound=BaseModel)


def is_available() -> bool:
    "API anahtari ve anthropic paketi hazir mi?"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:  # paket kurulu degilse sistem yine calismali
        return False
    return True


def _temizle(metin: str) -> str:
    "LLM'in sardigi ```json ... ``` isaretlerini soyar."
    s = metin.strip()
    if s.startswith("```"):
        satirlar = s.splitlines()
        satirlar = satirlar[1:]                      # acilis ```json satiri
        if satirlar and satirlar[-1].strip() == "```":
            satirlar = satirlar[:-1]                 # kapanis ``` satiri
        s = "\n".join(satirlar).strip()
    return s


def ask_json(system_prompt: str, user_prompt: str,
             schema_class: type[T]) -> Optional[T]:
    """LLM'e sorar, cevabi JSON parse edip schema_class ile valide eder.

    Basarili olursa pydantic nesnesi, herhangi bir sorunda None doner.
    Hicbir kosulda exception firlatmaz.
    """
    if not is_available():
        return None

    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        parcalar = [b.text for b in response.content if b.type == "text"]
        if not parcalar:
            return None

        ham = _temizle("".join(parcalar))
        veri = json.loads(ham)
        return schema_class.model_validate(veri)
    except Exception as e:
        print(f"   ! LLM cagrisi basarisiz: {type(e).__name__}: {e}")
        return None
