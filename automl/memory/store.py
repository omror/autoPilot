"""Gecmis run'lari okur ve benzerlerini bulur."""
import json
from pathlib import Path

RUNS_DIR = Path("runs")


def tum_runlar() -> list[dict]:
    """Diskteki tum run'lari yeniden eskiye siralar."""
    if not RUNS_DIR.exists():
        return []

    kayitlar = []
    for f in sorted(RUNS_DIR.glob("*/run.json"), reverse=True):
        try:
            kayitlar.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue  # bozuk dosyayi atla
    return kayitlar


def _benzerlik(p1: dict, p2: dict) -> float:
    """Iki veri profili ne kadar benziyor? 0.0 - 1.0 arasi."""
    if p1["task_type"] != p2["task_type"]:
        return 0.0  # farkli task, karsilastirilamaz

    puan = 0.4  # ayni task tipi

    # Veri boyutu ayni mertebede mi (2 kat icinde)
    r1, r2 = p1["n_rows"], p2["n_rows"]
    if max(r1, r2) / max(min(r1, r2), 1) <= 2:
        puan += 0.3

    # Kolon tipi dagilimi yakin mi
    def oran(p):
        num = sum(1 for c in p["columns"] if c["inferred_type"] == "numeric")
        return num / max(len(p["columns"]), 1)

    if abs(oran(p1) - oran(p2)) < 0.25:
        puan += 0.3

    return puan


def benzer_runlar(profile, esik: float = 0.6, limit: int = 5) -> list[dict]:
    """Verilen profile benzeyen gecmis run'lari dondurur."""
    hedef = profile.model_dump()
    sonuc = []

    for r in tum_runlar():
        if not r.get("profile") or not r.get("result"):
            continue
        skor = _benzerlik(hedef, r["profile"])
        if skor >= esik:
            sonuc.append({"benzerlik": skor, "run": r})

    sonuc.sort(key=lambda x: x["benzerlik"], reverse=True)
    return sonuc[:limit]