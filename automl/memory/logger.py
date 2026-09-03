"""Her run'i kalici olarak diske yazar."""
import json
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path("runs")


def save_run(state, sure_sn: float) -> Path:
    """Run'i runs/<timestamp>/ altina JSON + okunur ozet olarak yazar."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    klasor = RUNS_DIR / ts
    klasor.mkdir(parents=True, exist_ok=True)

    kayit = {
        "timestamp": ts,
        "data_path": state.data_path,
        "sure_sn": round(sure_sn, 2),
        "profile": state.profile.model_dump() if state.profile else None,
        "plan": state.plan.model_dump() if state.plan else None,
        "result": state.result.model_dump() if state.result else None,
        # self-improvement dongusu: hangi strateji hangi skoru verdi
        "iterasyonlar": list(getattr(state, "gecmis_denemeler", [])),
        "en_iyi_iterasyon": getattr(state, "iterasyon", 0),
        "en_iyi_strateji": getattr(state, "strateji", "varsayilan"),
    }

    (klasor / "run.json").write_text(
        json.dumps(kayit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (klasor / "run.txt").write_text(_ozet(kayit), encoding="utf-8")
    return klasor


def _ozet(k: dict) -> str:
    """Insan okuyabilsin diye kisa metin ozeti."""
    p, pl, r = k["profile"], k["plan"], k["result"]
    satirlar = [
        f"Zaman   : {k['timestamp']}",
        f"Veri    : {k['data_path']}",
        f"Sure    : {k['sure_sn']} sn",
    ]
    if p:
        satirlar += [
            f"Boyut   : {p['n_rows']} satir x {p['n_cols']} kolon",
            f"Hedef   : {p['target']}  ({p['task_type']})",
        ]
    if pl:
        satirlar += [
            f"Sayisal : {len(pl['numeric_cols'])} kolon",
            f"Kategorik: {len(pl['categorical_cols'])} kolon",
            f"Atilan  : {len(pl['drop_cols'])} kolon",
            f"PCA     : {pl['use_pca']} ({pl['n_components']})",
        ]
    if r:
        satirlar += [
            f"Model   : {r['model_name']}",
            f"Skor    : {r['metric_name']} = {r['metric_value']:.4f}",
        ]

    itlar = k.get("iterasyonlar") or []
    if itlar:
        satirlar.append(
            f"Iterasyon: {len(itlar)} deneme, en iyisi "
            f"{k.get('en_iyi_iterasyon')} ({k.get('en_iyi_strateji')})"
        )
        for d in itlar:
            if "hata" in d:
                satirlar.append(
                    f"  {d['iterasyon']}. {d['strateji']:18} HATA: {d['hata']}"
                )
                continue
            satirlar.append(
                f"  {d['iterasyon']}. {d['strateji']:18} "
                f"{d['metric_name']}={d['metric_value']:.4f} "
                f"model={d['model_name']} PCA={d['use_pca']} "
                f"imp={d['numeric_imputation']}"
            )
    return "\n".join(satirlar) + "\n"