"""Tum veri setlerini boru hattindan gecirip karsilastirmali rapor uretir.

Bu rapor projenin domain-bagimsizlik iddiasinin kanitidir: ayni kod,
hicbir veri setine ozel ayar olmadan, farkli alanlardan gelen verilerde
task tipini kendi tespit edip calisir.
"""
import contextlib
import io
import time
from pathlib import Path

from automl.orchestrator import run

DATA_DIR = Path("data")
CIKTI = Path("REPORT.md")


def _satir(yol: Path) -> dict:
    "Tek bir veri setini calistirir, rapor satirini uretir."
    baslangic = time.time()
    buf = io.StringIO()
    try:
        # Boru hattinin kendi ciktisi rapora karismasin
        with contextlib.redirect_stdout(buf):
            state = run(str(yol))
    except Exception as e:
        return {
            "veri": yol.name,
            "hata": f"{type(e).__name__}: {e}",
        }

    p, pl, r = state.profile, state.plan, state.result
    if p is None or pl is None or r is None:
        return {"veri": yol.name, "hata": "sonuc uretilemedi"}

    return {
        "veri": yol.name,
        "boyut": f"{p.n_rows} x {p.n_cols}",
        "task": p.task_type,
        "atilan": len(pl.drop_cols),
        "pca": f"{pl.n_components}" if pl.use_pca else "-",
        "model": r.model_name,
        "metrik": r.metric_name,
        "skor": f"{r.metric_value:.4f}",
        "iterasyon": len(state.gecmis_denemeler),
        "sure": f"{time.time() - baslangic:.1f} sn",
    }


def main() -> None:
    dosyalar = sorted(DATA_DIR.glob("*.csv"))
    if not dosyalar:
        print("data/ altinda csv yok, once make_data.py calistirin")
        return

    satirlar = []
    for yol in dosyalar:
        print(f"-> {yol.name} calisiyor...")
        s = _satir(yol)
        satirlar.append(s)
        if "hata" in s:
            print(f"   ! {s['hata']}")
        else:
            print(f"   {s['metrik']}={s['skor']} ({s['model']}, "
                  f"{s['iterasyon']} iterasyon, {s['sure']})")

    md = ["# Karsilastirmali Rapor", ""]
    md.append(
        "Asagidaki tablo tek bir kod tabaninin, veri setine ozel hicbir "
        "ayar yapilmadan farkli alanlardan gelen verilerde calistigini "
        "gosterir. Task tipi, kolon tipleri ve preprocessing plani her "
        "veri icin otomatik tespit edilmistir."
    )
    md.append("")
    md.append(
        "| veri seti | satir x kolon | task | atilan kolon | PCA | "
        "model | metrik | skor | iterasyon | sure |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|")

    for s in satirlar:
        if "hata" in s:
            md.append(
                f"| {s['veri']} | - | - | - | - | - | - | "
                f"HATA: {s['hata']} | - | - |"
            )
            continue
        md.append(
            f"| {s['veri']} | {s['boyut']} | {s['task']} | {s['atilan']} | "
            f"{s['pca']} | {s['model']} | {s['metrik']} | {s['skor']} | "
            f"{s['iterasyon']} | {s['sure']} |"
        )

    md.append("")
    md.append(f"Uretim zamani: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")

    CIKTI.write_text("\n".join(md), encoding="utf-8")
    print(f"\n{CIKTI} yazildi ({len(satirlar)} veri seti)")


if __name__ == "__main__":
    main()
