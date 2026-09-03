"""Profile bakarak preprocessing planini uretir."""
from automl import llm
from automl.agents.base import Agent
from automl.memory.store import benzer_runlar
from automl.schemas import DataProfile, PreprocessingPlan, RunState

SYSTEM_PROMPT = (
    "Sen bir ML preprocessing uzmanisin. Verilen veri profiline bakarak "
    "preprocessing kararlari oner."
)


def plan(state: RunState) -> RunState:
    "Profile bakarak preprocessing kararlarini otomatik uretir."
    p = state.profile
    if p is None:
        raise RuntimeError("plan: once profile adimi calismali")

    notes = []

    numeric_cols = []
    categorical_cols = []
    drop_cols = []

    for c in p.columns:
        if c.name == p.target:
            continue

        if c.null_ratio > 0.5:
            drop_cols.append(c.name)
            notes.append(f"{c.name}: %{c.null_ratio*100:.0f} bos, atildi")
            continue

        if c.inferred_type in ("text", "datetime"):
            drop_cols.append(c.name)
            notes.append(f"{c.name}: {c.inferred_type} tipi, atildi")
            continue

        if c.inferred_type == "categorical" and c.n_unique > 50:
            drop_cols.append(c.name)
            notes.append(f"{c.name}: {c.n_unique} essiz kategori, atildi")
            continue

        if c.inferred_type == "numeric":
            numeric_cols.append(c.name)
        else:
            categorical_cols.append(c.name)

    use_pca = len(numeric_cols) > 10 and p.n_rows > 50
    n_components = min(10, len(numeric_cols)) if use_pca else None
    if use_pca:
        notes.append(f"{len(numeric_cols)} sayisal kolon icin PCA acildi")

    numeric_imputation = "median"
    categorical_imputation = "most_frequent"

    # Strateji katmani: "varsayilan" disinda kurallari oynatir.
    # Onceki iterasyon eşigi gecemediyse orchestrator strateji degistirir.
    if state.strateji == "pca_ters":
        if use_pca:
            use_pca = False
            n_components = None
            notes.append("strateji: PCA kapatildi (tersine cevrildi)")
        elif len(numeric_cols) >= 2:
            use_pca = True
            n_components = min(10, len(numeric_cols))
            notes.append(
                f"strateji: PCA acildi (tersine cevrildi, "
                f"n_components={n_components})"
            )
        else:
            notes.append("strateji: PCA acilamadi, yeterli sayisal kolon yok")
    elif state.strateji == "imputation_degis":
        numeric_imputation = "mean"
        categorical_imputation = "constant"
        notes.append("strateji: imputation median->mean, "
                     "most_frequent->constant")

    state.plan = PreprocessingPlan(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        drop_cols=drop_cols,
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
        scaling="standard",
        encoding="onehot",
        use_pca=use_pca,
        n_components=n_components,
        notes=notes,
    )
    return state


def _profil_ozeti(p: DataProfile, kural_plan: PreprocessingPlan) -> str:
    "LLM'e verilecek profil + kural-tabanli plan ozeti."
    satirlar = [
        f"Satir sayisi: {p.n_rows}",
        f"Kolon sayisi: {p.n_cols}",
        f"Hedef kolon: {p.target}",
        f"Gorev tipi: {p.task_type}",
        "",
        "Kolonlar (ad | tip | essiz | null orani):",
    ]
    for c in p.columns:
        satirlar.append(
            f"  {c.name} | {c.inferred_type} | {c.n_unique} | "
            f"{c.null_ratio:.2f}"
        )

    if p.high_correlations:
        satirlar.append("")
        satirlar.append("Yuksek korelasyonlar (|r| >= 0.85):")
        for a, b, r in p.high_correlations:
            satirlar.append(f"  {a} <-> {b} r={r:+.3f}")

    satirlar += [
        "",
        "Kural-tabanli planin onerisi:",
        f"  numeric_cols: {kural_plan.numeric_cols}",
        f"  categorical_cols: {kural_plan.categorical_cols}",
        f"  drop_cols: {kural_plan.drop_cols}",
        f"  numeric_imputation: {kural_plan.numeric_imputation}",
        f"  categorical_imputation: {kural_plan.categorical_imputation}",
        f"  scaling: {kural_plan.scaling}",
        f"  encoding: {kural_plan.encoding}",
        f"  use_pca: {kural_plan.use_pca} (n_components="
        f"{kural_plan.n_components})",
        "",
        "Bu profile bakarak preprocessing planini JSON olarak dondur. "
        "Sadece JSON dondur, baska aciklama yazma. Alanlar: numeric_cols, "
        "categorical_cols, drop_cols, numeric_imputation, "
        "categorical_imputation, scaling, encoding, use_pca, n_components, "
        "notes. Kolon adlarini AYNEN yukaridaki listeden kullan, yeni kolon "
        "adi uydurma. Hedef kolonu ozellik listelerine koyma.",
    ]
    return "\n".join(satirlar)


def _gecmis_ozeti(p: DataProfile) -> tuple[str, list[str]]:
    """Benzer gecmis run'lari LLM baglami + insan okur not olarak dondurur.

    Donen: (llm_prompt_parcasi, notes_satirlari). Gecmis yoksa ikisi de bos.
    """
    try:
        benzerler = benzer_runlar(p)
    except Exception as e:
        print(f"   ! gecmis run'lar okunamadi: {type(e).__name__}: {e}")
        return "", []

    if not benzerler:
        return "", []

    prompt_satirlari = ["", "Benzer veride daha once denenen planlar:"]
    not_satirlari = []
    for kayit in benzerler:
        r = kayit["run"].get("result") or {}
        pl = kayit["run"].get("plan") or {}
        if not r or not pl:
            continue
        ozet = (
            f"benzerlik={kayit['benzerlik']:.2f} | "
            f"PCA={pl.get('use_pca')}({pl.get('n_components')}) | "
            f"imputation={pl.get('numeric_imputation')}/"
            f"{pl.get('categorical_imputation')} | "
            f"model={r.get('model_name')} | "
            f"{r.get('metric_name')}={r.get('metric_value'):.4f}"
        )
        prompt_satirlari.append(f"  {ozet}")
        not_satirlari.append(f"gecmis: {ozet}")

    if not not_satirlari:
        return "", []

    prompt_satirlari.append(
        "  Bu gecmis sonuclari dikkate al, daha iyi skor veren plana yaklas."
    )
    return "\n".join(prompt_satirlari), not_satirlari


def _kolonlar_gecerli(oneri: PreprocessingPlan, p: DataProfile) -> bool:
    "LLM'in onerdigi kolon adlari gercekten veride var mi?"
    gecerli = {c.name for c in p.columns if c.name != p.target}
    onerilen = (oneri.numeric_cols + oneri.categorical_cols
                + oneri.drop_cols)
    for ad in onerilen:
        if ad not in gecerli:
            print(f"   ! LLM uydurma kolon adi verdi: {ad!r}, plan reddedildi")
            return False
    if not oneri.numeric_cols and not oneri.categorical_cols:
        print("   ! LLM bos ozellik listesi verdi, plan reddedildi")
        return False
    return True


class PlannerAgent(Agent):
    """Preprocessing planini uretir."""

    name = "planner"

    def run(self, state: RunState) -> RunState:
        # 1) Kural-tabanli plan her zaman calisir, guvenli taban budur.
        state = plan(state)
        kural_plan = state.plan
        p = state.profile
        if kural_plan is None or p is None:
            return state

        # 2) Benzer gecmis run'lardan baglam topla.
        gecmis_prompt, gecmis_notlar = _gecmis_ozeti(p)

        # 3) LLM varsa gecmisi de vererek ikinci bir gorus al.
        oneri = None
        if llm.is_available():
            oneri = llm.ask_json(
                SYSTEM_PROMPT,
                _profil_ozeti(p, kural_plan) + gecmis_prompt,
                PreprocessingPlan,
            )
        elif gecmis_notlar:
            # LLM yok: gecmis bilgisi sadece bilgi amacli nota yazilir.
            kural_plan.notes.extend(gecmis_notlar)

        # 4) Gecerliyse LLM planini kullan, degilse kural-tabanliya dus.
        if oneri is not None and _kolonlar_gecerli(oneri, p):
            oneri.notes.append("LLM onerisi kullanildi")
            state.plan = oneri
        else:
            kural_plan.notes.append("kural-tabanli plan kullanildi")
            state.plan = kural_plan

        return state
