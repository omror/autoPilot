"""Gecmis run'lari listeler."""
from pathlib import Path
from automl.memory.store import tum_runlar

runlar = tum_runlar()
print(f"Toplam {len(runlar)} run\n")

for r in runlar:
    p, res = r.get("profile"), r.get("result")
    if not p or not res:
        continue
    print(
        f"{r['timestamp']}  {Path(r['data_path']).name:18} "
        f"{p['task_type']:15} {res['model_name']:20} "
        f"{res['metric_name']}={res['metric_value']:.4f}"
    )