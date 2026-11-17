from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/"results/statistical_audit/audit_manifest.json"
REQUIRED_REPORTS=[
 "STATISTICAL_ENGINE_AUDIT.md","MATHEMATICAL_IMPLEMENTATION_REVIEW.md",
 "MONTE_CARLO_VALIDATION.md","INFERENCE_CALIBRATION.md",
 "MULTIPLE_TESTING_AUDIT.md","MODIS_SELECTION_AUDIT.md",
 "NUMERICAL_REPRODUCIBILITY.md","EXECUTION_PRIORITY_CHECKPOINT.md",
]

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()

def main():
 manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
 failures=[]
 for rel, expected in manifest.get("outputs",{}).items():
  path=ROOT/rel
  if not path.is_file(): failures.append(f"missing output: {rel}"); continue
  if sha(path)!=expected["sha256"] or path.stat().st_size!=expected["bytes"]:
   failures.append(f"output checksum/size mismatch: {rel}")
 for rel, expected in manifest.get("input_hashes",{}).items():
  path=ROOT/rel
  if not path.is_file() or sha(path)!=expected:
   failures.append(f"input hash changed or missing: {rel}")
 for name in REQUIRED_REPORTS:
  if not (ROOT/"reports/statistical_audit"/name).is_file():
   failures.append(f"required report absent: {name}")
 gold=ROOT/"data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
 if sha(gold)!="5883fe2f7af8a8d0a58dd87c6bc90340a157c71e0d3fed51fff3a850eb2368b2":
  failures.append("frozen Gold hash changed")
 trends=pd.read_csv(ROOT/"results/statistical_audit/real_trend_recalculation.csv")
 sims=pd.read_parquet(ROOT/"results/statistical_audit/simulation_metrics.parquet")
 if len(trends)!=8: failures.append(f"trend rows={len(trends)}, expected 8")
 if len(sims)<30: failures.append(f"simulation summary rows={len(sims)}, expected >=30")
 result={"status":"PASS" if not failures else "FAIL","manifest_outputs_checked":len(manifest.get("outputs",{})),
   "manifest_inputs_checked":len(manifest.get("input_hashes",{})),"required_reports_found":len(REQUIRED_REPORTS),
   "trend_rows":len(trends),"simulation_metric_rows":len(sims),"failures":failures}
 print(json.dumps(result,indent=2,allow_nan=False))
 if failures: raise SystemExit(1)

if __name__=="__main__": main()
