"""Fetch structures for ECOTOX chemicals from PubChem PUG REST (public data).

Lookup by CAS (formatted with dashes) as a PubChem name; fallback to DTXSID. Responses are cached as raw JSON lines
in data_derived/pubchem_raw/ and parsed as data only. Rate limited to <5 requests/s per PubChem policy.
"""
import json, time, threading, urllib.request, urllib.parse, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

P = Path(__file__).resolve().parents[1]
OUT = P / "data_derived" / "pubchem_raw" / "pubchem_props.jsonl"
PROPS = "SMILES,ConnectivitySMILES,MolecularWeight,XLogP,InChIKey,IUPACName"
URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/" + PROPS + "/JSON"

def fmt_cas(c):
    c = str(c).strip()
    return f"{c[:-3]}-{c[-3:-1]}-{c[-1]}" if c.isdigit() and len(c) >= 5 else c

lock = threading.Lock(); last = [0.0]
def throttle():
    with lock:
        wait = 0.23 - (time.time() - last[0])
        if wait > 0: time.sleep(wait)
        last[0] = time.time()

def get(name):
    for attempt in range(4):
        throttle()
        try:
            with urllib.request.urlopen(URL.format(urllib.parse.quote(name)), timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return "ERROR"

def work(row):
    cas, dtx = row
    res = get(fmt_cas(cas)); src = "cas"
    if (res is None or res == "ERROR") and isinstance(dtx, str) and dtx.startswith("DTXSID"):
        r2 = get(dtx)
        if r2 not in (None, "ERROR"): res, src = r2, "dtxsid"
    return dict(cas=cas, dtxsid=dtx, source=src, status="error" if res == "ERROR" else ("notfound" if res is None else "ok"),
                props=(res or {}).get("PropertyTable", {}).get("Properties", []) if isinstance(res, dict) else [])

if __name__ == "__main__":
    x = pd.read_parquet(P / "data_derived" / "aquatic_acute_all.parquet", columns=["test_cas"])
    chem = pd.read_parquet(P / "data_derived" / "chemicals.parquet")
    need = chem[chem.cas_number.isin(x.test_cas.unique())][["cas_number", "dtxsid"]]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["cas"] for l in OUT.read_text(encoding="utf-8").splitlines() if '"status": "error"' not in l}
    todo = [tuple(r) for r in need.values if r[0] not in done]
    print("chemicals:", len(need), "to fetch:", len(todo), flush=True)
    with ThreadPoolExecutor(4) as ex, open(OUT, "a", encoding="utf-8") as f:
        for k, rec in enumerate(ex.map(work, todo)):
            f.write(json.dumps(rec) + "\n")
            if k % 250 == 0: print(k, flush=True); f.flush()
    print("done", flush=True)
