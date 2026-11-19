"""Extract reported LC50s and survival-based bounds from two public experiments."""
from hashlib import sha256
from io import BytesIO
import json
import re

import numpy as np
import pandas as pd
import requests
from docx import Document

from common import D, R, load_lc50
from E16_long_duration_validation import PAIR, REF, reconstruct_sources, normalized, point_endpoint

FPF = "https://ars.els-cdn.com/content/image/1-s2.0-S014765132200817X-mmc1.docx"
CHL = "https://virgile-baudrot.r-universe.dev/morse/data/chlordan/csv"


def download(url):
    answer = requests.get(url, timeout=60)
    answer.raise_for_status()
    return answer.content


def reported(rows, species, day):
    col = rows[0].index(str(day))
    idx = next(i for i, row in enumerate(rows) if row[:2] == [species, "LC50"])
    point = float(rows[idx][col])
    bounds = [float(v) for v in re.findall(r"[\d.]+", rows[idx+1][col])]
    return point, (bounds if len(bounds) == 2 else [None, None])


def main():
    paper, data = download(FPF), download(CHL)
    tables = [[[c.text.strip() for c in row.cells] for row in t.rows]
              for t in Document(BytesIO(paper)).tables]
    rows = []
    for short, species in [("C. dipterum","Cloeon dipterum"),("G. pulex","Gammarus pulex")]:
        acute, aci = reported(tables[6], short, 4)
        for day in (21, 28):
            long, lci = reported(tables[7], short, day)
            rows.append(dict(chemical="Flupyradifurone", species=species, day=day, acute=acute,
                             acute_low=aci[0], acute_high=aci[1], long=long, long_low=lci[0],
                             long_high=lci[1], factor=acute/long, kind="Point estimate",
                             uncertainty="Published 95% confidence limits", source="10.1016/j.ecoenv.2022.113977",
                             location="Supplementary Tables S7 and S8"))
    raw = pd.read_csv(BytesIO(data))
    surv = raw[raw.time.isin([0,4,21])].groupby(["conc","time"]).Nsurv.sum().unstack()
    surv.columns = ["initial","day4","day21"]
    surv.to_csv(R / "E16_chlordane_counts.csv")
    assert (surv.day4 / surv.initial > 0.5).all() and (surv.initial == 10).all()
    assert surv.loc[0,"day21"] == surv.loc[0,"initial"]
    low = float(surv.index[(surv.day21 / surv.initial) > 0.5].max())
    high = float(surv.index[(surv.day21 / surv.initial) < 0.5].min())
    lower4 = float(surv.index.max())
    assert low < high
    rows.append(dict(chemical="Chlordane",species="Daphnia magna",day=21,acute_low=lower4,
                     long_low=low,long_high=high,factor=lower4/high,kind="Empirical lower bound",
                     uncertainty="Tested doses bracketing 50% mortality",source="10.1897/08-564.1",
                     location="morse: chlordan, days 4 and 21"))
    records = load_lc50()
    refs = pd.read_parquet(D / "references.parquet")
    anchors, _, blocked, _ = reconstruct_sources(records, refs)
    selected = refs[refs.doi.fillna("").str.lower().isin(
        ["10.1016/j.ecoenv.2022.113977","10.1897/08-564.1","10.1002/tox.20616","10.1016/j.aquatox.2025.107487"])
        | refs.title.map(normalized).isin([normalized(t) for t in [
            "Comparing the acute and chronic toxicity of flupyradifurone and imidacloprid to non-target aquatic arthropod species",
            "Reproductive effects and bioaccumulation of chlordane in Daphnia magna",
            "Effects of legacy and emerging antimicrobial compounds to early-life stages of rainbow trout (Oncorhynchus mykiss)"]])]
    assert not set(selected[REF]) & blocked
    fpf = records[records.test_cas.eq("951659408")]
    assert not fpf.species_number.isin(["157","372"]).any()
    frozen = pd.read_parquet(R/"E07_standardized_records.parquet").reset_index(drop=True)
    for key in [*PAIR, "log_dur", "log10_mgL"]:
        assert np.array_equal(anchors[key], frozen[key])
    anchors["predicted_log"] = anchors.log10_mgL + frozen.y28d - frozen.y96
    # Amekor et al. (2025), measured 28-day endpoints in newly hatched trout.
    # EPA MRIDs 43969301 (triclosan), 00144378/41536903 (chloroxylenol).
    # The additional triclosan result 2113795 was a 3.89% formulation.
    for cas, chemical, ids, lc28, limits in [
        ("3380345","Triclosan",["2112411"],107,[91,128]),
        ("88040","Chloroxylenol",["2107389"],254,[203,301]),
        ("88040","Chloroxylenol",["2107389","2109708"],254,[203,301])]:
        a = anchors[anchors.result_id.astype(str).isin(ids)].copy()
        assert len(a)==len(ids) and point_endpoint(a).all()
        assert a.test_cas.eq(cas).all() and a.species_number.eq("4").all()
        assert a.dur_h.eq(96).all() and a.conc1_type.eq("A").all()
        acute = 1000*10**a.log10_mgL.mean()
        projected = 1000*10**a.predicted_log.mean()
        rows.append(dict(chemical=chemical,species="Oncorhynchus mykiss",day=28,acute=acute,
                         long=lc28,long_low=limits[0],long_high=limits[1],factor=acute/lc28,
                         predicted_long=projected,kind=("Qualifier sensitivity only" if len(ids)>1
                                                        else "Cross-study point estimate"),
                         uncertainty="Published 95% confidence limits, long test",
                         source="10.1016/j.aquatox.2025.107487",
                         location="28-day survival; ECOTOX acute result IDs "+" / ".join(ids)))
    pd.DataFrame(rows).to_csv(R / "E16_external_experiments.csv",index=False)
    audit = dict(url_sha256={FPF:sha256(paper).hexdigest(), CHL:sha256(data).hexdigest()},
                 related_archive_sources=selected[[REF,"title","doi"]].to_dict("records"),
                 related_sources_used_in_projection=sorted(set(selected[REF]) & blocked),
                 archived_fpf_sources=fpf[[REF,*PAIR,"latin_name"]].drop_duplicates().to_dict("records"))
    (R/"E16_external_provenance.json").write_text(json.dumps(audit,indent=2),encoding="utf8")


if __name__ == "__main__":
    main()
