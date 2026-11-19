"""Parse PubChem SMILES (cached JSON) and compute RDKit features for ECOTOX chemicals.

Outputs data_derived/chem_features.parquet with:
 - descriptors of the largest organic fragment (or whole molecule), element counts of the full substance,
 - metal-element indicators, MACCS keys, Morgan (r=2, 512-bit) fingerprints,
 - grouping keys: 'parent_group' (metal element for metal-containing substances, else InChIKey-14 of the largest
   fragment) and 'scaffold' (Murcko scaffold of the largest fragment) for leakage-safe chemical splits.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, MACCSkeys, AllChem, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

P = Path(__file__).resolve().parents[1]
METALS = ["Cu", "Zn", "Cd", "Hg", "Pb", "Ni", "Cr", "Ag", "As", "Se", "Al", "Co", "Mn", "Fe", "Sn", "Tl", "U", "V",
          "Mo", "Sb", "Ba", "Be", "Li", "B", "Au", "Pt", "Pd", "La", "Gd", "Ce", "Y", "Sr", "Ti", "W", "Bi", "In", "Ga"]
ELEMS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "Na", "K", "Ca", "Mg"] + METALS
DESC = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount",
        "NumAromaticRings", "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms", "BertzCT", "LabuteASA",
        "NumAliphaticRings", "NumSaturatedRings", "MaxPartialCharge", "MinPartialCharge", "NumValenceElectrons"]

# mode-of-action style chemical classes (first match wins); SMARTS defined on the largest fragment
CLASS_SMARTS = [
    ("organotin", None),
    ("metal/inorganic", None),
    ("organophosphate", "P(=[O,S])([O,S,N])([O,S,N])"),
    ("pyrethroid", "[$(C1(C)(C)CC1C(=O)O),$(N#CC(OC=O)c1cccc(Oc2ccccc2)c1),$(O=C(O[CH2,CH]c1cccc(Oc2ccccc2)c1))]"),
    ("neonicotinoid", "[$(Clc1ccc(CN)cn1),$(Clc1ncc(CN)s1)]"),
    ("carbamate", "[NX3][CX3](=O)O[#6]"),
    ("triazine", "c1ncncn1"),
    ("organochlorine", None),
    ("PAH", None),
    ("phenol", "c[OX2H]"),
    ("aniline", "c[NX3;H2,H1;!$(NC=O)]"),
    ("surfactant", "[$([SX4](=O)(=O)[O-,OH]),$([SX4](=O)(=O)O[CH2]),$([NX4+](C)(C)(C)C)]"),
]
PATTS = {k: Chem.MolFromSmarts(s) for k, s in CLASS_SMARTS if s}


def chem_class(big, counts, metals_present):
    if counts.get("Sn", 0) and counts.get("C", 0):
        return "organotin"
    if metals_present or not counts.get("C", 0):
        return "metal/inorganic"
    for k, s in CLASS_SMARTS:
        if k in PATTS and k not in ("phenol", "aniline", "surfactant") and big.HasSubstructMatch(PATTS[k]):
            return k
    ncl = sum(a.GetSymbol() == "Cl" for a in big.GetAtoms())
    if ncl >= 4 and not any(a.GetSymbol() == "P" for a in big.GetAtoms()):
        return "organochlorine"
    ri = big.GetRingInfo()
    arom_c = [r for r in ri.AtomRings() if all(big.GetAtomWithIdx(i).GetIsAromatic() and
                                                big.GetAtomWithIdx(i).GetSymbol() == "C" for i in r)]
    if len(arom_c) >= 3 and all(a.GetSymbol() in ("C", "H") for a in big.GetAtoms()):
        return "PAH"
    for k in ("surfactant", "phenol", "aniline"):
        if big.HasSubstructMatch(PATTS[k]):
            if k == "surfactant" and rdMolDescriptors.CalcNumHeavyAtoms(big) < 14:
                continue
            return k
    return "other organic"


recs = [json.loads(l) for l in (P / "data_derived/pubchem_raw/pubchem_props.jsonl").read_text(encoding="utf-8").splitlines()]
rows = []
for r in recs:
    if r["status"] != "ok" or not r["props"]:
        continue
    p = r["props"][0]
    smi = p.get("SMILES") or p.get("ConnectivitySMILES")
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        continue
    d = dict(test_cas=r["cas"], cid=p.get("CID"), smiles=smi, inchikey=p.get("InChIKey"), xlogp_pubchem=p.get("XLogP"),
             pubchem_mw=float(p.get("MolecularWeight", np.nan)), lookup=r["source"])
    frags = Chem.GetMolFrags(m, asMols=True)
    d["n_fragments"] = len(frags)
    counts = {}
    for a in m.GetAtoms():
        counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    for e in ELEMS:
        d[f"el_{e}"] = counts.get(e, 0)
    d["formal_charge_total"] = sum(a.GetFormalCharge() for a in m.GetAtoms())
    metals_present = [e for e in METALS if counts.get(e, 0) > 0 and e not in ("B",)]
    d["has_metal"] = int(len(metals_present) > 0)
    d["organic"] = int(counts.get("C", 0) > 0)
    big = max(frags, key=lambda f: (f.GetNumHeavyAtoms(), sum(at.GetSymbol() == "C" for at in f.GetAtoms())))
    for nm in DESC:
        try:
            d[nm] = float(getattr(Descriptors, nm)(big))
        except Exception:
            d[nm] = np.nan
    d["frag_logp"] = Crippen.MolLogP(big)
    d["chem_class"] = chem_class(big, counts, metals_present)
    ik = Chem.MolToInchiKey(big) if big.GetNumAtoms() else ""
    d["parent_group"] = ("METAL_" + metals_present[0]) if metals_present else (ik[:14] if ik else smi)
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=big) if big.GetRingInfo().NumRings() else ""
    except Exception:
        sc = ""
    d["scaffold"] = d["parent_group"] if metals_present else (sc if sc else "ACYCLIC_" + str(min(big.GetNumHeavyAtoms() // 4, 6)))
    mac = MACCSkeys.GenMACCSKeys(big)
    for k in range(1, 167):
        d[f"maccs_{k}"] = int(mac.GetBit(k))
    fp = AllChem.GetMorganFingerprintAsBitVect(big, 2, nBits=512)
    on = set(fp.GetOnBits())
    for k in range(512):
        d[f"morgan_{k}"] = int(k in on)
    rows.append(d)
F = pd.DataFrame(rows)
F.to_parquet(P / "data_derived/chem_features.parquet", index=False)
print("featurized:", len(F), "of", len(recs), "| metal-containing:", F.has_metal.sum(), "| parent groups:", F.parent_group.nunique(),
      "| scaffolds:", F.scaffold.nunique())
