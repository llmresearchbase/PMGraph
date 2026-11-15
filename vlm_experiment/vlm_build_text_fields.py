#!/usr/bin/env python3
"""
Builds the VLM's TEXT input, deliberately restricted to the SAME field scope
the MLP/GNN baselines use (Name, NAICS_Description_1, Business_Status,
Industry, Certification, Process_Capability, City, State, Employees) --
explicitly excluding Material_Capability (the target) AND Business_Description
/Product_i strings. The latter two ARE text fields in the directory, but
round-4's own analysis found they sometimes mention materials not in the
structured record (the 17.4%/24.6% inconsistency rates) -- giving the VLM
that free text would make its text condition strictly easier than what
MLP/GNN ever saw, for reasons having nothing to do with vision. This is a
deliberate scope-matching decision, stated explicitly per instruction
("若文本信息与MLP/GNN不同，明确披露").
"""
import json
import importlib.util

import pandas as pd

spec = importlib.util.spec_from_file_location("ga", "Graph-A-Builder.py")
ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga)


def main():
    df = pd.read_csv("Cleaned_Master_CSV_v3.csv", low_memory=False)
    id2pos = {ga.canonical_manufacturer_id(v): idx for idx, v in enumerate(df["Web"].fillna("").astype(str))}

    with open("material_task_split.json", encoding="utf-8") as f:
        split = json.load(f)
    test_manus = split["test_manufacturers"]

    import torch
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    manus_sorted = cov["manus_sorted"]
    has_image = cov["has_image"]
    idx_of = {m: i for i, m in enumerate(manus_sorted)}
    test_has_image = sorted(m for m in test_manus if m in idx_of and has_image[idx_of[m]])

    with open("vlm_experiment/image_manifest_val_dryrun.json", encoding="utf-8") as f:
        val_dryrun = json.load(f)
    val_dryrun_manus = val_dryrun["sampled_manufacturers"]
    test_has_image = sorted(set(test_has_image) | set(val_dryrun_manus))

    status_cols = [c for c in df.columns if c.startswith("Business_Status_")]
    industry_cols = [c for c in df.columns if c.startswith("Industry_")]
    cert_cols = [c for c in df.columns if c.startswith("Certification_")]
    proc_cols = [c for c in df.columns if c.startswith("Process_Capability_")]

    def listify(row, cols):
        vals = [str(row[c]).strip() for c in cols if pd.notna(row.get(c)) and str(row[c]).strip()]
        seen, out = set(), []
        for v in vals:
            if v.lower() not in seen:
                seen.add(v.lower())
                out.append(v)
        return out

    records = {}
    for manu in test_has_image:
        pos = id2pos.get(manu)
        if pos is None:
            records[manu] = {"error": "not found in CSV"}
            continue
        row = df.iloc[pos]
        name = str(row.get("Name", "")).strip()
        naics_desc = str(row.get("NAICS_Description_1", "")).strip() if pd.notna(row.get("NAICS_Description_1")) else ""
        status = listify(row, status_cols)
        industry = listify(row, industry_cols)
        cert = listify(row, cert_cols)
        proc = listify(row, proc_cols)
        city = str(row.get("City_1", "")).strip() if pd.notna(row.get("City_1")) else ""
        state = str(row.get("State_1", "")).strip() if pd.notna(row.get("State_1")) else ""
        employees = row.get("Employees", None)
        employees = int(employees) if pd.notna(employees) else None

        lines = [f"Company name: {name}" if name else "Company name: (unknown)"]
        if naics_desc:
            lines.append(f"NAICS description: {naics_desc}")
        if industry:
            lines.append(f"Industry: {', '.join(industry)}")
        if proc:
            lines.append(f"Process capabilities: {', '.join(proc)}")
        if cert:
            lines.append(f"Certifications: {', '.join(cert)}")
        if status:
            lines.append(f"Business status: {', '.join(status)}")
        if city or state:
            lines.append(f"Location: {city}{', ' if city and state else ''}{state}")
        if employees is not None:
            lines.append(f"Employees: {employees}")
        records[manu] = {"text": "\n".join(lines), "fields_used": [
            "Name", "NAICS_Description_1", "Industry_*", "Process_Capability_*",
            "Certification_*", "Business_Status_*", "City_1", "State_1", "Employees",
        ]}

    with open("vlm_experiment/text_fields.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1, ensure_ascii=False)

    n_err = sum(1 for v in records.values() if "error" in v)
    print(f"built text for {len(records)} manufacturers, {n_err} lookup errors")
    sample_k = next(k for k, v in records.items() if "text" in v)
    print(f"\nsample ({sample_k}):\n{records[sample_k]['text']}")


if __name__ == "__main__":
    main()
