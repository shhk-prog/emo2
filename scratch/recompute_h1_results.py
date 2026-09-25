import json
from pathlib import Path
import numpy as np

repo_root = Path("/mnt/nas/home/hiromi/src/emo2")
derived_dir = repo_root / "v3" / "results" / "derived"
raw_dir = repo_root / "v3" / "results" / "raw"

# 1. Load frozen directions
frozen_path = derived_dir / "frozen_confirmatory_sites.json"
with open(frozen_path, "r", encoding="utf-8") as f:
    frozen_sites = json.load(f)

h1_rep_dir = frozen_sites.get("h1_replication_direction", {
    "valence": {"peak": -1, "center_of_mass": -1},
    "arousal": {"peak": -1, "center_of_mass": 1},
})

sign_pk_v = int(h1_rep_dir["valence"]["peak"])
sign_ct_v = int(h1_rep_dir["valence"]["center_of_mass"])
sign_pk_a = int(h1_rep_dir["arousal"]["peak"])
sign_ct_a = int(h1_rep_dir["arousal"]["center_of_mass"])

print(f"Loaded frozen directions: V(pk={sign_pk_v}, ct={sign_ct_v}), A(pk={sign_pk_a}, ct={sign_ct_a})")

def update_dissoc_dict(d, sign_pk, sign_ct):
    raw_pk = float(d.get("delta_peak_raw", d.get("delta_d_peak", d.get("delta_d_star", 0.0))))
    raw_ct = float(d.get("delta_com_raw", d.get("delta_d_center", d.get("delta_bar_d", 0.0))))
    pk_ci = d["delta_d_peak_ci"]
    ct_ci = d["delta_d_center_ci"]

    al_pk = float(sign_pk * raw_pk)
    al_ct = float(sign_ct * raw_ct)

    al_pk_ci = [float(-pk_ci[1]), float(-pk_ci[0])] if sign_pk == -1 else [float(pk_ci[0]), float(pk_ci[1])]
    al_ct_ci = [float(-ct_ci[1]), float(-ct_ci[0])] if sign_ct == -1 else [float(ct_ci[0]), float(ct_ci[1])]

    pk_pass = bool(al_pk_ci[0] > 0)
    ct_pass = bool(al_ct_ci[0] > 0)
    h1_pass = bool(pk_pass and ct_pass)

    d["qwen_sign_peak"] = sign_pk
    d["qwen_sign_center"] = sign_ct
    d["qwen_sign_com"] = sign_ct
    d["delta_peak_raw"] = raw_pk
    d["delta_com_raw"] = raw_ct
    d["delta_peak_aligned"] = al_pk
    d["delta_com_aligned"] = al_ct
    d["delta_d_peak_aligned"] = al_pk
    d["delta_d_center_aligned"] = al_ct
    d["aligned_peak_ci"] = al_pk_ci
    d["aligned_com_ci"] = al_ct_ci
    d["aligned_center_ci"] = al_ct_ci
    d["delta_d_peak_aligned_ci"] = al_pk_ci
    d["delta_d_center_aligned_ci"] = al_ct_ci
    d["peak_replication_pass"] = pk_pass
    d["com_replication_pass"] = ct_pass
    d["center_replication_pass"] = ct_pass
    d["h1_replication_pass"] = h1_pass
    d["passed"] = h1_pass
    return d

family_results = {}
for fam_key in ["llama", "gemma", "olmo"]:
    fpaths = [
        raw_dir / f"v3_confirmatory_{fam_key}.json",
        raw_dir / f"v3_confirmatory_replication_{fam_key}.json",
    ]
    for p in fpaths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        target_d = data["results"] if "results" in data else data
        h1 = target_d["h1_dissociation"]
        update_dissoc_dict(h1["valence"], sign_pk_v, sign_ct_v)
        update_dissoc_dict(h1["arousal"], sign_pk_a, sign_ct_a)

        h1_pass = bool(h1["valence"]["h1_replication_pass"] and h1["arousal"]["h1_replication_pass"])
        h1["delta_peak_raw"] = h1["valence"]["delta_peak_raw"]
        h1["delta_com_raw"] = h1["valence"]["delta_com_raw"]
        h1["qwen_sign_peak"] = h1["valence"]["qwen_sign_peak"]
        h1["qwen_sign_com"] = h1["valence"]["qwen_sign_com"]
        h1["delta_peak_aligned"] = h1["valence"]["delta_peak_aligned"]
        h1["delta_com_aligned"] = h1["valence"]["delta_com_aligned"]
        h1["aligned_peak_ci"] = h1["valence"]["aligned_peak_ci"]
        h1["aligned_com_ci"] = h1["valence"]["aligned_com_ci"]
        h1["peak_replication_pass"] = h1["valence"]["peak_replication_pass"]
        h1["com_replication_pass"] = h1["valence"]["com_replication_pass"]
        h1["h1_replication_pass"] = h1_pass
        h1["passed"] = h1_pass

        # check overall auxiliary_qc_all_pass
        h2_pass = target_d.get("h2_sufficiency", {}).get("passed", False)
        h3_pass = target_d.get("h3_endogenous_relevance", {}).get("passed", False)
        h4_pass = target_d.get("h4_temporal_emergence", {}).get("passed", False)
        target_d["auxiliary_qc_all_pass"] = bool(h1_pass and h2_pass and h3_pass and h4_pass)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Updated {p.name}: V pass={h1['valence']['h1_replication_pass']}, A pass={h1['arousal']['h1_replication_pass']}, overall H1 pass={h1_pass}")

        family_name = target_d.get("family", fam_key)
        family_results[family_name] = target_d

# 2. Update v3_cross_model_replication_summary.json
summary_path = derived_dir / "v3_cross_model_replication_summary.json"
if summary_path.exists():
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Update family_wise_results and auxiliary_qc_checklist
    for fam, res in family_results.items():
        if fam in summary.get("family_wise_results", {}):
            summary["family_wise_results"][fam]["h1_dissociation"] = res["h1_dissociation"]
            summary["auxiliary_qc_checklist"]["H1_peak_dissociation"][fam] = res["h1_dissociation"]["passed"]

    # Recompute primary effect estimates for H1
    h1_dp_v = [res["h1_dissociation"]["valence"]["delta_d_peak"] for res in family_results.values()]
    h1_dp_a = [res["h1_dissociation"]["arousal"]["delta_d_peak"] for res in family_results.values()]
    h1_dp_v_al = [res["h1_dissociation"]["valence"]["delta_peak_aligned"] for res in family_results.values()]
    h1_dp_a_al = [res["h1_dissociation"]["arousal"]["delta_peak_aligned"] for res in family_results.values()]

    def boot_ci(vals):
        vals = np.array(vals)
        if len(vals) <= 1:
            return float(np.mean(vals)), float(np.mean(vals)), float(np.mean(vals))
        rng = np.random.default_rng(42)
        boots = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(1000)]
        return float(np.mean(vals)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    pt_h1_v, h1_v_low, h1_v_high = boot_ci(h1_dp_v)
    pt_h1_a, h1_a_low, h1_a_high = boot_ci(h1_dp_a)
    pt_h1_v_al, h1_v_al_low, h1_v_al_high = boot_ci(h1_dp_v_al)
    pt_h1_a_al, h1_a_al_low, h1_a_al_high = boot_ci(h1_dp_a_al)

    summary["primary_effect_estimates"]["H1_peak_dissociation"] = {
        "valence": {
            "mean_delta_d_peak": float(pt_h1_v),
            "ci_95": [float(h1_v_low), float(h1_v_high)],
            "mean_delta_d_peak_raw": float(pt_h1_v),
            "ci_95_raw": [float(h1_v_low), float(h1_v_high)],
            "mean_delta_d_peak_aligned": float(pt_h1_v_al),
            "ci_95_aligned": [float(h1_v_al_low), float(h1_v_al_high)],
        },
        "arousal": {
            "mean_delta_d_peak": float(pt_h1_a),
            "ci_95": [float(h1_a_low), float(h1_a_high)],
            "mean_delta_d_peak_raw": float(pt_h1_a),
            "ci_95_raw": [float(h1_a_low), float(h1_a_high)],
            "mean_delta_d_peak_aligned": float(pt_h1_a_al),
            "ci_95_aligned": [float(h1_a_al_low), float(h1_a_al_high)],
        },
    }

    all_pass = all(
        summary["auxiliary_qc_checklist"][h][fam]
        for h in summary["auxiliary_qc_checklist"]
        for fam in summary["auxiliary_qc_checklist"][h]
    )
    summary["auxiliary_qc_all_pass"] = bool(all_pass)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated {summary_path.name}")
