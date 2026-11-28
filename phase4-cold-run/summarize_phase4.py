"""Build the ablation tables from archived profiles, scores and phase 3 references."""

import csv
from datetime import datetime
import json
from math import isclose
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
SCENES = ("scene01", "scene02")
STAGES = ("Preprocessing", "IDT (Docker)", "Fisher Vector", "TW-FINCH")
METRICS = ("raw_ari", "oracle_ari", "oracle_accuracy", "oracle_correct_seconds")
GIB = 2**30


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(name, rows):
    with (ROOT / name).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def load_baseline():
    archive = read_csv(ROOT / "baseline_timings.csv")
    return {
        name: {
            "scenes": {
                scene: {"wall_s": {row["stage"]: float(row[field]) for row in archive if row["scene"] == scene}}
                for scene in SCENES
            },
            "fv_peak_gib": peak,
            "docker_peak_gib": None,
            "vram_peak_gib": None,
        }
        for name, field, peak in (("M0", "org_cold_wall_s", None), ("M1", "optz_cold_wall_s", 2.80))
    }


def load_run(name, backend, reference, finch_counts):
    scores = read_json(ROOT / name / "summary.json")
    assert [scores[scope]["seconds"] for scope in (*SCENES, "combined")] == [393, 221, 614]
    scenes, score_checks = {}, {}
    for scene in SCENES:
        archive = ROOT / name / scene
        rows = read_csv(archive / "profile.csv")
        profile = read_json(archive / "profile.json")
        assert tuple(row["stage"] for row in rows) == STAGES
        assert all(row["status"] == "ok" and row["counts_toward_pipeline"] == "True" for row in rows)
        assert profile["status"] == "ok" and profile["idt_workers"] == 6 and profile["seed"] == 0
        assert profile["backend"] == backend and profile["requested_clusters"] == 8
        wall = {row["stage"]: float(row["wall_s"]) for row in rows}
        wall["Four-stage total"] = sum(wall.values())
        assert isclose(wall["Four-stage total"], profile["measured_compute_s"], abs_tol=1e-8)
        docker = profile.get("docker_peak_ram_bytes")
        fv = int(rows[2]["peak_ram_bytes"])
        vram = profile.get("vram_peak_mib")
        if docker is not None:
            assert int(rows[1]["peak_ram_bytes"]) == docker
        artifacts = {row["stage"]: int(row["artifact_bytes"]) for row in rows}
        hashes = profile["hashes"]
        expected_fv = reference["fv_sha256"][backend][scene]
        expected_trajectory = reference["trajectory_sha256"][backend][scene]
        hierarchy = finch_counts[backend, scene]
        expected_hierarchy = reference["finch_hierarchy_counts"][backend][scene]
        resources_ok = (docker is not None and docker / GIB < 6.5 and fv / GIB < 2.5 and
                        (backend == "cpu" or (vram is not None and vram / 1024 < 3.5 and
                                              profile.get("gpu_samples", 0) > 0)))
        scenes[scene] = {
            "wall_s": wall,
            "power_before": profile["power_before"],
            "power_after": profile["power_after"],
            "power_ok": profile["four_stage_conditions_ok"],
            "docker_peak_gib": docker / GIB if docker is not None else None,
            "fv_peak_gib": fv / GIB,
            "vram_peak_gib": vram / 1024 if vram is not None else None,
            "gpu_peak_celsius": profile.get("gpu_peak_celsius"),
            "gpu_samples": profile.get("gpu_samples"),
            "gpu_failed_samples": len(profile.get("gpu_errors", [])),
            "resources_ok": resources_ok,
            "intermediate_bytes": artifacts,
            "hashes": hashes,
            "expected_trajectory_sha256": expected_trajectory,
            "trajectory_matches": hashes["trajectory"] == expected_trajectory,
            "expected_fv_sha256": expected_fv,
            "fv_matches": hashes["fv_text"] == expected_fv,
            "finch_hierarchy_counts": hierarchy,
            "finch_hierarchy_matches": hierarchy == expected_hierarchy,
            "image_id": profile["image"],
            "image_tag": profile["image_tag"],
        }
    for scope in (*SCENES, "combined"):
        score_checks[scope] = {}
        for metric, old in reference[backend][scope].items():
            new = scores[scope][metric]
            score_checks[scope][metric] = {
                "phase3_seed0_matches": isclose(new, old["seed0"], rel_tol=0, abs_tol=1e-12)
                if "seed0" in old else None,
                "standard_deviations_from_mean": (new - old["mean"]) / old["sd"],
            }
    return {
        "scenes": scenes,
        **{metric: max((scene[metric] for scene in scenes.values() if scene[metric] is not None), default=None)
           for metric in ("docker_peak_gib", "fv_peak_gib", "vram_peak_gib")},
        "scores": scores,
        "score_checks": score_checks,
        "resources_ok": all(row["resources_ok"] for row in scenes.values()),
        "power_ok": all(row["power_ok"] for row in scenes.values()),
        "quality_reproduced": (
            all(row["trajectory_matches"] and row["fv_matches"] and row["finch_hierarchy_matches"]
                for row in scenes.values()) and
            all(metric["phase3_seed0_matches"] is not False
                for scope in score_checks.values() for metric in scope.values())
        ),
    }


def quality_rows(runs, reference):
    rows = []
    for metric in METRICS:
        row = {"metric": metric}
        for run, backend in (("E1", "cpu"), ("E2", "cuda")):
            current = runs[run]["scores"]["combined"][metric]
            source = reference[backend]["combined"]
            if metric == "oracle_correct_seconds":
                old = {"mean": source["oracle_accuracy"]["mean"] * 614,
                       "sd": source["oracle_accuracy"]["sd"] * 614}
            else:
                old = source[metric]
            row.update({f"{run}_seed0": current, f"{run}_phase3_mean": old["mean"], f"{run}_phase3_sd": old["sd"]})
        rows.append(row)
    return rows


def main():
    reference = read_json(ROOT / "phase3_reference.json")
    log = (ROOT / "run.log").read_text(encoding="utf-8-sig")
    chunks = re.split(r"^START (cuda|cpu) (scene0[12]) W=6 seed=0[^\n]*$", log, flags=re.MULTILINE)[1:]
    finch_counts = {
        (backend, scene): [int(count) for count in re.findall(r"^Partition \d+: (\d+) clusters$", body, re.MULTILINE)]
        for backend, scene, body in zip(chunks[0::3], chunks[1::3], chunks[2::3], strict=True)
    }
    runs = load_baseline()
    runs["E1"] = load_run("E1", "cpu", reference, finch_counts)
    runs["E2"] = load_run("E2", "cuda", reference, finch_counts)
    perf_rows = []
    baseline_s = sum(scene["wall_s"]["Four-stage total"] for scene in runs["M0"]["scenes"].values())
    for name, run in runs.items():
        total = sum(scene["wall_s"]["Four-stage total"] for scene in run["scenes"].values())
        run["total_s"] = total
        run["speedup_vs_m0"] = baseline_s / total
        row = {"configuration": name}
        for scene, data in run["scenes"].items():
            for key, stage in zip(("total_s", "preprocessing_s", "idt_s", "fv_s", "tw_finch_s"),
                                  ("Four-stage total", *STAGES), strict=True):
                row[f"{scene}_{key}"] = data["wall_s"][stage]
        row.update({key: run[key] for key in ("total_s", "speedup_vs_m0", "docker_peak_gib",
                                              "fv_peak_gib", "vram_peak_gib")})
        perf_rows.append(row)
    quality = quality_rows(runs, reference)
    starts, ends = (re.findall(f"COOLDOWN {state} (.+)", log) for state in ("START", "END"))
    cooldowns = [(datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
                 for start, end in zip(starts, ends, strict=True)]
    removed = {name: int(count) for name, count in re.findall(r"(E[12]) CLEANED (\d+) bytes", log)}
    assert set(removed) == {"E1", "E2"}
    for name, count in removed.items():
        assert count == sum(scene["intermediate_bytes"][stage]
                            for scene in runs[name]["scenes"].values() for stage in STAGES[:3])
    acceptance = {
        "four_stages_complete": True,
        "power_conditions": all(runs[name]["power_ok"] for name in ("E1", "E2")),
        "minimum_180_s_cooldown": bool(cooldowns) and all(wait >= 180 for wait in cooldowns),
        "resource_limits": all(runs[name]["resources_ok"] for name in ("E1", "E2")),
        "cuda_speedup_at_least_3x": runs["E2"]["speedup_vs_m0"] >= 3,
        "phase3_seed0_reproduced": all(runs[name]["quality_reproduced"] for name in ("E1", "E2")),
    }
    summary = {
        "status": "PASS" if all(acceptance.values()) else "CHƯA PASS",
        "acceptance": acceptance,
        "runs": runs,
        "combined_quality": quality,
        "cooldown_s": cooldowns,
        "deleted_intermediate_bytes": removed,
        "cuda_speedup_vs_e1": runs["E1"]["total_s"] / runs["E2"]["total_s"],
        "cpu_optimization_speedup_vs_m1": runs["M1"]["total_s"] / runs["E1"]["total_s"],
    }
    write_csv("ablation_performance.csv", perf_rows)
    write_csv("ablation_quality.csv", quality)
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary["status"])


if __name__ == "__main__":
    main()
