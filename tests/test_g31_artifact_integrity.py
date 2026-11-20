import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "g31"


def _json(name):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def _jsonl(name):
    return [
        json.loads(line)
        for line in (ARTIFACTS / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_pilot_dataset_hash_matches_manifest_without_model_loading():
    manifest = _json("pilot_manifest.json")
    actual = hashlib.sha256((ARTIFACTS / "pilot.jsonl").read_bytes()).hexdigest()
    assert actual == manifest["dataset_sha256"]


def test_checkpoint_hashes_match_reports_without_deserialization():
    for report_name in ("pilot_b1_report.json", "staging_b1_report.json"):
        report = _json(report_name)
        checkpoint = ROOT / report["checkpoint"]
        expected = report["checkpoint_sha256"]
        assert checkpoint.is_file()
        assert len(expected) == 64
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected


def test_staging_b1_predictions_preserve_pilot_pairing_and_action_channel():
    baseline = _jsonl("pilot_predictions.jsonl")
    staging = _jsonl("staging_b1_predictions.jsonl")
    report = _json("staging_b1_report.json")
    assert len(baseline) == len(staging) == 100


    def key(row):
        return (row["episode_id"], row["timestep"])


    baseline_by_key = {key(row): row for row in baseline}
    staging_by_key = {key(row): row for row in staging}
    assert baseline_by_key.keys() == staging_by_key.keys()
    changed_predictions = 0
    for row_key, baseline_row in baseline_by_key.items():
        staging_row = staging_by_key[row_key]
        assert staging_row["applied_action"] == baseline_row["applied_action"]
        assert staging_row["target_physical"] == baseline_row["target_physical"]
        assert staging_row["checkpoint_sha256"] == report["checkpoint_sha256"]
        changed_predictions += staging_row["predicted_physical"] != baseline_row["predicted_physical"]
    assert changed_predictions > 0


def test_pilot_reports_remain_exploratory_and_confirm_closed():
    manifest = _json("pilot_manifest.json")
    for report_name in ("pilot_b1_report.json", "staging_b1_report.json"):
        report = _json(report_name)
        assert report["gate"] == "INCONCLUSIVE"
        assert report["threshold"] == "TO_LOCK"
        assert report["confirm_accessed"] is False
    assert manifest["confirm_accessed"] is False
    assert manifest["scored"] is False
