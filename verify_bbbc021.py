from pathlib import Path
from collections import Counter
import json, csv, hashlib
import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent / "bbbc021"


def main():
    p = json.loads((ROOT / "treatment_profiles.json").read_text())
    pred = json.loads((ROOT / "predictions.json").read_text())
    summary = json.loads((ROOT / "classification_summary.json").read_text())
    fields = json.loads((ROOT / "field_features.json").read_text())
    selected = json.loads((ROOT / "selected_images.json").read_text())
    assert {r["image"] for r in fields} == {r["Image_FileName_DAPI"] for r in selected}
    lookup = {
        (r["compound"], float(r["concentration"])): r["moa"]
        for r in csv.DictReader((ROOT / "BBBC021_v1_moa.csv").open())
    }
    for r in pred:
        assert lookup[r["compound"], r["concentration"]] == r["moa"]
    groups = np.array([r["compound"] for r in p])
    labels = np.array([r["moa"] for r in p])
    conditions = ["original", "blurred", "restored"]
    X = {c: np.array([r["features"][c] for r in p]) for c in conditions}
    for i, r in enumerate(pred):
        indices = np.flatnonzero(groups != r["compound"])
        raw = X["original"][indices]
        mu = np.mean(raw, axis=0)
        sd = np.sqrt(np.mean((raw - mu) ** 2, axis=0))
        mask = sd > 1e-6
        train = (raw[:, mask] - mu[mask]) / sd[mask]
        for c in conditions:
            test = (X[c][i, mask] - mu[mask]) / sd[mask]
            distance = cdist(test[None, :], train, metric="cosine")[0]
            nearest = indices[np.argmin(distance)]
            assert labels[nearest] == r[c + "_prediction"], (i, c)
            assert groups[nearest] == r[c + "_neighbor_compound"]
    for c in conditions:
        good = sum(r[c + "_prediction"] == r["moa"] for r in pred)
        assert good == summary["metrics"][c]["correct"]
    for change, vals in summary["comparisons"].items():
        left, right = change.split("_to_")
        fixed = sum(
            r[left + "_prediction"] != r["moa"] and r[right + "_prediction"] == r["moa"]
            for r in pred
        )
        harmed = sum(
            r[left + "_prediction"] == r["moa"] and r[right + "_prediction"] != r["moa"]
            for r in pred
        )
        assert fixed == vals["corrected"] and harmed == vals["harmed"]
        assert np.isclose(vals["accuracy_difference"], (fixed - harmed) / len(pred))
    source_manifest = json.loads((ROOT / "image_manifest.json").read_text())
    assert len(source_manifest) == len(selected) * 3
    for r in source_manifest:
        path = ROOT / r["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r["sha256"]
    audit = dict(
        status="passed",
        fields=len(fields),
        profiles=len(p),
        compounds=len(set(groups)),
        classes=len(set(labels)),
        source_images_verified=len(source_manifest),
        independent_classifier_predictions=len(pred) * 3,
        corrected_harmed_identity_verified=True,
        metadata_labels_verified=True,
    )
    (ROOT / "independent_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
