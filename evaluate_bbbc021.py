from pathlib import Path
import numpy as np
import json, csv, hashlib

ROOT = Path(__file__).resolve().parent / "bbbc021"
CONDITIONS = ["original", "blurred", "restored"]


def main():
    fields = json.loads((ROOT / "field_features.json").read_text())
    config = json.loads((ROOT / "experiment_config.json").read_text())
    assert len(fields) == config["fields"]
    grouping = {}
    for r in fields:
        grouping.setdefault((r["compound"], r["concentration"]), []).append(r)
    keys = sorted(grouping)
    feature_names = sorted(fields[0]["features"]["original"])
    profiles = []
    for key in keys:
        rs = grouping[key]
        labels = {r["moa"] for r in rs}
        assert len(labels) == 1
        assert len(set(r["replicate"] for r in rs)) == len(rs)
        fs = {
            c: np.mean([[r["features"][c][f] for f in feature_names] for r in rs], axis=0).tolist()
            for c in CONDITIONS
        }
        profiles.append(
            dict(
                compound=key[0],
                concentration=key[1],
                moa=rs[0]["moa"],
                replicates=len(rs),
                features=fs,
            )
        )
    assert len(profiles) == 103
    X = {c: np.asarray([r["features"][c] for r in profiles]) for c in CONDITIONS}
    group = np.array([r["compound"] for r in profiles])
    y = np.array([r["moa"] for r in profiles])
    compounds = np.unique(group)
    classes = np.unique(y)
    assert len(compounds) == 38 and len(classes) == 12
    preds = {c: np.empty(len(profiles), dtype=object) for c in CONDITIONS}
    neighbors = {c: np.empty(len(profiles), dtype=int) for c in CONDITIONS}
    fold_logs = []
    for compound in compounds:
        train = np.flatnonzero(group != compound)
        test = np.flatnonzero(group == compound)
        assert not (set(group[train]) & set(group[test]))
        assert set(y[test]) <= set(y[train])
        mean = X["original"][train].mean(0)
        std = X["original"][train].std(0)
        active = std > 1e-6
        a = (X["original"][train][:, active] - mean[active]) / std[active]
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        for c in CONDITIONS:
            b = (X[c][test][:, active] - mean[active]) / std[active]
            b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
            nn = train[np.argmax(b @ a.T, axis=1)]
            neighbors[c][test] = nn
            preds[c][test] = y[nn]
            assert not np.any(group[nn] == compound)
        fold_logs.append(
            dict(
                held_out_compound=str(compound),
                train_profiles=len(train),
                test_profiles=len(test),
                active_features=int(active.sum()),
            )
        )
    rows = []
    for i, r in enumerate(profiles):
        rows.append(
            dict(
                compound=r["compound"],
                concentration=r["concentration"],
                moa=r["moa"],
                replicates=r["replicates"],
                **{c + "_prediction": str(preds[c][i]) for c in CONDITIONS},
                **{c + "_neighbor_compound": str(group[neighbors[c][i]]) for c in CONDITIONS},
                **{
                    c + "_neighbor_concentration": profiles[neighbors[c][i]]["concentration"]
                    for c in CONDITIONS
                }
            )
        )
    totals = np.array([(group == g).sum() for g in compounds])
    correct = {c: preds[c] == y for c in CONDITIONS}
    sums = {c: np.array([correct[c][group == g].sum() for g in compounds]) for c in CONDITIONS}
    rng = np.random.default_rng(20261011)
    counts = rng.multinomial(
        len(compounds), np.repeat(1 / len(compounds), len(compounds)), size=10000
    )
    denom = counts @ totals
    ba = {c: (counts @ sums[c]) / denom for c in CONDITIONS}
    metrics = {}
    for c in CONDITIONS:
        byclass = {
            str(cl): dict(
                n=int((y == cl).sum()),
                correct=int(correct[c][y == cl].sum()),
                recall=float(correct[c][y == cl].mean()),
            )
            for cl in classes
        }
        conf = np.array([[((y == t) & (preds[c] == p)).sum() for p in classes] for t in classes])
        metrics[c] = dict(
            n=103,
            correct=int(correct[c].sum()),
            accuracy=float(correct[c].mean()),
            balanced_accuracy=float(np.mean([v["recall"] for v in byclass.values()])),
            accuracy_ci95=np.quantile(ba[c], [0.025, 0.975]).tolist(),
            by_class=byclass,
            confusion_matrix=conf.tolist(),
        )
    comparisons = {}
    for left, right in [("blurred", "restored"), ("original", "blurred"), ("original", "restored")]:
        fixed = (~correct[left]) & correct[right]
        harmed = correct[left] & (~correct[right])
        comparisons[left + "_to_" + right] = dict(
            accuracy_difference=float(correct[right].mean() - correct[left].mean()),
            ci95=np.quantile(ba[right] - ba[left], [0.025, 0.975]).tolist(),
            corrected=int(fixed.sum()),
            harmed=int(harmed.sum()),
            changed_prediction=int((preds[left] != preds[right]).sum()),
        )
    summary = dict(
        fields=len(fields),
        treatments=103,
        compounds=38,
        classes=classes.tolist(),
        features=len(feature_names),
        metrics=metrics,
        comparisons=comparisons,
        bootstrap_replicates=10000,
        bootstrap_unit="compound",
        seed=20261011,
        feature_rows_sha256=hashlib.sha256((ROOT / "field_features.json").read_bytes()).hexdigest(),
    )
    for name, obj in [
        ("treatment_profiles.json", profiles),
        ("predictions.json", rows),
        ("fold_audit.json", fold_logs),
        ("classification_summary.json", summary),
        ("feature_names.json", feature_names),
    ]:
        (ROOT / name).write_text(json.dumps(obj, indent=2))
    with (ROOT / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(
        json.dumps(
            {
                c: {k: v for k, v in m.items() if k not in ["by_class", "confusion_matrix"]}
                for c, m in metrics.items()
            },
            indent=2,
        )
    )
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
