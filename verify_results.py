"""Recalculate recorded summaries and independently check compound predictions."""

from pathlib import Path
import subprocess, sys, json
import numpy as np
from scipy.spatial.distance import cdist

root = Path(__file__).resolve().parent


def compare(a, b, path="root"):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            compare(a[k], b[k], path + "." + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + str(i))
    elif isinstance(a, (float, int)):
        assert np.isclose(a, b, rtol=1e-10, atol=1e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


p = root / "bbbc021"
before = json.loads((p / "classification_summary.json").read_text())
subprocess.run(
    [sys.executable, str(root / "evaluate_bbbc021.py")],
    check=True,
    stdout=subprocess.DEVNULL,
)
compare(before, json.loads((p / "classification_summary.json").read_text()))
p = root / "bbbc021"
profiles = json.loads((p / "treatment_profiles.json").read_text())
pred = json.loads((p / "predictions.json").read_text())
groups = np.array([r["compound"] for r in profiles])
y = np.array([r["moa"] for r in profiles])
conditions = ["original", "blurred", "restored"]
X = {c: np.array([r["features"][c] for r in profiles]) for c in conditions}
checked = 0
for i, r in enumerate(pred):
    train_idx = np.flatnonzero(groups != r["compound"])
    raw = X["original"][train_idx]
    mean = raw.mean(0)
    sd = np.sqrt(np.mean((raw - mean) ** 2, axis=0))
    active = sd > 1e-6
    train = (raw[:, active] - mean[active]) / sd[active]
    for c in conditions:
        test = (X[c][i, active] - mean[active]) / sd[active]
        nearest = train_idx[np.argmin(cdist(test[None, :], train, metric="cosine")[0])]
        assert groups[nearest] != r["compound"]
        assert y[nearest] == r[c + "_prediction"]
        assert groups[nearest] == r[c + "_neighbor_compound"]
        checked += 1
assert checked == 309
print("Independent cosine-distance verification: 309 predictions passed")
print(
    "Source images, model weights and original metadata were not checked by this quick audit."
)
