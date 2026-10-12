from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, hashlib, importlib.util, csv
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from scipy import ndimage as ndi
from skimage.measure import regionprops_table
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, clear_border, relabel_sequential

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "bbbc021"
weights = ROOT / "models/deblur_nuclei"
weight = weights


def load_model():
    torch.set_num_threads(2)
    torch.manual_seed(20261011)
    sp = importlib.util.spec_from_file_location(
        "cpnet_arch", ROOT / "vendor/cellpose/resnet_torch.py"
    )
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    net = mod.CPnet([1, 32, 64, 128, 256], 1, 3)
    net.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    net.eval()
    return net


def normalize(a):
    lo, hi = np.percentile(a, [1, 99])
    return ((a - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def load(path):
    deadline = time.time() + 1200
    while not path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"Missing source image: {path}")
        time.sleep(5)
    time.sleep(0.05)
    a = np.asarray(Image.open(path), dtype=np.float32)
    h, w = a.shape
    assert h % 2 == 0 and w % 2 == 0
    return normalize(a.reshape(h // 2, 2, w // 2, 2).mean((1, 3)))


@torch.inference_mode()
def restore(a, net):
    h, w = a.shape
    x = torch.from_numpy(normalize(a))[None, None]
    py, px = (-h) % 16, (-w) % 16
    if px or py:
        x = F.pad(x, (0, px, 0, py), mode="reflect")
    return net(x)[0][0, 0, :h, :w].numpy()


def segment(a):
    fg = a > threshold_otsu(a)
    fg = ndi.binary_fill_holes(fg)
    dist = ndi.distance_transform_edt(fg)
    peaks = peak_local_max(
        dist, min_distance=5, threshold_abs=2, labels=fg.astype(np.uint8), exclude_border=False
    )
    markers = np.zeros(a.shape, np.int32)
    if len(peaks):
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    seg = watershed(-dist, markers, mask=fg)
    sizes = np.bincount(seg.ravel())
    seg[np.isin(seg, np.flatnonzero(sizes < 30))] = 0
    return relabel_sequential(clear_border(seg))[0]


def features(channels):
    channels = [normalize(c) for c in channels]
    seg = segment(channels[0])
    inside = seg > 0
    out = {}
    p = regionprops_table(
        seg,
        properties=[
            "area",
            "eccentricity",
            "solidity",
            "extent",
            "axis_major_length",
            "axis_minor_length",
            "perimeter",
        ],
    )
    out["nuclei_count"] = float(seg.max())
    out["nuclei_area_fraction"] = float(inside.mean())

    def quantiles(prefix, v):
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        q = np.quantile(v, [0.1, 0.5, 0.9]) if len(v) else [0.0, 0.0, 0.0]
        for k, val in zip(["q10", "q50", "q90"], q):
            out[prefix + "_" + k] = float(val)

    for name, v in p.items():
        quantiles("nucleus_" + name, v)
    for channel, a in zip(["DNA", "Tubulin", "Actin"], channels):
        quantiles(channel + "_pixels", a.ravel())
        out[channel + "_mean"] = float(a.mean())
        out[channel + "_std"] = float(a.std())
        out[channel + "_in_mean"] = float(a[inside].mean()) if inside.any() else 0.0
        out[channel + "_out_mean"] = float(a[~inside].mean()) if (~inside).any() else 0.0
        ip = regionprops_table(
            seg, intensity_image=a, properties=["intensity_mean", "intensity_std", "intensity_max"]
        )
        for name, v in ip.items():
            quantiles(channel + "_nucleus_" + name, v)
        for sigma in [1, 2, 4]:
            sm = ndi.gaussian_filter(a, sigma)
            out[f"{channel}_texture_{sigma}"] = float(np.mean((a - sm) ** 2))
        gx, gy = np.gradient(a)
        out[channel + "_gradient_energy"] = float(np.mean(gx * gx + gy * gy))
    assert all(np.isfinite(list(out.values())))
    return out


def process_field(r, net):
    name = r["Image_FileName_DAPI"]
    plate = r["Image_Metadata_Plate_DAPI"]
    ch = [
        load(DATA / "images" / plate / r["Image_FileName_" + c])
        for c in ["DAPI", "Tubulin", "Actin"]
    ]
    blur = ndi.gaussian_filter(ch[0], 2, mode="reflect")
    restored = restore(blur, net)
    fs = {
        condition: features([a, ch[1], ch[2]])
        for condition, a in [("original", ch[0]), ("blurred", blur), ("restored", restored)]
    }
    return dict(
        image=name,
        plate=plate,
        compound=r["Image_Metadata_Compound"],
        concentration=float(r["Image_Metadata_Concentration"]),
        replicate=r["Replicate"],
        moa=r["moa"],
        features=fs,
    )


def main():
    net = load_model()
    selected = json.loads((DATA / "selected_images.json").read_text())
    output = DATA / "field_features.json"
    rows = json.loads(output.read_text()) if output.exists() else []
    done = {r["image"] for r in rows}
    config = dict(
        seed=20261011,
        fields=len(selected),
        compounds=38,
        treatments=103,
        classes=12,
        image_selection="first plate/well/filename per compound/concentration/replicate; no DMSO",
        downsample="2x2 area averaging for all channels",
        blur_sigma=2,
        perturbed_channel="DNA only",
        restorer="Cellpose3 deblur_nuclei",
        restorer_inference="full frame; native analysis size; padding to multiple of 16",
        restorer_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
        restorer_training_data_overlap="not excluded",
        segmenter="Otsu + filled holes + distance watershed; peak minimum distance 5, threshold 2; minimum area 30; remove border objects",
        classifier="1 nearest training treatment profile, cosine distance, train-only standardization",
        training="original images only; fixed hyperparameters; leave one whole compound out",
        profile_aggregation="arithmetic mean of available selected replicate field features",
        bootstrap="10000 paired resamples of 38 compounds",
        scope="pilot subset; no independent batch holdout",
        compute_workers=3,
        torch_threads=2,
    )
    (DATA / "experiment_config.json").write_text(json.dumps(config, indent=2))
    start = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(process_field, r, net)
            for r in selected
            if r["Image_FileName_DAPI"] not in done
        ]
        for f in as_completed(futures):
            rows.append(f.result())
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(rows))
            temporary.replace(output)
            i = len(rows)
            if i % 10 == 0 or i == len(selected):
                print(f"{i}/{len(selected)} fields, {time.time()-start:.1f}s", flush=True)
    assert len(rows) == len(selected)
    print("Feature extraction complete", flush=True)


if __name__ == "__main__":
    main()
