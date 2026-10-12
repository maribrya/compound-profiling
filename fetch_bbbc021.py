from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from zipfile import ZipFile
import urllib.request, csv, io, json, struct, zlib, hashlib, time

ROOT = Path(__file__).resolve().parent / "bbbc021"
BASE = "https://data.broadinstitute.org/bbbc/BBBC021/"


def request(url, start=None, end=None):
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=90
            ) as r:
                if start is not None:
                    assert r.status == 206, (url, r.status)
                    assert r.headers["Content-Range"].startswith(f"bytes {start}-"), r.headers
                body = r.read()
                if start is not None:
                    assert len(body) == end - start + 1
                return body
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def zip_index(plate):
    cache = ROOT / "indices" / f"{plate}_v2.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = BASE + f"BBBC021_v1_images_{plate}.zip"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60
    ) as r:
        assert r.status == 206
        size = int(r.headers["Content-Range"].split("/")[-1])
        r.read()
    tail = request(url, size - 65557, size - 1)
    pos = tail.rfind(b"PK\x05\x06")
    assert pos >= 0
    eocd = struct.unpack("<4s4H2IH", tail[pos : pos + 22])
    cd_size, cd_offset = eocd[5:7]
    raw = request(url, cd_offset, cd_offset + cd_size - 1)
    # Rebase central-directory start to zero. Central entries still contain
    # absolute original local-header offsets, which ZipFile preserves here.
    newe = bytearray(tail[pos:])
    struct.pack_into("<I", newe, 16, 0)
    out = {}
    with ZipFile(io.BytesIO(raw + newe)) as z:
        for inf in z.infolist():
            if inf.is_dir():
                continue
            out[Path(inf.filename).name] = dict(
                name=inf.filename,
                offset=inf.header_offset,
                compressed=inf.compress_size,
                size=inf.file_size,
                crc=inf.CRC,
                method=inf.compress_type,
            )
    cache.write_text(json.dumps(out))
    return out


def fetch_one(item):
    plate, filename, inf = item
    dest = ROOT / "images" / plate / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        data = dest.read_bytes()
        assert len(data) == inf["size"] and zlib.crc32(data) == inf["crc"]
        return len(data)
    url = BASE + f"BBBC021_v1_images_{plate}.zip"
    raw = request(url, inf["offset"], inf["offset"] + inf["compressed"] + 511)
    assert raw[:4] == b"PK\x03\x04"
    h = struct.unpack("<4s5H3I2H", raw[:30])
    fn, extra = h[-2:]
    assert 30 + fn + extra <= 512
    assert raw[30 : 30 + fn].decode() == inf["name"]
    data = raw[30 + fn + extra : 30 + fn + extra + inf["compressed"]]
    if inf["method"] == 8:
        data = zlib.decompress(data, -15)
    else:
        assert inf["method"] == 0, inf
    assert len(data) == inf["size"] and zlib.crc32(data) == inf["crc"]
    dest.write_bytes(data)
    return len(data)


def main():
    (ROOT / "indices").mkdir(exist_ok=True)
    moa = [
        r for r in csv.DictReader((ROOT / "BBBC021_v1_moa.csv").open()) if r["compound"] != "DMSO"
    ]
    labels = {(r["compound"], float(r["concentration"])): r["moa"] for r in moa}
    rows = list(csv.DictReader((ROOT / "BBBC021_v1_image.csv").open()))
    matches = [
        r
        for r in rows
        if (r["Image_Metadata_Compound"], float(r["Image_Metadata_Concentration"])) in labels
    ]
    matches.sort(
        key=lambda r: (
            r["Image_Metadata_Plate_DAPI"],
            r["Image_Metadata_Well_DAPI"],
            r["Image_FileName_DAPI"],
        )
    )
    chosen = {}
    for r in matches:
        key = (
            r["Image_Metadata_Compound"],
            float(r["Image_Metadata_Concentration"]),
            r["Replicate"],
        )
        if key not in chosen:
            chosen[key] = {**r, "moa": labels[key[:2]]}
    selected = list(chosen.values())
    assert (
        len(
            {
                (r["Image_Metadata_Compound"], float(r["Image_Metadata_Concentration"]))
                for r in selected
            }
        )
        == 103
    )
    assert len({r["Image_Metadata_Compound"] for r in selected}) == 38
    (ROOT / "selected_images.json").write_text(json.dumps(selected, indent=2))
    plates = sorted({r["Image_Metadata_Plate_DAPI"] for r in selected})
    print("Selected", len(selected), "fields across", len(plates), "plates", flush=True)
    indices = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        fs = {ex.submit(zip_index, p): p for p in plates}
        for f in as_completed(fs):
            indices[fs[f]] = f.result()
    jobs = {}
    for r in selected:
        plate = r["Image_Metadata_Plate_DAPI"]
        for channel in ["DAPI", "Tubulin", "Actin"]:
            name = r["Image_FileName_" + channel]
            jobs[(plate, name)] = (plate, name, indices[plate][name])
    total = 0
    start = time.time()
    manifest = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        fs = {ex.submit(fetch_one, j): j for j in jobs.values()}
        for i, f in enumerate(as_completed(fs), 1):
            total += f.result()
            j = fs[f]
            p = ROOT / "images" / j[0] / j[1]
            manifest.append(
                dict(
                    path=str(p.relative_to(ROOT)).replace("\\", "/"),
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    **j[2],
                )
            )
            if i % 50 == 0 or i == len(fs):
                print(
                    f"{i}/{len(fs)} files, {total/1e6:.1f} MB, {time.time()-start:.1f}s", flush=True
                )
    (ROOT / "image_manifest.json").write_text(
        json.dumps(sorted(manifest, key=lambda r: r["path"]), indent=2)
    )
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
