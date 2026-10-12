from pathlib import Path

import urllib.request, csv, io, json

ROOT = Path(__file__).resolve().parent / "bbbc021"

BASE = "https://data.broadinstitute.org/bbbc/BBBC021/"


def main():
    ROOT.mkdir(exist_ok=True)
    for name in ["BBBC021_v1_image.csv", "BBBC021_v1_moa.csv"]:
        p = ROOT / name
        if not p.exists():
            p.write_bytes(urllib.request.urlopen(BASE + name, timeout=60).read())
        rows = list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8-sig"))))
        print(name, "rows", len(rows), "columns", list(rows[0]), "first", rows[:2], flush=True)
    url = BASE + "BBBC021_v1_images_Week1_22123.zip"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"Range": "bytes=0-63"}), timeout=60
    ) as r:
        print("Range response", r.status, dict(r.headers), flush=True)
        print("First bytes", r.read(64), flush=True)


if __name__ == "__main__":
    main()
