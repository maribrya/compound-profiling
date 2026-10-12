from pathlib import Path

from urllib.request import urlopen

from hashlib import sha256


def main():
    p = Path(__file__).resolve().parent / "models/deblur_nuclei"
    p.parent.mkdir(exist_ok=True)
    if not p.exists():
        p.write_bytes(urlopen("https://www.cellpose.org/models/deblur_nuclei", timeout=120).read())
    expected = "00a2c9a84998c8e32cbfbbb7d7dfcf20a7d1413081ecb3e6a30a1c3e5b48171c"
    assert (
        sha256(p.read_bytes()).hexdigest() == expected
    ), "Upstream weights changed; do not treat as an exact repeat."
    print("Official model checksum verified.")


if __name__ == "__main__":
    main()
