"""Download the Zenodo record of the spectra and unpack it where the scripts expect it.

    data/spectra/*.asc            raw Andor exports (from measurements_RAW_data.zip)
    data/repeatability.xlsx, data/power_dependence.xlsx
    code/v6_out, code/v10_out, code/*.csv    derived tables (from derived_results.zip)

Standard library only. Run from anywhere: python code/fetch_data.py
"""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

RECORD_ID = "23817355"
ROOT = Path(__file__).resolve().parent.parent


def fetch(url, dest=None):
    req = urllib.request.Request(url, headers={"User-Agent": "gated-thermometry-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if dest is None:
            return json.load(r)
        h = hashlib.md5()
        with open(dest, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                h.update(b)
        return h.hexdigest()


def main():
    meta = fetch(f"https://zenodo.org/api/records/{RECORD_ID}")
    print("record", RECORD_ID, "-", meta["metadata"]["title"])
    tmp = ROOT / "data" / "_download"
    tmp.mkdir(parents=True, exist_ok=True)
    for f in meta["files"]:
        key = f["key"]
        dest = tmp / key
        md5 = fetch(f["links"]["self"], dest)
        expected = f.get("checksum", "").replace("md5:", "")
        if expected and md5 != expected:
            raise RuntimeError(f"checksum mismatch for {key}")
        print("downloaded", key, round(dest.stat().st_size / 1e6, 2), "MB, md5 ok")
    (ROOT / "data" / "spectra").mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(tmp / "measurements_RAW_data.zip").extractall(ROOT / "data" / "spectra")
    for x in ("repeatability.xlsx", "power_dependence.xlsx"):
        (ROOT / "data" / x).write_bytes((tmp / x).read_bytes())
    zipfile.ZipFile(tmp / "derived_results.zip").extractall(ROOT / "code")
    print("data ready in", ROOT / "data", "and derived tables in", ROOT / "code")


if __name__ == "__main__":
    main()
