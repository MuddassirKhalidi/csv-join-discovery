# setup.py
# Usage: python setup.py
from pathlib import Path
import re
import shutil
import zipfile
import urllib.request

URL = "https://webdatacommons.org/webtables/data/sample10.zip"

def download_zip(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[skip] zip exists: {dst}")
        return dst
    print(f"[get ] {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"[ok  ] downloaded -> {dst}")
    return dst

def unzip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"[ok  ] extracted -> {out_dir}")
    # Find the sample10 folder (some archives nest differently)
    candidates = [p for p in out_dir.rglob("*") if p.is_dir() and p.name == "sample10"]
    if candidates:
        return candidates[0]
    # If archive extracted directly as sample10 at out_dir
    direct = out_dir / "sample10"
    if direct.exists():
        return direct
    raise SystemExit("sample10 folder not found after extraction.")

def organize_by_prefix(root: Path):
    pat = re.compile(r"^([^_]+)_")
    moved = 0
    for f in root.iterdir():
        if not f.is_file():
            continue
        m = pat.match(f.name)
        if not m:
            continue
        prefix = m.group(1)
        dest_dir = root / prefix
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / f.name
        if dest.exists():
            stem, suf = f.stem, f.suffix
            i = 1
            while (dest_dir / f"{stem}__dup{i}{suf}").exists():
                i += 1
            dest = dest_dir / f"{stem}__dup{i}{suf}"
        shutil.move(str(f), str(dest))
        moved += 1
    print(f"[ok  ] organized by prefix. files moved: {moved}")

def collect_csvs(sample_dir: Path, csvs_dir: Path):
    csvs_dir.mkdir(exist_ok=True)
    copied = 0
    for subdir in sample_dir.iterdir():
        if not subdir.is_dir():
            continue
        for csv_file in subdir.glob("*.csv"):
            dest = csvs_dir / csv_file.name
            if dest.exists():
                stem, suf = csv_file.stem, csv_file.suffix
                i = 1
                while (csvs_dir / f"{stem}__dup{i}{suf}").exists():
                    i += 1
                dest = csvs_dir / f"{stem}__dup{i}{suf}"
            shutil.copy2(csv_file, dest)
            copied += 1
    print(f"[ok  ] copied CSVs -> {csvs_dir} (count={copied})")

if __name__ == "__main__":
    project_root = Path.cwd()
    artifacts = project_root / "artifacts"
    zip_path = artifacts / "sample10.zip"

    download_zip(URL, zip_path)
    sample10_dir = unzip(zip_path, artifacts)

    # Ensure sample10 is at project_root/sample10 (move if extracted elsewhere)
    target_sample10 = project_root / "sample10"
    if sample10_dir != target_sample10:
        if target_sample10.exists():
            print(f"[info] removing existing {target_sample10}")
            shutil.rmtree(target_sample10)
        shutil.move(str(sample10_dir), str(target_sample10))
        print(f"[ok  ] sample10 -> {target_sample10}")

    organize_by_prefix(target_sample10)
    collect_csvs(target_sample10, project_root / "csvs")
    print("[done] setup complete.")
