"""List or download the complete public three-/four-piece Syzygy set."""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

BASE_URL = "http://tablebase.sesse.net/syzygy/3-4-5/"


def file_names() -> list[str]:
    with urllib.request.urlopen(BASE_URL, timeout=30) as response:
        listing = response.read().decode()
    names = re.findall(r'href="([^"]+\.(?:rtbw|rtbz))"', listing)
    return [name for name in names if len(name.partition(".")[0].replace("v", "")) <= 4]


def remote_size(name: str) -> int:
    request = urllib.request.Request(BASE_URL + name, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers["Content-Length"])


def download(name: str, destination: Path, expected_size: int) -> None:
    target = destination / name
    if target.is_file() and target.stat().st_size == expected_size:
        return
    temporary = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(BASE_URL + name, temporary)
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(f"incomplete {name}")
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", type=Path)
    arguments = parser.parse_args()
    files = [(name, remote_size(name)) for name in file_names()]
    print(f"{len(files)} files, {sum(size for _, size in files):,} bytes", flush=True)
    print(" ".join(name for name, _ in files), flush=True)
    if arguments.download is not None:
        arguments.download.mkdir(parents=True, exist_ok=True)
        for index, (name, size) in enumerate(files, 1):
            download(name, arguments.download, size)
            print(f"{index}/{len(files)} {name} {size:,}", flush=True)


if __name__ == "__main__":
    main()
