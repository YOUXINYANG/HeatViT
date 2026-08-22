#!/usr/bin/env python3
"""Resumable wheel downloader (used for large PyTorch CUDA wheels).

Usage: python tools/p2/download_wheel.py <url> <dest>

Uses urllib (same TLS stack as pip, which works inside this sandbox while
curl's schannel fails with exit 35) with a Range header for resume. Prints
progress every 25 MiB and retries on transient errors.
"""

import os
import sys
import time
import urllib.request


def main():
    url, dest = sys.argv[1], sys.argv[2]
    attempts = 20
    for attempt in range(1, attempts + 1):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                mode = "ab" if have else "wb"
                with open(dest, mode) as out:
                    last = have
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        have += len(chunk)
                        if have - last >= 25 * (1 << 20):
                            print(f"{have / 1e6:8.1f} MB", flush=True)
                            last = have
            print(f"DONE {have / 1e6:.1f} MB")
            return 0
        except Exception as exc:  # noqa: BLE001 - resume loop
            print(f"attempt {attempt}/{attempts} failed: {exc}", flush=True)
            time.sleep(10)
    print("GAVE UP", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
