"""Download helpers shared by all dataset loaders.

- ``cached_get``: GET a URL once and cache the bytes on disk.
- ``HTTPRangeFile``: a read-only, seekable file object backed by HTTP range requests,
  so ``zipfile.ZipFile`` can pull a single member out of a large remote zip without
  downloading the whole archive.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import requests

USER_AGENT = "qmlbench/0.1 (open research benchmark; python-requests)"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def get(url: str, params: dict | None = None, retries: int = 4, timeout: float = 60.0,
        headers: dict | None = None) -> requests.Response:
    """GET with simple exponential backoff on network errors and 5xx/429 responses."""
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
                return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return r


def cached_get(url: str, cache_path: Path, params: dict | None = None,
               refresh: bool = False) -> bytes:
    """Return the body of ``url``, reading from / writing to ``cache_path``."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    content = get(url, params=params).content
    tmp = cache_path.with_suffix(cache_path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(cache_path)
    return content


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file using HTTP ``Range`` requests."""

    def __init__(self, url: str, block_size: int = 1 << 20):
        super().__init__()
        self.url = url
        head = get(url, headers={"Range": "bytes=0-0"})
        if head.status_code != 206:
            raise OSError(f"Server does not support range requests for {url}")
        self.size = int(head.headers["Content-Range"].split("/")[-1])
        self.pos = 0
        self.block_size = block_size
        self._cache: dict[int, bytes] = {}
        self.bytes_downloaded = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def _block(self, idx: int) -> bytes:
        if idx not in self._cache:
            start = idx * self.block_size
            end = min(start + self.block_size, self.size) - 1
            data = get(self.url, headers={"Range": f"bytes={start}-{end}"}).content
            self.bytes_downloaded += len(data)
            self._cache[idx] = data
        return self._cache[idx]

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, self.block_size)
            chunk = self._block(idx)[off: off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)
