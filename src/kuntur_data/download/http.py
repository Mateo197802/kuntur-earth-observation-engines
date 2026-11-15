"""Conservative HTTPS downloader with bounded retries and atomic finalization."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class DownloadError(RuntimeError):
    """Base exception for safe download failures."""


class AuthenticationRequired(DownloadError):
    """Raised for HTTP 401/403 without trying alternate identities or endpoints."""


def redact_url(url: str) -> str:
    """Return a URL suitable for logs without query strings or fragments."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _url_fingerprint(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


@dataclass
class RateLimiter:
    minimum_interval_seconds: float = 1.0
    _last_request_at: float | None = None

    def wait(self, sleeper: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        if self._last_request_at is None:
            self._last_request_at = clock()
            return
        delay = self.minimum_interval_seconds - (clock() - self._last_request_at)
        if delay > 0:
            sleeper(delay)
        self._last_request_at = clock()


class _RedirectPolicy(HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str], token_present: bool):
        super().__init__()
        self.allowed_hosts = allowed_hosts
        self.token_present = token_present

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Message, newurl: str) -> Request | None:
        target = urlsplit(newurl)
        source = urlsplit(request.full_url)
        host = (target.hostname or "").lower()
        if target.scheme.lower() != "https" or host not in self.allowed_hosts:
            raise DownloadError("Blocked redirect outside the configured HTTPS host allowlist.")
        if self.token_present and host != (source.hostname or "").lower():
            raise DownloadError("Blocked cross-host redirect while Earthdata authorization is attached.")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _validate_url(url: str, allowed_hosts: set[str]) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    normalized_hosts = {name.lower() for name in allowed_hosts}
    if parts.scheme.lower() != "https" or not host or host not in normalized_hosts:
        raise DownloadError("URL must use HTTPS and an explicitly allowed provider host.")
    if parts.username or parts.password:
        raise DownloadError("URLs containing embedded credentials are forbidden.")
    return host


def download_file(
    url: str,
    destination: Path,
    *,
    allowed_hosts: set[str],
    expected_size_bytes: int | None = None,
    expected_sha256: str | None = None,
    max_bytes: int,
    token: str | None = None,
    use_environment_token: bool = True,
    timeout_seconds: float = 60.0,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    rate_limiter: RateLimiter | None = None,
    opener: Any | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Download one approved object without replacing an existing final file.

    A partial file is retained after network or integrity failures for safe review or
    resume. Existing final files are accepted only when a provenance sidecar matches
    the URL fingerprint and the stored checksum verifies.
    """
    _validate_url(url, allowed_hosts)
    safe_url = redact_url(url)
    destination = Path(destination)
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative.")
    if expected_size_bytes is not None and expected_size_bytes > max_bytes:
        raise DownloadError("Expected object size exceeds the remaining transfer budget.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")
    state_path = destination.with_name(destination.name + ".part.json")
    provenance_path = destination.with_name(destination.name + ".provenance.json")
    fingerprint = _url_fingerprint(url)

    if destination.exists():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provenance = {}
        actual_hash = sha256_file(destination)
        if (
            provenance.get("url_fingerprint") == fingerprint
            and provenance.get("sha256") == actual_hash
            and (expected_sha256 is None or actual_hash.lower() == expected_sha256.lower())
            and (expected_size_bytes is None or destination.stat().st_size == expected_size_bytes)
        ):
            return {
                "status": "ALREADY_VALID",
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": actual_hash,
                "url": safe_url,
                "reused": True,
            }
        raise DownloadError("Destination already exists but has no matching verified provenance; it was left untouched.")

    if part_path.exists():
        try:
            partial_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DownloadError("An untracked partial file exists; it was preserved for inspection.") from exc
        if partial_state.get("url_fingerprint") != fingerprint:
            raise DownloadError("A partial file belongs to a different URL; it was preserved.")
        if partial_state.get("expected_size_bytes") != expected_size_bytes:
            raise DownloadError("A partial file has a different expected size; it was preserved.")
        offset = part_path.stat().st_size
        if offset > max_bytes:
            raise DownloadError("Existing partial file exceeds the remaining transfer budget.")
    else:
        offset = 0
        state_path.write_text(
            json.dumps(
                {"url_fingerprint": fingerprint, "expected_size_bytes": expected_size_bytes},
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    if expected_size_bytes is not None and offset > expected_size_bytes:
        raise DownloadError("Existing partial file exceeds CMR's expected size; it was preserved.")
    if expected_size_bytes is not None and offset == expected_size_bytes and part_path.exists():
        digest = sha256_file(part_path)
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise DownloadError("SHA-256 mismatch for a complete partial file; it was preserved.")
        if destination.exists():
            raise DownloadError("Destination appeared during transfer; it was left untouched.")
        try:
            os.link(part_path, destination)
        except FileExistsError as exc:
            raise DownloadError("Destination appeared during transfer; it was left untouched.") from exc
        except OSError as exc:
            raise DownloadError("Could not safely finalize the complete partial file without replacing an existing path.") from exc
        part_path.unlink()
        provenance_path.write_text(
            json.dumps(
                {
                    "url_fingerprint": fingerprint,
                    "source_url_redacted": safe_url,
                    "bytes": offset,
                    "sha256": digest,
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "expected_size_bytes": expected_size_bytes,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass
        return {
            "status": "DOWNLOADED",
            "path": str(destination),
            "bytes": offset,
            "sha256": digest,
            "url": safe_url,
            "reused": False,
        }

    if token is None and use_environment_token:
        token = os.environ.get("EARTHDATA_TOKEN")
    active_opener = opener
    if active_opener is None:
        active_opener = build_opener(_RedirectPolicy({h.lower() for h in allowed_hosts}, bool(token)))
    limiter = rate_limiter or RateLimiter()
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        headers = {"User-Agent": "kuntur-data/0.1 (bounded scientific data transfer)"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(url, headers=headers)
        limiter.wait(sleeper=sleeper)
        try:
            response = active_opener.open(request, timeout=timeout_seconds)
            with response:
                status = int(response.status if hasattr(response, "status") else response.getcode())
                if status in (401, 403):
                    raise AuthenticationRequired(f"Provider returned HTTP {status}; obtain legitimate access.")
                if status not in (200, 206):
                    raise DownloadError(f"Provider returned unexpected HTTP status {status}.")
                if offset and status != 206:
                    raise DownloadError("Provider did not honor the byte-range resume request; partial file was preserved.")
                content_range = response.headers.get("Content-Range", "")
                if offset and not content_range.startswith(f"bytes {offset}-"):
                    raise DownloadError("Provider returned a mismatched Content-Range; partial file was preserved.")
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    raise AuthenticationRequired("Provider returned HTML instead of a data object; check legitimate access.")
                response_length = response.headers.get("Content-Length")
                remaining_budget = max_bytes - offset
                if response_length is not None and int(response_length) > remaining_budget:
                    raise DownloadError("Response Content-Length exceeds the remaining transfer budget.")
                mode = "ab" if offset else "wb"
                received_this_attempt = 0
                with part_path.open(mode) as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        received_this_attempt += len(block)
                        if offset + received_this_attempt > max_bytes:
                            raise DownloadError("Transfer reached the configured byte budget; partial file was preserved.")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                total_size = part_path.stat().st_size
                if response_length is not None and received_this_attempt != int(response_length):
                    raise DownloadError("Response ended before its advertised Content-Length; partial file was preserved.")
                if expected_size_bytes is not None and total_size != expected_size_bytes:
                    if total_size < expected_size_bytes:
                        raise DownloadError("Downloaded object is incomplete relative to CMR's expected size; partial file was preserved.")
                    raise DownloadError("Downloaded object is larger than CMR's expected size; partial file was preserved.")
                digest = sha256_file(part_path)
                if expected_sha256 and digest.lower() != expected_sha256.lower():
                    raise DownloadError("SHA-256 mismatch; the untrusted partial file was preserved.")
                if destination.exists():
                    raise DownloadError("Destination appeared during transfer; it was left untouched.")
                try:
                    os.link(part_path, destination)
                except FileExistsError as exc:
                    raise DownloadError("Destination appeared during transfer; it was left untouched.") from exc
                except OSError as exc:
                    raise DownloadError("Could not safely finalize the file without replacing an existing path.") from exc
                part_path.unlink()
                provenance = {
                    "url_fingerprint": fingerprint,
                    "source_url_redacted": safe_url,
                    "bytes": total_size,
                    "sha256": digest,
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "expected_size_bytes": expected_size_bytes,
                }
                provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
                try:
                    state_path.unlink()
                except FileNotFoundError:
                    pass
                return {
                    "status": "DOWNLOADED",
                    "path": str(destination),
                    "bytes": total_size,
                    "sha256": digest,
                    "url": safe_url,
                    "reused": False,
                }
        except AuthenticationRequired:
            raise
        except (HTTPError, URLError, TimeoutError, OSError, DownloadError) as exc:
            if isinstance(exc, HTTPError) and exc.code in (401, 403):
                raise AuthenticationRequired(f"Provider returned HTTP {exc.code}; obtain legitimate access.") from exc
            last_error = exc
            if isinstance(exc, DownloadError) and any(
                text in str(exc)
                for text in (
                    "budget",
                    "Content-Range",
                    "did not honor",
                    "incomplete relative",
                    "larger than CMR",
                    "SHA-256 mismatch",
                    "already exists",
                    "different URL",
                    "different expected size",
                )
            ):
                raise
            if attempt >= max_retries:
                break
            sleeper(retry_backoff_seconds * (2**attempt))
            offset = part_path.stat().st_size if part_path.exists() else 0

    raise DownloadError(f"Download failed after bounded retries from {safe_url}: {type(last_error).__name__ if last_error else 'unknown error'}") from last_error
