import io
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

from kuntur_data.download.http import AuthenticationRequired, DownloadError, download_file, redact_url


class FakeResponse:
    def __init__(self, body, status=200, headers=None, fail_after=None):
        self.body = body
        self.status = status
        self.headers = headers or {}
        self.position = 0
        self.fail_after = fail_after
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True

    def read(self, amount):
        if self.fail_after is not None and self.position >= self.fail_after:
            raise ConnectionResetError("software test interrupted stream")
        if self.position >= len(self.body):
            return b""
        end = min(self.position + amount, len(self.body))
        if self.fail_after is not None:
            end = min(end, self.fail_after)
        chunk = self.body[self.position:end]
        self.position = end
        return chunk

    def getcode(self):
        return self.status


class FakeOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        next_response = self.responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


class DownloaderTests(unittest.TestCase):
    allowed_hosts = {"data.example.test"}
    url = "https://data.example.test/science/object.bin?session=secret-value"

    @staticmethod
    def no_sleep(_seconds):
        return None

    def test_download_hashes_atomically_and_rerun_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "object.bin"
            opener = FakeOpener([FakeResponse(b"abcde", headers={"Content-Length": "5", "Content-Type": "application/octet-stream"})])
            first = download_file(
                self.url,
                path,
                allowed_hosts=self.allowed_hosts,
                expected_size_bytes=5,
                max_bytes=10,
                opener=opener,
                max_retries=0,
                sleeper=self.no_sleep,
            )
            self.assertEqual(first["status"], "DOWNLOADED")
            self.assertEqual(path.read_bytes(), b"abcde")
            second = download_file(
                self.url,
                path,
                allowed_hosts=self.allowed_hosts,
                expected_size_bytes=5,
                max_bytes=10,
                opener=FakeOpener([]),
                max_retries=0,
                sleeper=self.no_sleep,
            )
            self.assertEqual(second["status"], "ALREADY_VALID")
            self.assertTrue(second["reused"])
            self.assertNotIn("secret-value", first["url"])

    def test_resume_uses_range_after_interrupted_stream(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "object.bin"
            first = FakeResponse(b"abcde", headers={"Content-Length": "5"}, fail_after=3)
            resumed = FakeResponse(
                b"de",
                status=206,
                headers={"Content-Length": "2", "Content-Range": "bytes 3-4/5", "Content-Type": "application/octet-stream"},
            )
            opener = FakeOpener([first, resumed])
            result = download_file(
                self.url,
                path,
                allowed_hosts=self.allowed_hosts,
                expected_size_bytes=5,
                max_bytes=10,
                opener=opener,
                max_retries=1,
                retry_backoff_seconds=0,
                sleeper=self.no_sleep,
            )
            self.assertEqual(result["status"], "DOWNLOADED")
            self.assertEqual(path.read_bytes(), b"abcde")
            self.assertEqual(opener.requests[1].get_header("Range"), "bytes=3-")

    def test_401_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            error = HTTPError(self.url, 401, "unauthorized", {}, io.BytesIO(b"no"))
            opener = FakeOpener([error])
            with self.assertRaises(AuthenticationRequired):
                download_file(
                    self.url,
                    Path(folder) / "object.bin",
                    allowed_hosts=self.allowed_hosts,
                    expected_size_bytes=2,
                    max_bytes=10,
                    opener=opener,
                    max_retries=3,
                    sleeper=self.no_sleep,
                )
            self.assertEqual(len(opener.requests), 1)

    def test_does_not_replace_untracked_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "object.bin"
            path.write_bytes(b"keep")
            with self.assertRaises(DownloadError):
                download_file(
                    self.url,
                    path,
                    allowed_hosts=self.allowed_hosts,
                    expected_size_bytes=4,
                    max_bytes=10,
                    opener=FakeOpener([]),
                    max_retries=0,
                    sleeper=self.no_sleep,
                )
            self.assertEqual(path.read_bytes(), b"keep")

    def test_rejects_unapproved_host_and_redacts_query(self):
        with self.assertRaises(DownloadError):
            download_file(
                "https://evil.example.test/data?token=secret",
                Path("unused.bin"),
                allowed_hosts=self.allowed_hosts,
                max_bytes=10,
                opener=FakeOpener([]),
                max_retries=0,
                sleeper=self.no_sleep,
            )
        self.assertEqual(redact_url(self.url), "https://data.example.test/science/object.bin")


if __name__ == "__main__":
    unittest.main()
