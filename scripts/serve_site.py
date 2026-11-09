"""Preview the built documentation site on loopback, with HTTP range requests so videos can seek.

Python's ``http.server`` ignores ``Range`` and always sends whole files, so browsers restart a video
instead of jumping to the chosen point. Usage: ``python scripts/serve_site.py /tmp/venn-site [--port 8080]``.
"""
import argparse
import functools
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not match or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        first, last = match.groups()
        if first:
            start, end = int(first), min(int(last), size - 1) if last else size - 1
        elif last:
            start, end = max(size - int(last), 0), size - 1
        else:
            return super().send_head()
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        source = open(path, "rb")
        source.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return source

    def copyfile(self, source, destination):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, destination)
        while remaining > 0:
            block = source.read(min(1 << 16, remaining))
            if not block:
                break
            destination.write(block)
            remaining -= len(block)
        self._remaining = None

    def end_headers(self):
        if not self._headers_buffer or b"Accept-Ranges" not in b"".join(self._headers_buffer):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    handler = functools.partial(RangeHandler, directory=args.directory)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Serving {args.directory} at http://127.0.0.1:{args.port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
