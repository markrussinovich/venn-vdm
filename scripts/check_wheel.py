"""Smoke-test the installed wheel over loopback without optional dependencies."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def check():
    with tempfile.TemporaryDirectory(prefix="venn-wheel-check-") as directory:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}"
        env = {**os.environ, "VENN_HOME": directory, "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1"}
        server = Path(__file__).with_name("contract_server.py").resolve()
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(
                [sys.executable, "-I", str(server), "--port", str(port)],
                cwd=directory, env=env, stdout=log, stderr=log)
            try:
                for _ in range(60):
                    if process.poll() is not None:
                        raise RuntimeError("Wheel HTTP service exited before becoming ready")
                    try:
                        with urlopen(endpoint + "/health", timeout=1) as response:
                            assert json.load(response)["status"] == "ok"
                        break
                    except (URLError, TimeoutError):
                        time.sleep(.5)
                else:
                    raise RuntimeError("Wheel HTTP service did not become ready")
                with urlopen(endpoint + "/playground", timeout=5) as response:
                    assert b"Venn local playground" in response.read()
                request = Request(endpoint + "/api/alpha/decisions",
                    data=json.dumps({"model": "venn-1", "state": "Cannot log in",
                        "questions": {"team": {"type": "choice", "instructions": "Route the ticket",
                            "criteria": {"account": "Login problems", "billing": "Refunds"}}}}).encode(),
                    headers={"Content-Type": "application/json", "Origin": endpoint})
                with urlopen(request, timeout=10) as response:
                    assert "team" in json.load(response)["answers"]
                for headers, status in (({"Host": "attacker.example"}, 400),
                                        ({"Origin": "http://attacker.example"}, 403)):
                    try:
                        urlopen(Request(endpoint + "/health", headers=headers), timeout=5).close()
                    except HTTPError as error:
                        assert error.code == status
                    else:
                        raise AssertionError("Wheel accepted a foreign browser request")
            except Exception:
                log.seek(0)
                sys.stderr.write(log.read().decode(errors="replace"))
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    print("Installed wheel: HTTP decisions, playground and browser guards passed")


if __name__ == "__main__":
    check()
