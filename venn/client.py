"""Minimal HTTP client for a running Venn server."""

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class Client:
    """Calls a Venn server over HTTP using only the standard library.

    ``endpoint`` must start with ``http://`` or ``https://``; failed requests raise ``RuntimeError`` with the
    server's error body.
    """
    def __init__(self, endpoint="http://127.0.0.1:8000", *, timeout=60):
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("Use an HTTP(S) Venn endpoint")
        self.endpoint, self.timeout = endpoint.rstrip("/"), timeout

    def decide(self, state, questions, *, model="venn-1", experimental_media=False, return_vector=False):
        """POST a Decisions API request to ``/api/alpha/decisions`` and return the parsed response.

        Set ``experimental_media`` to send raw image or audio states.
        """
        headers = {"Content-Type": "application/json"}
        if experimental_media:
            headers["X-Venn-Experimental"] = "media"
        if return_vector:
            headers["X-Venn-Return-Vector"] = "1"
        request = Request(self.endpoint + "/api/alpha/decisions",
                          data=json.dumps({"model": model, "state": state,
                                           "questions": questions}, allow_nan=False).encode(),
                          headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"Venn HTTP {error.code}: {error.read().decode()}") from error

    def decide_batch(self, states, questions, *, representation="image-thirds-70", epsilon=None):
        """POST to ``/decide-batch`` and return the parsed response.

        ``states`` is a list of tagged vector dicts (sent as JSON) or a uint8 RGB image array (sent as raw
        bytes with shape metadata). Raises ``ValueError`` for other arrays.
        """
        import numpy as np

        metadata = {"model": "venn-1", "questions": questions, "representation": representation}
        if epsilon is not None:
            metadata["epsilon"] = list(epsilon)
        if isinstance(states, np.ndarray):
            if states.dtype != np.uint8 or states.ndim != 4 or states.shape[-1] != 3:
                raise ValueError("Raw image batches must be RGB uint8 arrays")
            metadata["shape"] = list(states.shape)
            body = states.tobytes()
            headers = {"Content-Type": "application/octet-stream", "X-Venn-Experimental": "media",
                       "X-Venn-Batch": json.dumps(metadata, allow_nan=False)}
        else:
            body = json.dumps({**metadata, "states": states}, allow_nan=False).encode()
            headers = {"Content-Type": "application/json"}
        request = Request(self.endpoint + "/decide-batch", data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"Venn HTTP {error.code}: {error.read().decode()}") from error
