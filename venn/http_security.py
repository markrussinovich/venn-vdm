"""Browser request guards for local development services."""

from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse


LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def _origin(value):
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127
           for character in value):
        raise ValueError("Invalid origin")
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path or parsed.query or parsed.fragment
            or parsed.netloc.endswith(":")):
        raise ValueError("Invalid origin")
    port = parsed.port
    return parsed.scheme, parsed.hostname, port if port is not None else (443 if parsed.scheme == "https" else 80)


def allowed_host_set(allowed_hosts=None):
    """Return the lower-cased hostnames a guard accepts (loopback names by default); raise ``ValueError`` if invalid."""
    hosts = frozenset(host.lower() for host in (LOCAL_HOSTS if allowed_hosts is None else allowed_hosts))
    if not hosts or any("*" in host for host in hosts):
        raise ValueError("Provide explicit allowed hostnames, without wildcards")
    return hosts


class LocalHTTPGuard:
    """Reject unapproved hosts and browser requests from another origin.

    Native clients may omit Origin. This is not authentication for network deployments.
    """

    def __init__(self, app, allowed_hosts=None):
        self.app = app
        self.allowed_hosts = allowed_host_set(allowed_hosts)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        hosts = headers.getlist("host")
        try:
            if len(hosts) != 1:
                raise ValueError("Provide one Host header")
            request_origin = _origin(f'{scope["scheme"]}://{hosts[0]}')
            if request_origin[1] not in self.allowed_hosts:
                raise ValueError("Unapproved host")
        except ValueError:
            return await PlainTextResponse("Invalid Host header", status_code=400)(scope, receive, send)
        origins = headers.getlist("origin")
        if origins:
            try:
                if len(origins) != 1 or _origin(origins[0]) != request_origin:
                    raise ValueError("Foreign origin")
            except ValueError:
                return await PlainTextResponse("Invalid Origin header", status_code=403)(scope, receive, send)
        return await self.app(scope, receive, send)
