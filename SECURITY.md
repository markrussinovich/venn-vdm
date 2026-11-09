# Security policy

## What Venn is not

Venn is not an authorization boundary and not a validated guardrail. A frozen
embedding can change its decision under paraphrase, adversarial wording,
negation, numeric changes or drift in the source data. Embeddings can be
inverted, so treat stored vectors with the same care as the text they encode.

## Running the server

The development server has no authentication, quotas or tenant isolation. It
binds to loopback by default; keep it there. Authentication, transport
security, access to caches and registries, and data retention are the
deployer's responsibility.

The API and local web demos reject Host headers other than `127.0.0.1`,
`localhost` and `[::1]`, and reject browser Origin headers that do not match
the request's scheme, hostname and port. Native clients may omit Origin.
These checks prevent foreign websites from accessing the service through
DNS rebinding or cross-origin requests; they do not authenticate local clients.

For a deliberate network deployment behind authentication and TLS, configure
exact hostnames with repeatable `venn serve --allowed-host HOSTNAME` options
or `create_app(..., allowed_hosts=[...])`. This replaces the default allowlist.
Changing `--host` alone does not allow arbitrary HTTP hostnames. Do not use
wildcards or trust arbitrary forwarded-host headers.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on this repository. If that is
not enabled, contact the maintainer privately before sharing details. Do not
post sensitive inputs, credentials, private embeddings or exploit details in
public issues. Findings are kept private until fixed or responsibly disclosed.
No security SLA is offered.
