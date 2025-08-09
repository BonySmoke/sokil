# Infrastructure

Deploy infrastructure around `Sokil`.

## CVAT

`Sokil` integrates with CVAT for data annotation and later training. This section describes how to run and upgrade it.

| | |
|---|---|
| Upstream | https://github.com/cvat-ai/cvat/blob/v2.75.0/docker-compose.yml |
| Vendored version | **v2.75.0** |

### Running

```bash
cp infra/.env.example infra/.env    # once
make cvat-up
make cvat-superuser                 # once, creates the admin login, store it in infra/.env
make help | grep cvat               # see available commands for CVAT
```

### Upgrading

1. Download `docker-compose.yml` for the new tag from the upstream URL: https://github.com/cvat-ai/cvat
2. Selectively apply the diff of the used services. Upstream brings services like ClickHouse that are unused but consume resources.
3. Bump `CVAT_VERSION` in `infra/.env` and in `.env.example`
4. Bump `cvat-sdk` python package version in `pyproject.toml`.
5. `make cvat-pull && make cvat-up`. The server migrates its own database on start.
Watch logs using `make cvat-logs` to spot errors.
