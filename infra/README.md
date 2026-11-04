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

## Hailo compiler

Optional maintainer tooling: compiles the shuttle and court models to Hailo HEF,
locally or on an AWS spot GPU. For using it, see
[Accelerators](../docs/accelerators.md#hailo).

| Path | What |
|---|---|
| `hailo/hailo.mk` | the `hailo-*` make targets, included by the root `Makefile` |
| `hailo/hailo.sh` | the shell behind them |
| `hailo/image/` | the compile image on Hailo's AI Software Suite, and `export_hailo.py`; used by both |
| `hailo/cloud/` | the AWS compile VM, as two OpenTofu stacks |

### Compile VM (AWS)

Needs an AWS account with spot G-instance quota in its region; the default is
us-east-2, set by the persistent stack's `region` variable, and the runner
follows it. Each stack keeps its state in a local, gitignored file:

| Stack | Creates | Lifetime | Cost |
|---|---|---|---|
| `hailo/cloud/persistent` | S3 bucket, VM role and instance profile, an outbound-only security group in the default VPC | applied once | ~$0.20/month for the 8.5 GB suite tarball |
| `hailo/cloud/runner` | one spot `g4dn.2xlarge` from the AWS Deep Learning Base GPU AMI | one run, applied and destroyed by `make hailo-cloud` | ~$0.39/hour while it exists, plus its public IPv4 at $0.005/hour |

The bucket's `runs/` prefix holds a run's inputs and results, and is deleted
once they are downloaded; a lifecycle rule expires whatever is left after 7
days, and incomplete uploads after one.

The VM has no inbound access; `make hailo-cloud-shell` reaches it through
Session Manager. It can read the suite and read and write `runs/`, and nothing
else in the account.

#### Tearing down

```bash
make hailo-cloud-teardown
```

It asks for the bucket's name first, then destroys a run's VM if one is still
up, empties the bucket (which `destroy` requires) and destroys the persistent
stack. The default subnets stay; they cost nothing. To keep the stack and stop
paying only for the suite, delete just `suite/` and upload it again before the
next run.
