# Isolated simulator runs

The ordinary runner executes your local simulator and is appropriate for your own
circuits. For third-party circuits, use the Docker backend on a disposable scoring
machine. This is defense in depth, not a claim that a container is an impenetrable
security boundary. Do not run an entrant's Python generator to obtain its circuit.

To reproduce a leaderboard score, follow the [reader walkthrough](../REPRODUCE.md)
and [load the published reference image](#reference-image-artifact). The build
example below is for local simulator development, not obtaining the frozen image.

From the repository root:

```bash
docker build -f competition/container/Dockerfile -t spicenn2-runner:ngspice46 .
python competition/prepare_toy.py /tmp/isolated-blobs.npz
python competition/runner.py competition/examples/blobs.cir /tmp/isolated-blobs.npz \
  --epochs 10 --startup 0.02 --docker-image spicenn2-runner:ngspice46 \
  --output /tmp/isolated-blobs-score
```

Run as a normal, non-root user with access to a local Docker daemon. Python/NumPy
and the trusted runner remain on the host. Only ngspice runs inside the container;
the dataset archive and its labels never get mounted there. A remote Docker daemon
will not see the host's temporary directory, so this mode requires local bind mounts.

The backend resolves the image to an immutable local image ID before running it,
never pulls during scoring, and records the ID in `report.json`. It applies:

- No network, no Docker socket, no repository or home-directory mounts.
- Read-only root filesystem; the only host mount is a fresh simulator work directory.
- Non-root UID, all Linux capabilities dropped, and no-new-privileges.
- 2 CPUs, 4 GiB memory including swap, 64 processes, 2 GiB per output file.
- 128 MiB temporary filesystem and the runner's wall-time timeout.
- Explicit container cleanup on timeout or failure, including when the Docker client dies.

To exercise the container integration tests after building:

```bash
TEST_DOCKER_IMAGE=spicenn2-runner:ngspice46 python -m unittest competition.test_workflow -v
```

The temporary simulator directory contains the generated harness and simulator
artifacts only. Startup, training, and evaluation **inputs** are in that harness;
do not publish a private evaluation harness. On failure, safe partial traces are
retained for debugging. Symlink/special-file traces are rejected, not followed.

The Dockerfile pins the Python base manifest and ngspice 46's upstream commit
`ebdaf58ec76a06ffaac7e0f138360dd1cf5ee4b6`. No runner code or NumPy is installed
in the final image. Build dependencies come from Debian, so a later rebuild is
not guaranteed to produce identical image bytes. Rebuilds are useful for local
testing, but organizer scoring uses the exact image in [release.json](../release.json).

## Reference image artifact

Download [ngspice46-v0-image.tar.gz](https://github.com/thomasnormal/spicenn2/releases/download/v0/ngspice46-v0-image.tar.gz)
from the [v0 release](https://github.com/thomasnormal/spicenn2/releases/tag/v0)
(Linux/amd64). Verify its SHA-256 before loading it:

```bash
sha256sum ngspice46-v0-image.tar.gz
# macOS: shasum -a 256 ngspice46-v0-image.tar.gz
docker image load --input ngspice46-v0-image.tar.gz
docker image inspect --format '{{.Id}}' \
  sha256:a2ae90e8bc3a84c14f0ce18ac2151189804d06b8c571ca24455a4d896f21113d
```

Expected archive hash:
`ffad711075b754155b406ef7e49a8716f290a48ce209a6e653b17ffe2a44c045`.
The expected image ID is the one in the command above. Use that immutable ID for
`--docker-image` when following the [organizer workflow](../MAINTAINERS.md).
The archive has been tested by exporting and loading the reference image. Do not
substitute a locally rebuilt tag for this release artifact. Native ARM builds are
exploratory, not v0 reference runs.

The image retains ngspice's upstream `COPYING`, complete source archive, and build
Dockerfile under `/usr/share/doc/ngspice/`, plus the project's MIT notice. ngspice
and the base image have their own third-party licenses; the repository's MIT
license does not replace them.
