# DevPulse Nexus architecture

## Digital Twin

`get_digital_twin()` creates a point-in-time snapshot of the developer
environment. It combines the timestamp, host telemetry from
`get_system_telemetry()`, and Docker container telemetry from
`get_docker_metrics()`. The snapshot is recorded in telemetry history and is
the factual input to deterministic diagnosis and model comparison.

## Docker-memory diagnosis

`build_diagnosis()` runs before language-model generation. It uses absolute
container memory usage, not Docker's percentage-of-container-limit value, to
calculate the share of the Docker/WSL memory budget. A material container share
means Docker may be contributing at that snapshot; it does not prove that
Docker is the sole cause of memory pressure on the Windows host.

## Evidence pipeline

DevPulse retrieves curated Markdown knowledge through ChromaDB and repository
evidence through Sourcegraph. Repository lines are scoped to the configured
repository. The local Ollama models receive the same telemetry, diagnosis, and
retrieved evidence. A deterministic evaluator then measures answer completion,
evidence alignment, citation support, clarity, and unsupported claims before a
winner can be selected.
