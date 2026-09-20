import subprocess
import json
import os


def get_docker_metrics():
    try:
        socket_path = os.getenv("DOCKER_SOCKET", "/var/run/docker.sock")
        if not os.path.exists(socket_path):
            return {
                "available": False,
                "error": f"Docker socket is unavailable at {socket_path}",
                "containers": [],
                "container_count": 0,
            }

        result = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{json .}}"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return {
                "available": False,
                "error": result.stderr.strip() or "Docker stats failed",
                "containers": [],
                "container_count": 0,
            }

        containers = []

        for line in result.stdout.splitlines():

            if not line.strip():
                continue

            data = json.loads(line)

            containers.append({
                "name": data.get("Name"),
                "cpu_percent": data.get("CPUPerc"),
                "memory_usage": data.get("MemUsage"),
                # Docker's MemPerc is a percentage of the container memory
                # limit (or Docker/WSL limit when no container limit exists),
                # not universally a percentage of physical host memory.
                "memory_percent_of_limit": data.get("MemPerc"),
                "network_io": data.get("NetIO"),
                "block_io": data.get("BlockIO"),
                "pids": data.get("PIDs")
            })

        return {
            "available": True,
            "container_count": len(containers),
            "containers": containers,
            "source": "Docker Engine socket",
        }

    except FileNotFoundError:

        return {
            "available": False,
            "error": "Docker command not found",
            "containers": [],
            "container_count": 0,
        }

    except subprocess.TimeoutExpired:

        return {
            "available": False,
            "error": "Docker stats timed out",
            "containers": [],
            "container_count": 0,
        }


if __name__ == "__main__":

    print(
        json.dumps(
            get_docker_metrics(),
            indent=2
        )
    )
