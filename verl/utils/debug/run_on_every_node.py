import subprocess
import sys
from typing import Dict, List

import ray


@ray.remote
def _execute_command(command: str) -> Dict[str, str]:
    """Execute a command and return the result with node info."""
    hostname = subprocess.getoutput("hostname")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )
        return {
            "node": hostname,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
    except Exception as exc:
        return {
            "node": hostname,
            "stdout": "",
            "stderr": str(exc),
            "return_code": -1,
        }


class NodeExecutor:
    def __init__(self):
        self.nodes = [node for node in ray.nodes() if node.get("Alive")]

    def run_command(self, command: str) -> List[Dict[str, str]]:
        """Execute a shell command on every live Ray node."""
        futures = []
        for node in self.nodes:
            node_ip = node["NodeManagerAddress"]
            futures.append(
                _execute_command.options(
                    resources={f"node:{node_ip}": 0.01}
                ).remote(command)
            )
        return ray.get(futures)

    def get_node_count(self) -> int:
        """Return the number of live nodes in the cluster."""
        return len(self.nodes)


def print_results(
    results: List[Dict[str, str]],
    show_empty: bool = False,
) -> None:
    """Print execution results in a formatted way."""
    for result in results:
        print(f"\n=== Node: {result['node']} ===")
        if result["return_code"] == 0:
            if result["stdout"] or show_empty:
                print(f"Output:\n{result['stdout']}")
        else:
            print(
                f"Error (code {result['return_code']}):\n"
                f"{result['stderr']}"
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Execute commands across Ray cluster nodes"
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Command to execute on all nodes",
    )
    parser.add_argument(
        "--show-empty",
        action="store_true",
        help="Show output even if empty",
    )
    parser.add_argument(
        "--list-nodes",
        action="store_true",
        help="List all nodes in the cluster",
    )
    args = parser.parse_args()

    if not args.command and not args.list_nodes:
        parser.print_help()
        return

    ray.init()
    executor = NodeExecutor()

    if args.list_nodes:
        print(f"Found {executor.get_node_count()} nodes in the cluster:")
        for node in executor.nodes:
            name = node.get("NodeName") or node["NodeManagerAddress"]
            print(f"- {name}")
        return

    results = executor.run_command(args.command)
    print_results(results, args.show_empty)
    if any(result["return_code"] != 0 for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
