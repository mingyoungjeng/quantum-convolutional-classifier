import argparse
from pathlib import Path


def handle_arguments():
    """
    Handle input arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Quantum Haar Transform experiments"
    )
    parser.add_argument("--results_path", type=Path)
    args = parser.parse_args()
    # print(args)
    return args


def main(args: argparse.Namespace):
    pass


if __name__ == "__main__":
    main(handle_arguments())
