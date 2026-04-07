from pathlib import Path

import pytest


if __name__ == "__main__":
    tests_dir = Path(__file__).resolve().parent
    raise SystemExit(pytest.main([str(tests_dir)]))
