"""
Cleanup redundant files:
- pytest.ini (unified into pyproject.toml)
- v1/pyproject.toml (root is the single source of truth)
- v1/src/ (root src/ is the single canonical package)
"""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent

# 1. pytest.ini
pytest_ini = root / "pytest.ini"
if pytest_ini.exists():
    pytest_ini.unlink()
    print("Deleted pytest.ini")

# 2. v1/pyproject.toml
v1_pyproject = root / "v1/pyproject.toml"
if v1_pyproject.exists():
    v1_pyproject.unlink()
    print("Deleted v1/pyproject.toml")

# 3. v1/src
v1_src = root / "v1/src"
if v1_src.exists():
    shutil.rmtree(v1_src)
    print("Deleted v1/src/")

print("Cleanup completed successfully.")
