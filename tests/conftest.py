import sys
from pathlib import Path

# Allow `import serving` / `import streaming` when pytest is run from the repo root
# without the package being pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
