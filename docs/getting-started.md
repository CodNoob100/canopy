# Try CANOPY: quick path and advanced path

CANOPY's core architecture scoring needs **Python 3.10+** and does not require a GPU, PyTorch, benchmark files, or model downloads. A score is an **architecture-level region-count upper bound**, not a measured accuracy or exact region count.

## Path A — first result in minutes

From a terminal with Python 3.10+:

```bash
git clone https://github.com/CodNoob100/canopy.git
cd canopy
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1` (PowerShell execution policy may require using Command Prompt instead)
- Windows Command Prompt: `.venv\Scripts\activate.bat`

Then run the same commands on any platform:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m canopy.cli --version
python -m canopy.cli score --widths 2,4,1
```

You should see a nonnegative integer followed by `architecture-level region-count upper bound`. No benchmark dataset is involved. The `python -m` invocation avoids shell/PATH issues with installed console scripts.

For JSON output:

```bash
python -m canopy.cli score --widths 2,64,64,1 --mode tight --json
```

The JSON `score` is intentionally a string to preserve large integers. Compare the `generic` and `tight` modes; neither predicts real-world accuracy by itself.

## Path B — granular, reproducible developer workflow

Install the developer extras and run the tests before changing anything:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m pytest tests/test_pipeline.py -q
```

On Windows PowerShell, the quoted `'.[dev]'` requirement is also accepted. If a test filename differs on your checkout, inspect `tests/` or run the full suite.

Inspect the CLI and its options:

```bash
python -m canopy.cli --help
python -m canopy.cli score --help
python -m canopy.cli evaluate --help
```

Programmatic scoring:

```python
from canopy.nas.architecture import ArchitectureSpec
from canopy.nas.tropical_score import tropical_expressivity_score

arch = ArchitectureSpec(layer_widths=(2, 64, 64, 1))
generic = tropical_expressivity_score(arch, mode="generic")
tight = tropical_expressivity_score(arch, mode="tight")
print({"generic_bound": str(generic), "tight_bound": str(tight)})
```

### Real-data evaluation (advanced, optional)

Real benchmark evaluation is **not** part of the quickstart. Obtain NATS-Bench TSS data separately from a trusted source, and treat serialized benchmark artifacts as potentially unsafe. Never treat a synthetic benchmark or mocked test as empirical validation.

```bash
python -m pip install -e '.[bench]'
python -m canopy.cli evaluate \
  --benchmark-data /trusted/path/to/NATS-tss-v1_0-3ffb9-simple \
  --dataset cifar100 --hp 200 --seed 42 --limit 200 \
  --output results/cifar100-seed42
```

The output directory must not already exist. Use a different output directory for each run. For a **single file**, `--sha256` can check a digest obtained from an independent trusted source; it does not establish source trust on its own. For full metric definitions, undefined-correlation handling, and provenance caveats, read [CLI documentation](cli.md).

### Troubleshooting

- `python` not found: install Python 3.10+ and verify `python --version` (some systems use `python3` or Windows `py -3`).
- `No module named canopy`: activate the same virtual environment used for installation, then rerun `python -m pip install -e .`.
- `canopy` command not found: use `python -m canopy.cli` as shown above.
- Evaluation fails with missing benchmark data: scoring does **not** require benchmark files; return to Path A. Do not enable synthetic fallback to claim measured accuracy.
- Dependency installation fails: verify your Python version, upgrade pip inside the environment, and copy the full installation error into a GitHub issue.

No benchmark data, network model downloads, or GPU are needed for the quick path.
