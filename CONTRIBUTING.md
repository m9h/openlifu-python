# Contributing to openlifu

Thank you for your interest in contributing to **openlifu**, the open-source
toolkit for planning and controlling transcranial focused ultrasound (tFUS)
treatments. This guide covers the development workflow, coding standards, and
domain-specific considerations for the project.

## 1. Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (preferred) or pip
- For GPU-accelerated jwave simulations: CUDA 12+ and JAX with CUDA support

### Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/OpenwaterHealth/openlifu-python.git
cd openlifu-python

# Create a virtual environment and install in editable mode
uv venv
uv pip install -e ".[dev,test,docs]"

# Verify the installation
uv run pytest tests/ -x --timeout=60
```

### Project Layout

```
openlifu-python/
  src/openlifu/          # Library source
    seg/seg_methods/     # Segmentation (uniform, heterogeneous skull)
    sim/                 # Simulation backends (k-Wave, jwave, phase correction)
    bf/                  # Beamforming (delays, apodization, focal patterns)
    plan/                # Treatment planning (protocols, solutions, analysis)
    xdc/                 # Transducer geometry
    io/                  # Hardware interface (TX7332, HV controller)
  tests/                 # pytest test suite
  benchmarks/            # GPU benchmarks (Modal, A100)
  docs/                  # Sphinx documentation (RST)
  examples/              # Runnable demo scripts
```

## 2. Development Workflow

1. **Create a feature branch** from `main` (or from the relevant feature
   branch, e.g. `feature/heterogeneous-skull-segmentation`).
2. **Write tests first** (TDD). Place tests in `tests/test_<module>.py`.
3. **Implement the feature.** Follow the existing coding style (dataclasses,
   `to_dict()`/`from_dict()` serialization, xarray for simulation grids).
4. **Run the test suite:**
   ```bash
   uv run pytest tests/ -x -q
   ```
5. **Update documentation** if you add or change public API.
6. **Open a pull request** against the target branch with a clear description
   of the change and any benchmark results.

### Branching Conventions

- `main` -- stable release branch
- `feature/<name>` -- feature development branches
- `fix/<name>` -- bug-fix branches

## 3. Coding Standards

### Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) with a 100-character line
  limit.
- Use **NumPy-style docstrings** for all public functions and classes. Include
  `Parameters`, `Returns`, and (where appropriate) `Examples` sections.
- Type-annotate all function signatures (`from __future__ import annotations`).
- Prefer `dataclasses` for domain objects; implement `to_dict()` and
  `from_dict()` via the `DictMixin` base class.

### Simulation Conventions

- Acoustic property maps use `xarray.Dataset` with variables `sound_speed`,
  `density`, and `attenuation`, each carrying `units` and `ref_value` attrs.
- Tissue labels follow the **openlifu convention**: 0=water, 1=scalp, 2=skull,
  3=CSF, 4=gray_matter, 5=white_matter. If you add a new label source (e.g. a
  new atlas), provide a remapping function (see `remap_simnibs_labels`).
- Coordinate units must be explicit (`mm`, `m`, etc.) and stored in xarray
  `attrs["units"]`.

### Units

The library uses SI internally for simulation (metres, seconds, Pascals) but
allows user-facing quantities in clinical units (mm, kHz, MPa). Always use
`openlifu.util.units.getunitconversion()` for conversions.

## 4. Testing

### Running Tests

```bash
# Full suite
uv run pytest tests/

# Single module
uv run pytest tests/test_phase_correction.py -v

# Skip tests that require external data
uv run pytest tests/ -k "not birnbaum and not tfuscapes"
```

### Test Categories

- **Unit tests**: Pure-Python logic (serialization, geometry, label remapping).
- **Simulation tests**: Require JAX/jwave. Marked with `pytest.mark.skipif`
  when the dependency is unavailable.
- **Data tests**: Require external datasets (Birnbaum, TFUScapes, SCI head
  model). Skipped when data is not present.
- **Hardware tests**: Mock-based tests for the TX7332 and HV controller.

### Writing Good tFUS Tests

When testing acoustic simulations:
- Use small grids (32x32x32 or smaller) to keep CI fast.
- Assert on **physical plausibility** (pressure > 0, peak inside the domain,
  skull attenuation reduces focal pressure vs. water).
- For phase-correction tests, verify that optimised delays **improve** focal
  pressure relative to random delays -- do not assert exact values, because
  Helmholtz solver tolerance and grid resolution cause small variations.

## 5. Documentation

The documentation is built with Sphinx and hosted on Read the Docs.

```bash
# Build locally
cd docs
uv run make html
open _build/html/index.html
```

### Adding a Tutorial

1. Create `docs/tutorials/<name>.rst` using reStructuredText.
2. Add the tutorial to the `toctree` in `docs/index.rst`.
3. Include code blocks with `.. code-block:: python` and cross-reference API
   with `:func:`, `:class:`, and `:mod:` roles.

### API Documentation

Public modules are documented via `sphinx.ext.autodoc` and
`sphinx.ext.autosummary`. When you add a new module, add a corresponding
`automodule` directive in `docs/api.rst`.

## 6. AI-Assisted Development

This project welcomes contributions that use AI coding assistants (GitHub
Copilot, Claude Code, ChatGPT, etc.) under the following guidelines:

- **Review all generated code.** AI tools can produce plausible-looking but
  physically incorrect acoustic parameters. Always verify against published
  ITRUSST benchmarks or the IT'IS tissue property database.
- **Validate simulation results.** If an AI tool generates a new segmentation
  method or acoustic model, compare outputs against k-Wave reference
  solutions or the TFUScapes dataset before merging.
- **Disclose AI assistance.** Include `Co-Authored-By:` in commit messages
  when AI tools made substantive contributions to the code.
- **Do not commit AI-generated test data** as ground truth without
  independent verification. Synthetic phantoms are fine for regression tests,
  but clinical validation requires real data.
- **Docstrings and documentation** are excellent use cases for AI assistance.
  Ensure NumPy-style formatting and that parameter descriptions match the
  actual implementation.

## 7. Domain-Specific Notes

### Transcranial Focused Ultrasound (tFUS) Context

openlifu targets low-intensity tFUS for neuromodulation, where:

- Frequencies are typically 250 kHz -- 1 MHz.
- Focal pressures are on the order of 100 kPa -- 1 MPa (not HIFU ablation).
- The skull is the primary aberrator; heterogeneous skull modeling and phase
  correction are critical for accurate targeting.
- Safety metrics include Mechanical Index (MI < 1.9), Thermal Index Cranial
  (TIC), and spatial-peak temporal-average intensity (ISPTA).

### Label Sources and Remapping

Different neuroimaging pipelines produce tissue labels with different
conventions. The project uses a canonical mapping (0=water through 5=WM) and
provides remapping functions for each source:

| Source    | Function                    | Notes                          |
|-----------|-----------------------------|--------------------------------|
| SimNIBS   | `remap_simnibs_labels()`    | CHARM / headreco outputs       |
| Birnbaum  | (in `test_birnbaum.py`)     | 8-class labels incl. air       |
| SCI       | `sci_bridge.py`             | University of Utah FEM mesh    |
| Pseudo-CT | `_segment_pseudoct()`       | T1w intensity thresholding     |

When adding a new label source, create a `NEWSOURCE_TO_OPENLIFU` dict and a
`remap_newsource_labels()` function following the established pattern.

### Simulation Backends

| Backend   | Module              | Use Case                              |
|-----------|---------------------|---------------------------------------|
| k-Wave    | `sim/kwave_if.py`   | Time-domain pulsed FUS, gold standard |
| jwave     | `sim/jwave_if.py`   | Differentiable CW/time-domain via JAX |

The jwave backend supports automatic differentiation through the acoustic
simulation, enabling gradient-based phase correction. Use it when you need
`jax.grad` through the pressure field.
