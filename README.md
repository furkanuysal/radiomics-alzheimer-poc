## Technology Stack

- **Python:** 3.10 (chosen for compatibility and stability)
- **PyRadiomics:** Radiomics feature extraction
- **SimpleITK:** Medical image I/O and processing
- **NumPy / Pandas:** Numerical processing and tabular data
- **Scikit-learn:** Utility functions (no heavy ML usage)
- **Matplotlib:** Visualization

---

## Environment Setup

### Python Version
This project requires **Python 3.10**.

Newer Python versions (e.g. 3.11, 3.12) may cause compatibility issues with
PyRadiomics due to build and dependency constraints.

---

### Installation

Create and activate a virtual environment:

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

### Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

Installation Note (Important)

On some systems, PyRadiomics may require disabling build isolation during installation:
```bash
pip install pyradiomics --no-build-isolation
```
This is a known issue related to PyRadiomics build-time dependencies and Python packaging behavior.