# Radiomics-Based MRI Cognitive Analysis (Alzheimer PoC)

This project is an **educational / research-oriented proof-of-concept** demonstrating how
radiomics features extracted from structural brain MRI can be used to:

- Perform **binary cognitive status classification** (Healthy vs Alzheimer)
- **Estimate MMSE score** directly from MRI data
- Provide an **end-to-end pipeline** from MRI upload → analysis → interpretable output

> ⚠️ **Disclaimer**  
> This project is **not a clinical diagnostic tool** and must not be used for medical decision-making.

---

## What This Project Does

- Extracts **radiomics features** from the **right hippocampus ROI**
- Uses **pre-trained machine learning models** to:
  - Predict cognitive status (Healthy / Alzheimer)
  - Estimate MMSE score (with error margin)
- Provides:
  - **FastAPI backend**
  - **React + TypeScript frontend**
  - **Dockerized setup** for reproducible execution on other machines

---

## Technology Stack

### Backend & ML
- **Python:** 3.10
- **PyRadiomics**
- **SimpleITK**
- **NumPy / Pandas**
- **Scikit-learn**
- **Joblib**
- **FastAPI**
- **Uvicorn**

### Frontend
- **React**
- **TypeScript**
- **Vite**
- **Nginx**

### Deployment
- **Docker**
- **Docker Compose**

---

## Data & Model Artifacts (Important)

This repository does **not** include:

- Raw MRI data (`data/`)
- Extracted feature matrices
- Trained model files (`outputs/`)

Due to data size and licensing constraints, these artifacts must be
generated locally.

### Required steps before running inference:

1. Obtain the MRI dataset (e.g. OASIS-style structure)
2. Place data under the `data/` directory
3. Run feature extraction scripts
4. Train models to generate files under `outputs/`

Only after these steps, the backend inference API will function correctly.

---

## Supported MRI Formats

- `.img`
- `.hdr`

Both files must be provided together.

---

## Docker Compose Ports

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000

---

## Run with Docker

```bash
docker compose up --build
```

---

## API Endpoint

```
POST /analyze-mri
```

Multipart form-data:
- `img_file`
- `hdr_file`

---

## Example Response

```json
{
  "status": "ok",
  "results": {
    "diagnosis": {
      "prediction": "HEALTHY"
    },
    "mmse": {
      "predicted": 26.33,
      "cognitive_level": "MILD IMPAIRMENT",
      "error_note": "This value may vary by ± 2.9 points."
    }
  },
  "disclaimer": "This is not a clinical diagnosis."
}
```

---

## MMSE Interpretation

| Score | Level |
|------|------|
| ≥ 28 | Normal |
| 24–27 | Mild Impairment |
| 18–23 | Moderate Alzheimer |
| < 18 | Severe Alzheimer |

---

## Local Setup (Optional)

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install "numpy<2.0"
pip install -r requirements.txt
```
Installation Note (Important)

On some systems, PyRadiomics may require disabling build isolation during installation:
```bash
pip install pyradiomics --no-build-isolation
```
This is a known issue related to PyRadiomics build-time dependencies and Python packaging behavior.

---

## License

Educational / Research use only.
