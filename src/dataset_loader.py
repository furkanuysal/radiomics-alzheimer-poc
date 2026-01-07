import os
import re
import pandas as pd

DATA_ROOT = "data/raw/disc1"
CLINICAL_PATH = "data/clinical/oasis_cross-sectional.xlsx"  # .csv ise burada .csv yaz

def load_clinical_data(path: str) -> pd.DataFrame:
    """
    Loads OASIS clinical data from either .csv or .xlsx and normalizes column names.
    Returns a DataFrame with at least: ID, CDR, MMSE (if available).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported clinical file type: {ext}")

    # Normalize column names: strip spaces, unify
    df.columns = [str(c).strip() for c in df.columns]

    # Some files use different labels; map common variants
    col_map = {}
    for c in df.columns:
        c_norm = c.strip().lower()

        if c_norm in ("id", "subject", "subject_id", "subject id"):
            col_map[c] = "ID"
        elif c_norm in ("cdr", "clinical dementia rating"):
            col_map[c] = "CDR"
        elif c_norm in ("mmse", "mini-mental state examination"):
            col_map[c] = "MMSE"

    df = df.rename(columns=col_map)

    if "ID" not in df.columns:
        raise ValueError(f"'ID' column not found. Available columns: {list(df.columns)}")

    # Ensure ID formatting matches folder naming: OAS1_0001_MR1
    df["ID"] = df["ID"].astype(str).str.strip()

    return df


def get_subject_id_from_folder(folder_name: str) -> str:
    # Folder already looks like OAS1_0001_MR1
    return folder_name.strip()


def choose_image_path(raw_dir: str) -> str | None:
    """
    Pick one image per subject. Prefer mpr-1 .hdr.
    """
    hdrs = [f for f in os.listdir(raw_dir) if f.lower().endswith(".hdr")]
    if not hdrs:
        return None

    # Prefer mpr-1 if present
    mpr1 = [h for h in hdrs if re.search(r"mpr-1_.*\.hdr$", h, re.IGNORECASE)]
    chosen = mpr1[0] if mpr1 else sorted(hdrs)[0]
    return os.path.join(raw_dir, chosen)


def get_mri_subjects():
    subjects = []
    for subject_folder in os.listdir(DATA_ROOT):
        subject_path = os.path.join(DATA_ROOT, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        raw_dir = os.path.join(subject_path, "RAW")
        if not os.path.isdir(raw_dir):
            continue

        image_path = choose_image_path(raw_dir)
        if not image_path:
            continue

        subjects.append({
            "id": get_subject_id_from_folder(subject_folder),
            "image_path": image_path
        })

    return subjects


def build_dataset_index():
    clinical_df = load_clinical_data(CLINICAL_PATH)
    mri_subjects = get_mri_subjects()

    # Index clinical rows by ID for fast lookup
    clinical_index = clinical_df.set_index("ID", drop=False)

    dataset = []
    missing = 0

    for subj in mri_subjects:
        sid = subj["id"]

        if sid not in clinical_index.index:
            missing += 1
            continue

        row = clinical_index.loc[sid]

        dataset.append({
            "id": sid,
            "image_path": subj["image_path"],
            "cdr": row["CDR"] if "CDR" in clinical_index.columns else None,
            "mmse": row["MMSE"] if "MMSE" in clinical_index.columns else None
        })

    return dataset, len(mri_subjects), missing, list(clinical_df.columns)


if __name__ == "__main__":
    data, n_mri, missing, cols = build_dataset_index()
    print(f"Clinical columns: {cols}")
    print(f"Found MRI subjects in disc1: {n_mri}")
    print(f"Matched subjects: {len(data)}")
    print(f"Missing in clinical table: {missing}")
    print("Sample rows:")
    for item in data[:5]:
        print(item)
