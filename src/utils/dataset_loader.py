import os
import pandas as pd

DATA_ROOTS = [
    "data/raw/disc1",
    "data/raw/disc2",
    "data/raw/disc3",
    "data/raw/disc4",
    "data/raw/disc5"
]

CLINICAL_PATH = "data/clinical/oasis_cross-sectional.xlsx"


def load_clinical_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported clinical file type: {ext}")

    df.columns = [str(c).strip() for c in df.columns]

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

    df["ID"] = df["ID"].astype(str).str.strip()
    return df


def get_subject_id_from_folder(folder_name: str) -> str:
    return folder_name.strip()


def find_processed_image(subject_dir: str) -> str | None:
    processed_dir = os.path.join(subject_dir, "PROCESSED")

    if not os.path.isdir(processed_dir):
        return None

    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith("masked_gfc.img"):
                return os.path.join(root, file)

    return None


def get_mri_subjects():
    subjects = []

    for data_root in DATA_ROOTS:
        if not os.path.exists(data_root):
            print(f"WARNING: {data_root} folder not found!")
            continue

        for subject_folder in os.listdir(data_root):
            subject_path = os.path.join(data_root, subject_folder)

            if not os.path.isdir(subject_path):
                continue

            image_path = find_processed_image(subject_path)

            if not image_path:
                continue

            subjects.append({
                "id": get_subject_id_from_folder(subject_folder),
                "image_path": image_path
            })

    return subjects


def build_dataset_index():
    print("Loading clinical data...")
    clinical_df = load_clinical_data(CLINICAL_PATH)

    print("Scanning MRI directories for 'masked_gfc' images...")
    mri_subjects = get_mri_subjects()

    clinical_index = clinical_df.set_index("ID", drop=False)

    dataset = []
    missing_clinical = 0

    for subj in mri_subjects:
        sid = subj["id"]

        if sid not in clinical_index.index:
            missing_clinical += 1
            continue

        row = clinical_index.loc[sid]

        dataset.append({
            "id": sid,
            "image_path": subj["image_path"],
            "cdr": row["CDR"] if "CDR" in clinical_index.columns else None,
            "mmse": row["MMSE"] if "MMSE" in clinical_index.columns else None
        })

    return dataset, len(mri_subjects), missing_clinical, list(clinical_df.columns)


if __name__ == "__main__":
    data, n_mri, missing, cols = build_dataset_index()
    print(f"Clinical columns: {cols}")
    print(f"Found MRI subjects (with masked images): {n_mri}")
    print(f"Matched subjects (MRI + Clinical): {len(data)}")
    print(f"Missing in clinical table: {missing}")

    if len(data) > 0:
        print("\nSample rows:")
        for item in data[:3]:
            print(item)
    else:
        print("\nNo matching data found! Check paths.")
