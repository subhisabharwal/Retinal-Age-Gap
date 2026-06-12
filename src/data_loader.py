import os
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================
# 1. ODIR DATASET LOADER
# =========================================
def load_odir(source):
    df = pd.read_excel(source["csv_path"])

    rows = []

    for _, row in df.iterrows():
        pid = row[source["patient_id_col"]]
        age = row[source["age_col"]]

        left = row[source["left_col"]]
        right = row[source["right_col"]]

        # skip invalid age
        if pd.isna(age) or age <= 0 or age >= 100:
            continue

        # LEFT EYE
        if pd.notna(left):
            left_path = os.path.join(source["image_folder"], left.strip())
            if os.path.exists(left_path):
                rows.append({
                    "patient_id": pid,
                    "image_path": left_path,
                    "age": age
                })

        # RIGHT EYE
        if pd.notna(right):
            right_path = os.path.join(source["image_folder"], right.strip())
            if os.path.exists(right_path):
                rows.append({
                    "patient_id": pid,
                    "image_path": right_path,
                    "age": age
                })

    df = pd.DataFrame(rows)

    print(f"ODIR cleaned samples: {len(df)}")
    return df


# =========================================
# 2. NEW DATASET LOADER (CSV WITH SINGLE IMAGE ID)
# =========================================
def load_new_dataset(source):
    df = pd.read_csv(source["csv_path"])

    rows = []

    for _, row in df.iterrows():
        pid = row[source["patient_id_col"]]
        age = row[source["age_col"]]
        img_name = str(row[source["image_col"]]).strip()

        # skip invalid age or missing image id
        if pd.isna(age) or age <= 0 or age >= 100:
            continue
        if img_name == "" or img_name.lower() == "nan":
            continue

        if os.path.splitext(img_name)[1] == "":
            img_name = f"{img_name}.jpg"

        img_path = os.path.join(source["image_folder"], img_name)

        if os.path.exists(img_path):
            rows.append({
                "patient_id": pid,
                "image_path": img_path,
                "age": age
            })

    df = pd.DataFrame(rows)

    print(f"BRSET cleaned samples: {len(df)}")
    return df

# =========================================
# 3. COMBINE + SPLIT
# =========================================
def prepare_data(data_sources, test_size=0.2):
    all_dfs = []

    for source in data_sources:
        print(f"\nProcessing dataset: {source['type']}")

        if source["type"] == "odir":
            df = load_odir(source)

        elif source["type"] == "new":
            df = load_new_dataset(source)

        else:
            raise ValueError("Unknown dataset type")

        all_dfs.append(df)

    # COMBINE BOTH DATASETS
    combined_df = pd.concat(all_dfs, ignore_index=True)

    print("\nTotal combined samples:", len(combined_df))

    # SPLIT BY PATIENT (VERY IMPORTANT)
    patients = combined_df["patient_id"].unique()

    train_patients, val_patients = train_test_split(
        patients, test_size=test_size, random_state=42
    )

    train_df = combined_df[
        combined_df["patient_id"].isin(train_patients)
    ]

    val_df = combined_df[
        combined_df["patient_id"].isin(val_patients)
    ]

    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))

    return train_df, val_df
