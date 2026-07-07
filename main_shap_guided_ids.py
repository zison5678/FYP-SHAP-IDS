# ============================================================
# MULTICLASS IDS MODEL — SHAP-GUIDED FEATURE FINE-TUNING
# Updated from the uploaded base model code.
#
# What this version adds:
# 1. Keeps feature names throughout the pipeline.
# 2. Adds SHAP-informed feature sets based on the provided SHAP plots.
# 3. Adds validation-based hyperparameter fine-tuning using macro F1.
# 4. Saves summary, per-class results, confusion matrices, and selected features.
# ============================================================

# =========================
# IMPORTS
# =========================
import os
import warnings
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    make_scorer
)

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
FOLDER_PATH = r"D:\SCHOOL STUFF\project\project\dataset\archive"
OUTPUT_PATH = r"D:\SCHOOL STUFF\project\project\result_shap_finetuned"

RANDOM_STATE = 42

MIN_SAMPLES_PER_CLASS = 1000
MAX_SAMPLES_PER_CLASS = 100000

# Set to True if the run is too slow.
FAST_MODE = False

# RandomizedSearchCV settings
N_ITER_SEARCH = 20 if not FAST_MODE else 6
CV_FOLDS = 3

LEAKAGE_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Source Port",
    "Destination Port",
    "SimillarHTTP",
    "Inbound"
]


# ============================================================
# SHAP-INFORMED FEATURE LISTS FROM YOUR PROVIDED RESULTS
# ============================================================
# These are taken from the global and class-wise SHAP plots you shared.
# The code automatically removes any feature that is not present in the dataset.

GLOBAL_SHAP_TOP_20_FROM_PLOTS = [
    "Init_Win_bytes_backward",
    "min_seg_size_forward",
    "Bwd Packets/s",
    "Bwd Packet Length Min",
    "Init_Win_bytes_forward",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Bwd Packet Length Mean",
    "Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Std",
    "Avg Bwd Segment Size",
    "Max Packet Length",
    "Fwd IAT Mean",
    "Average Packet Size",
    "Flow IAT Max",
    "Flow IAT Std",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Packet Length Std"
]

CLASSWISE_SHAP_PRIOR_FROM_PLOTS = {
    "BENIGN": [
        "Init_Win_bytes_backward",
        "min_seg_size_forward",
        "Init_Win_bytes_forward",
        "Bwd Packet Length Min",
        "Bwd Packets/s",
        "Bwd Packet Length Std",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Bwd Packet Length Max",
        "Packet Length Mean"
    ],
    "Bot": [
        "Init_Win_bytes_backward",
        "min_seg_size_forward",
        "Bwd Packet Length Min",
        "Bwd Packets/s",
        "Init_Win_bytes_forward",
        "Flow IAT Mean",
        "Flow Packets/s",
        "Bwd IAT Min",
        "Average Packet Size",
        "Bwd Packet Length Mean"
    ],
    "DDoS": [
        "min_seg_size_forward",
        "Init_Win_bytes_backward",
        "Bwd Packet Length Max",
        "Bwd Packet Length Mean",
        "Fwd Packet Length Max",
        "Fwd Packet Length Std",
        "Bwd Packet Length Std",
        "Total Length of Bwd Packets",
        "Total Length of Fwd Packets",
        "Avg Bwd Segment Size",
        "Subflow Bwd Bytes"
    ],
    "DoS_GoldenEye": [
        "min_seg_size_forward",
        "Bwd Packet Length Max",
        "Bwd Packet Length Std",
        "Init_Win_bytes_backward",
        "Flow Packets/s",
        "Bwd Packets/s",
        "Flow IAT Mean",
        "Init_Win_bytes_forward",
        "Bwd Packet Length Mean",
        "Max Packet Length",
        "Avg Bwd Segment Size"
    ],
    "DoS_Hulk": [
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "Bwd Packet Length Std",
        "Avg Bwd Segment Size",
        "Bwd Packet Length Max",
        "Bwd Packet Length Mean",
        "Bwd Packets/s",
        "Max Packet Length",
        "Packet Length Mean",
        "Flow Packets/s",
        "Average Packet Size"
    ],
    "DoS_SlowHTTPTest": [
        "Bwd Packets/s",
        "Flow IAT Mean",
        "Flow Packets/s",
        "min_seg_size_forward",
        "Active Min",
        "Flow IAT Std",
        "Active Mean",
        "Fwd IAT Mean",
        "Init_Win_bytes_backward",
        "Fwd IAT Max",
        "Fwd IAT Std"
    ],
    "DoS_Slowloris": [
        "Init_Win_bytes_backward",
        "Flow Packets/s",
        "min_seg_size_forward",
        "Flow IAT Mean",
        "Bwd Packets/s",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Fwd IAT Mean",
        "Avg Bwd Segment Size",
        "Fwd IAT Max",
        "Flow IAT Max",
        "Flow IAT Std"
    ],
    "FTP": [
        "Init_Win_bytes_backward",
        "min_seg_size_forward",
        "Flow Packets/s",
        "Fwd Packet Length Std",
        "Flow IAT Mean",
        "Average Packet Size",
        "Packet Length Mean",
        "Bwd Packet Length Min",
        "Packet Length Std",
        "Fwd Packet Length Mean",
        "Packet Length Variance"
    ],
    "PortScan": [
        "Bwd Packet Length Min",
        "Init_Win_bytes_backward",
        "PSH Flag Count",
        "Packet Length Mean",
        "Bwd Packets/s",
        "Fwd Packet Length Mean",
        "Flow Packets/s",
        "Total Length of Fwd Packets",
        "Init_Win_bytes_forward",
        "Flow IAT Mean",
        "Subflow Fwd Bytes"
    ],
    "SSH": [
        "Init_Win_bytes_backward",
        "Init_Win_bytes_forward",
        "min_seg_size_forward",
        "Bwd Packets/s",
        "Bwd Packet Length Min",
        "Flow IAT Mean",
        "Flow Packets/s",
        "Bwd Header Length",
        "Flow Bytes/s",
        "Flow IAT Max",
        "Fwd IAT Mean"
    ],
    "WebAttack_BruteForce": [
        "Init_Win_bytes_backward",
        "Bwd Packets/s",
        "Fwd IAT Min",
        "Flow IAT Mean",
        "Flow Packets/s",
        "Init_Win_bytes_forward",
        "min_seg_size_forward",
        "Flow IAT Max",
        "Fwd IAT Mean",
        "Fwd Header Length",
        "Flow IAT Std"
    ]
}


# =========================
# UTILS
# =========================
def ensure_output_path():
    os.makedirs(OUTPUT_PATH, exist_ok=True)


def save_csv(df, filename):
    ensure_output_path()
    path = os.path.join(OUTPUT_PATH, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_json(obj, filename):
    ensure_output_path()
    path = os.path.join(OUTPUT_PATH, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)
    print(f"Saved: {path}")


def available_features(feature_list, X_columns):
    """Keep only features that exist in the loaded dataset."""
    return [f for f in feature_list if f in X_columns]


def unique_keep_order(items):
    return list(dict.fromkeys(items))


# =========================
# LOAD DATA
# =========================
def load_data():
    dfs = []

    for file in os.listdir(FOLDER_PATH):
        if file.endswith(".csv"):
            path = os.path.join(FOLDER_PATH, file)
            print("Loading:", file)
            df = pd.read_csv(path, encoding="latin1", low_memory=False)
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No CSV files found.")

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()

    print("\nDataset shape:", df.shape)
    return df


# =========================
# LABEL FIX
# =========================
def fix_label_column(df):
    if "Label" in df.columns:
        return df

    for col in ["label", "Attack", "attack", "Class", "class"]:
        if col in df.columns:
            df = df.rename(columns={col: "Label"})
            return df

    raise ValueError("No label column found.")


# =========================
# CLEAN DATA
# =========================
def clean_data(df):
    print("\nCleaning data...")
    before = len(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print("Rows removed:", before - len(df))
    return df


# =========================
# STANDARDISE LABELS
# =========================
def standardize_labels(df):
    print("\nStandardising labels...")

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("Ï¿½", "-", regex=False)
        .str.replace("ï¿½", "-", regex=False)
        .str.replace("â€“", "-", regex=False)
        .str.replace("–", "-", regex=False)
        .str.replace("—", "-", regex=False)
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    def map_label(label):
        label = label.strip().upper()

        if label == "BENIGN":
            return "BENIGN"
        if label == "DDOS":
            return "DDoS"
        if label == "DOS HULK":
            return "DoS_Hulk"
        if label == "DOS GOLDENEYE":
            return "DoS_GoldenEye"
        if label == "DOS SLOWLORIS":
            return "DoS_Slowloris"
        if label == "DOS SLOWHTTPTEST":
            return "DoS_SlowHTTPTest"
        if label in ["PORTSCAN", "PORT SCAN"]:
            return "PortScan"
        if label in ["BOT", "BOTNET"]:
            return "Bot"
        if label in ["FTP-PATATOR", "FTP PATATOR", "FTP"]:
            return "FTP"
        if label in ["SSH-PATATOR", "SSH PATATOR", "SSH"]:
            return "SSH"
        if label == "INFILTRATION":
            return "Infiltration"
        if label == "HEARTBLEED":
            return "Heartbleed"
        if "WEB ATTACK" in label and "BRUTE FORCE" in label:
            return "WebAttack_BruteForce"
        if "WEB ATTACK" in label and "SQL" in label:
            return "WebAttack_SQLInjection"
        if "WEB ATTACK" in label and "XSS" in label:
            return "WebAttack_XSS"

        return label

    df["Label"] = df["Label"].apply(map_label)

    print("\nLabels after cleaning:")
    print(df["Label"].value_counts())

    return df


# =========================
# DROP LEAKAGE
# =========================
def remove_leakage(df):
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    print("\nDropping leakage columns:", drop_cols)
    return df.drop(columns=drop_cols, errors="ignore")


# =========================
# FILTER CLASSES
# =========================
def filter_classes(df):
    counts = df["Label"].value_counts()
    valid = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    removed = counts[counts < MIN_SAMPLES_PER_CLASS]

    print("\nRemoved rare classes:")
    print(removed)

    df = df[df["Label"].isin(valid)].copy()

    if df["Label"].nunique() < 2:
        raise ValueError("Not enough classes after filtering.")

    print("\nDistribution after filtering:")
    print(df["Label"].value_counts())

    return df


# =========================
# CAP LARGE CLASSES
# =========================
def cap_classes(df):
    print(f"\nCapping classes to maximum {MAX_SAMPLES_PER_CLASS} samples per class...")

    out = []

    for label, group in df.groupby("Label"):
        if len(group) > MAX_SAMPLES_PER_CLASS:
            group = group.sample(MAX_SAMPLES_PER_CLASS, random_state=RANDOM_STATE)
        out.append(group)

    df = pd.concat(out).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nDistribution after capping:")
    print(df["Label"].value_counts())

    return df


# =========================
# FEATURES
# =========================
def prepare_features(df):
    y = df["Label"]
    X = df.drop(columns=["Label"], errors="ignore")

    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if X.shape[1] == 0:
        raise ValueError("No numeric features found.")

    print("\nNumber of numeric features:", X.shape[1])
    return X, y


# =========================
# SPLIT
# =========================
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    print("\nTrain distribution:")
    print(y_train.value_counts())

    print("\nValidation distribution:")
    print(y_val.value_counts())

    print("\nTest distribution:")
    print(y_test.value_counts())

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================
# ENCODE LABELS
# =========================
def encode_labels(y_train, y_val, y_test):
    le = LabelEncoder()

    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    print("\nClass mapping:")
    for idx, cls in enumerate(le.classes_):
        print(f"{idx}: {cls}")

    return y_train_enc, y_val_enc, y_test_enc, le


# ============================================================
# SHAP-GUIDED FEATURE SETS
# ============================================================
def build_shap_guided_feature_sets(X_columns, class_names):
    """
    Build several feature sets from the SHAP plots.
    The goal is not to blindly use only global SHAP, but to preserve
    minority/weak-class-specific signals.
    """
    feature_sets = {}

    # 1. Original baseline
    feature_sets["All_Features"] = list(X_columns)

    # 2. Global SHAP top features from the plot
    global_top20 = available_features(GLOBAL_SHAP_TOP_20_FROM_PLOTS, X_columns)
    feature_sets["SHAP_Global_Top20_From_Plots"] = global_top20

    # 3. Class-wise union using only classes that exist after filtering
    classwise_union = []

    for cls in class_names:
        if cls in CLASSWISE_SHAP_PRIOR_FROM_PLOTS:
            classwise_union.extend(CLASSWISE_SHAP_PRIOR_FROM_PLOTS[cls])

    classwise_union = available_features(unique_keep_order(classwise_union), X_columns)
    feature_sets["SHAP_Classwise_Union_From_Plots"] = classwise_union

    # 4. Hybrid: global top 20 + class-wise union
    hybrid = unique_keep_order(global_top20 + classwise_union)
    feature_sets["SHAP_Hybrid_Global20_ClasswiseUnion"] = available_features(hybrid, X_columns)

    # 5. Compact hybrid: prioritise the repeated features that appear across many class-wise plots
    repeated_priority = [
        "Init_Win_bytes_backward",
        "min_seg_size_forward",
        "Bwd Packets/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Max",
        "Bwd Packet Length Std",
        "Init_Win_bytes_forward",
        "Average Packet Size",
        "Packet Length Mean",
        "Packet Length Std",
        "Max Packet Length",
        "Avg Bwd Segment Size",
        "Fwd IAT Mean",
        "Fwd IAT Max",
        "Fwd IAT Std",
        "Fwd IAT Min",
        "Active Min",
        "Active Mean",
        "PSH Flag Count",
        "Fwd Header Length",
        "Bwd Header Length",
        "Flow Bytes/s"
    ]
    feature_sets["SHAP_Compact_Repeated_Priority"] = available_features(repeated_priority, X_columns)

    # Remove empty feature sets
    feature_sets = {k: v for k, v in feature_sets.items() if len(v) > 0}

    # Save selected features
    selected_rows = []
    for strategy, features in feature_sets.items():
        for rank, feat in enumerate(features, start=1):
            selected_rows.append({
                "Feature Strategy": strategy,
                "Rank": rank,
                "Feature": feat
            })

    save_csv(pd.DataFrame(selected_rows), "selected_shap_guided_feature_sets.csv")

    print("\nFeature sets prepared:")
    for name, features in feature_sets.items():
        print(f"{name}: {len(features)} features")

    return feature_sets


# ============================================================
# MODELS AND SEARCH SPACES
# ============================================================
def get_model_search_spaces():
    """
    Pipelines are used so that scaling is fitted inside each CV fold.
    This avoids leakage from validation/test data into training.
    """
    spaces = {}

    spaces["Logistic Regression"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="saga",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "params": {
            "model__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
            "model__penalty": ["l1", "l2"]
        }
    }

    spaces["Linear SVC (SGD)"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SGDClassifier(
                loss="hinge",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "params": {
            "model__alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
            "model__max_iter": [1000, 2000, 3000],
            "model__tol": [1e-3, 1e-4]
        }
    }

    spaces["Random Forest"] = {
        "pipeline": Pipeline([
            ("model", RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "params": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None]
        }
    }

    spaces["Extra Trees"] = {
        "pipeline": Pipeline([
            ("model", ExtraTreesClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "params": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None]
        }
    }

    spaces["KNN"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }
    }

    return spaces


# ============================================================
# EVALUATION
# ============================================================
def evaluate_predictions(y_true, y_pred, le, dataset_name, feature_strategy, model_name):
    acc = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    worst_class_f1 = np.min(per_class_f1)
    worst_class_recall = np.min(per_class_recall)

    print(f"\n{dataset_name} results:")
    print("Accuracy:", round(acc, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Worst-class F1:", round(worst_class_f1, 4))
    print("Worst-class Recall:", round(worst_class_recall, 4))
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    per_class_df = pd.DataFrame({
        "Feature Strategy": feature_strategy,
        "Model": model_name,
        "Dataset": dataset_name,
        "Class": le.classes_,
        "Precision": per_class_precision,
        "Recall": per_class_recall,
        "F1": per_class_f1
    })

    summary = {
        "Feature Strategy": feature_strategy,
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": acc,
        "Weighted F1": weighted_f1,
        "Macro F1": macro_f1,
        "Worst-class F1": worst_class_f1,
        "Worst-class Recall": worst_class_recall
    }

    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.insert(0, "Actual Class", le.classes_)

    return summary, per_class_df, cm_df


# ============================================================
# TRAIN + FINE-TUNE
# ============================================================
def tune_and_evaluate_feature_set(
    strategy_name,
    selected_features,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    le
):
    print("\n" + "=" * 100)
    print("FEATURE STRATEGY:", strategy_name)
    print("NO. FEATURES:", len(selected_features))
    print("=" * 100)

    X_train_s = X_train[selected_features]
    X_val_s = X_val[selected_features]
    X_test_s = X_test[selected_features]

    model_spaces = get_model_search_spaces()
    macro_scorer = make_scorer(f1_score, average="macro", zero_division=0)

    summary_rows = []
    per_class_rows = []
    best_params_rows = []

    for model_name, config in model_spaces.items():
        print("\n" + "-" * 100)
        print("Tuning:", model_name)
        print("-" * 100)

        search = RandomizedSearchCV(
            estimator=config["pipeline"],
            param_distributions=config["params"],
            n_iter=N_ITER_SEARCH,
            scoring=macro_scorer,
            cv=CV_FOLDS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train_s.astype(np.float32), y_train)

        best_model = search.best_estimator_

        print("Best CV macro F1:", round(search.best_score_, 4))
        print("Best params:", search.best_params_)

        best_params_rows.append({
            "Feature Strategy": strategy_name,
            "Model": model_name,
            "Best CV Macro F1": search.best_score_,
            "Best Params": json.dumps(search.best_params_)
        })

        for dataset_name, X_data, y_data in [
            ("Validation", X_val_s, y_val),
            ("Test", X_test_s, y_test)
        ]:
            pred = best_model.predict(X_data.astype(np.float32))

            summary, per_class_df, cm_df = evaluate_predictions(
                y_true=y_data,
                y_pred=pred,
                le=le,
                dataset_name=dataset_name,
                feature_strategy=strategy_name,
                model_name=model_name
            )

            summary["No. Features"] = len(selected_features)
            summary["Selected Features"] = ", ".join(selected_features)
            summary["Best CV Macro F1"] = search.best_score_
            summary["Best Params"] = json.dumps(search.best_params_)

            summary_rows.append(summary)
            per_class_rows.append(per_class_df)

            safe_strategy = strategy_name.replace(" ", "_").replace("/", "_")
            safe_model = model_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            save_csv(cm_df, f"confusion_matrix_{safe_strategy}_{safe_model}_{dataset_name}.csv")

    return (
        pd.DataFrame(summary_rows),
        pd.concat(per_class_rows, ignore_index=True),
        pd.DataFrame(best_params_rows)
    )


# =========================
# MAIN
# =========================
def main():
    ensure_output_path()

    # --------------------------------------------------------
    # Load and preprocess
    # --------------------------------------------------------
    df = load_data()
    df = fix_label_column(df)
    df = clean_data(df)
    df = standardize_labels(df)

    df = remove_leakage(df)
    df = filter_classes(df)
    df = cap_classes(df)

    X, y = prepare_features(df)

    X_train, X_val, X_test, y_train_str, y_val_str, y_test_str = split_data(X, y)

    y_train, y_val, y_test, le = encode_labels(y_train_str, y_val_str, y_test_str)

    # --------------------------------------------------------
    # Build SHAP-guided feature sets
    # --------------------------------------------------------
    feature_sets = build_shap_guided_feature_sets(
        X_columns=X_train.columns.tolist(),
        class_names=list(le.classes_)
    )

    save_json(feature_sets, "selected_shap_guided_feature_sets.json")

    # --------------------------------------------------------
    # Fine-tune each model for each feature set
    # --------------------------------------------------------
    all_summary = []
    all_per_class = []
    all_best_params = []

    for strategy_name, selected_features in feature_sets.items():
        summary_df, per_class_df, best_params_df = tune_and_evaluate_feature_set(
            strategy_name=strategy_name,
            selected_features=selected_features,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            le=le
        )

        all_summary.append(summary_df)
        all_per_class.append(per_class_df)
        all_best_params.append(best_params_df)

        # Save partial progress after every strategy
        partial_summary = pd.concat(all_summary, ignore_index=True)
        partial_per_class = pd.concat(all_per_class, ignore_index=True)
        partial_best_params = pd.concat(all_best_params, ignore_index=True)

        save_csv(partial_summary, "partial_finetuning_summary.csv")
        save_csv(partial_per_class, "partial_finetuning_per_class.csv")
        save_csv(partial_best_params, "partial_best_params.csv")

    final_summary = pd.concat(all_summary, ignore_index=True)
    final_per_class = pd.concat(all_per_class, ignore_index=True)
    final_best_params = pd.concat(all_best_params, ignore_index=True)

    save_csv(final_summary, "final_shap_guided_finetuning_summary.csv")
    save_csv(final_per_class, "final_shap_guided_finetuning_per_class.csv")
    save_csv(final_best_params, "final_shap_guided_best_params.csv")

    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------
    test_summary = final_summary[final_summary["Dataset"] == "Test"].copy()

    print("\n" + "=" * 100)
    print("FINAL TEST RESULTS SORTED BY MACRO F1")
    print("=" * 100)
    print(
        test_summary.sort_values(
            by=["Macro F1", "Worst-class F1", "Accuracy"],
            ascending=False
        )[
            [
                "Feature Strategy",
                "Model",
                "No. Features",
                "Accuracy",
                "Weighted F1",
                "Macro F1",
                "Worst-class F1",
                "Worst-class Recall",
                "Best CV Macro F1"
            ]
        ]
    )

    print("\n" + "=" * 100)
    print("FINAL TEST RESULTS SORTED BY WORST-CLASS F1")
    print("=" * 100)
    print(
        test_summary.sort_values(
            by=["Worst-class F1", "Macro F1", "Accuracy"],
            ascending=False
        )[
            [
                "Feature Strategy",
                "Model",
                "No. Features",
                "Accuracy",
                "Weighted F1",
                "Macro F1",
                "Worst-class F1",
                "Worst-class Recall",
                "Best CV Macro F1"
            ]
        ]
    )

    print(f"\nAll outputs saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
