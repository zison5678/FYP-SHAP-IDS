# ============================================================
# MULTICLASS IDS:
# ALL FEATURES + GLOBAL SHAP + CLASS-WISE SHAP + WEAK-CLASS SHAP
# ============================================================

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import shap


# ============================================================
# CONFIGURATION
# ============================================================

FOLDER_PATH = r"D:\SCHOOL STUFF\project\project\dataset\archive"
OUTPUT_PATH = r"D:\SCHOOL STUFF\project\project\result"

RANDOM_STATE = 42

MIN_SAMPLES_PER_CLASS = 1000
MAX_SAMPLES_PER_CLASS = 100000

SHAP_SAMPLES_PER_CLASS = 300

CLASSWISE_TOP_N = 10
WEAK_CLASS_TOP_N = 20
WEAK_CLASS_F1_THRESHOLD = 0.95

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

warnings.filterwarnings("ignore")

# ============================================================
# HELPER: Save to OUTPUT_PATH
# ============================================================

def save_csv(df, filename):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    path = os.path.join(OUTPUT_PATH, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_fig(filename):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    path = os.path.join(OUTPUT_PATH, filename)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved figure: {path}")


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_data():
    dfs = []

    for file in os.listdir(FOLDER_PATH):
        if file.endswith(".csv"):
            print(f"Loading: {file}")
            path = os.path.join(FOLDER_PATH, file)

            df = pd.read_csv(
                path,
                encoding="latin1",
                low_memory=False
            )

            dfs.append(df)

    if len(dfs) == 0:
        raise FileNotFoundError("No CSV files found in the dataset folder.")

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()

    print("\nDataset shape:", df.shape)

    return df


def fix_label_column(df):
    if "Label" in df.columns:
        return df

    possible_label_columns = ["label", "Attack", "attack", "Class", "class"]

    for col in possible_label_columns:
        if col in df.columns:
            df = df.rename(columns={col: "Label"})
            return df

    raise ValueError("No label column found.")


def clean_data(df):
    print("\nCleaning data...")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(df)
    df.dropna(inplace=True)
    after = len(df)

    print("Rows removed:", before - after)

    return df


def standardize_labels(df):
    print("\nStandardizing labels...")

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Label"] = (
        df["Label"]
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

    print("\nUnique labels after cleaning:")
    for label in sorted(df["Label"].unique()):
        print(" -", label)

    return df


def show_distribution(df, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    counts = df["Label"].value_counts()
    percentages = (df["Label"].value_counts(normalize=True) * 100).round(4)

    distribution = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    print(distribution)


def remove_leakage_columns(df):
    drop_cols = [col for col in LEAKAGE_COLUMNS if col in df.columns]

    print("\nDropping leakage columns:")
    print(drop_cols)

    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def filter_rare_classes(df):
    print("\nFiltering rare classes...")

    class_counts = df["Label"].value_counts()

    rare_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS]
    valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index

    print("\nRemoved classes with fewer than", MIN_SAMPLES_PER_CLASS, "samples:")
    print(rare_classes)

    df = df[df["Label"].isin(valid_classes)].copy()

    show_distribution(df, "DISTRIBUTION AFTER REMOVING RARE CLASSES")

    return df


def cap_large_classes(df):
    print(f"\nCapping large classes to maximum {MAX_SAMPLES_PER_CLASS} samples per class...")

    sampled_dfs = []

    for label, group in df.groupby("Label"):
        if len(group) > MAX_SAMPLES_PER_CLASS:
            group = group.sample(
                n=MAX_SAMPLES_PER_CLASS,
                random_state=RANDOM_STATE
            )

        sampled_dfs.append(group)

    df_sampled = pd.concat(sampled_dfs, ignore_index=True)

    df_sampled = df_sampled.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    show_distribution(df_sampled, "DISTRIBUTION AFTER CLASS CAPPING")

    return df_sampled


def prepare_features(df):
    print("\nPreparing features...")

    y = df["Label"]

    X = df.drop(columns=["Label"], errors="ignore")

    X = X.select_dtypes(include=[np.number])

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    print("Number of numeric features:", X.shape[1])

    return X, y


def split_data(X, y):
    print("\nSplitting data into train, validation, and test sets...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
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


def encode_labels(y_train, y_val, y_test):
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    print("\nClass mapping:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"{i}: {cls}")

    return y_train_encoded, y_val_encoded, y_test_encoded, label_encoder


# ============================================================
# MODELS
# ============================================================

def get_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "Linear SVC": LinearSVC(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        )
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y, label_encoder, dataset_name):
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)

    weighted_f1 = f1_score(
        y,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y,
        y_pred,
        average="macro",
        zero_division=0
    )

    per_class_precision = precision_score(
        y,
        y_pred,
        average=None,
        zero_division=0
    )

    per_class_recall = recall_score(
        y,
        y_pred,
        average=None,
        zero_division=0
    )

    per_class_f1 = f1_score(
        y,
        y_pred,
        average=None,
        zero_division=0
    )

    worst_class_f1 = np.min(per_class_f1)
    worst_class_recall = np.min(per_class_recall)

    print(f"\n{dataset_name} Accuracy:           {accuracy:.4f}")
    print(f"{dataset_name} Weighted F1:        {weighted_f1:.4f}")
    print(f"{dataset_name} Macro F1:           {macro_f1:.4f}")
    print(f"{dataset_name} Worst-class F1:     {worst_class_f1:.4f}")
    print(f"{dataset_name} Worst-class Recall: {worst_class_recall:.4f}")

    print(f"\n{dataset_name} Classification Report:")
    print(
        classification_report(
            y,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    print(f"\n{dataset_name} Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    per_class_df = pd.DataFrame({
        "Class": label_encoder.classes_,
        f"{dataset_name}_Precision": per_class_precision,
        f"{dataset_name}_Recall": per_class_recall,
        f"{dataset_name}_F1": per_class_f1
    })

    metrics = {
        "Accuracy": accuracy,
        "Weighted F1": weighted_f1,
        "Macro F1": macro_f1,
        "Worst-class F1": worst_class_f1,
        "Worst-class Recall": worst_class_recall
    }

    return metrics, per_class_df


def train_and_evaluate_feature_set(
    feature_strategy_name,
    selected_features,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    label_encoder
):
    print("\n" + "=" * 100)
    print("FEATURE STRATEGY:", feature_strategy_name)
    print("NUMBER OF FEATURES:", len(selected_features))
    print("=" * 100)

    print("\nSelected features:")
    for feature in selected_features:
        print(" -", feature)

    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_selected.astype(np.float32))
    X_val_scaled = scaler.transform(X_val_selected.astype(np.float32))
    X_test_scaled = scaler.transform(X_test_selected.astype(np.float32))

    models = get_models()

    summary_rows = []
    per_class_rows = []
    trained_models = {}

    for model_name, model in models.items():
        print("\n" + "-" * 100)
        print("MODEL:", model_name)
        print("FEATURE STRATEGY:", feature_strategy_name)
        print("-" * 100)

        try:
            model.fit(X_train_scaled, y_train)

            val_metrics, val_per_class = evaluate_model(
                model,
                X_val_scaled,
                y_val,
                label_encoder,
                "Validation"
            )

            test_metrics, test_per_class = evaluate_model(
                model,
                X_test_scaled,
                y_test,
                label_encoder,
                "Test"
            )

            summary_rows.append({
                "Feature Strategy": feature_strategy_name,
                "No. Features": len(selected_features),
                "Model": model_name,
                "Validation Accuracy": val_metrics["Accuracy"],
                "Validation Weighted F1": val_metrics["Weighted F1"],
                "Validation Macro F1": val_metrics["Macro F1"],
                "Validation Worst-class F1": val_metrics["Worst-class F1"],
                "Validation Worst-class Recall": val_metrics["Worst-class Recall"],
                "Test Accuracy": test_metrics["Accuracy"],
                "Test Weighted F1": test_metrics["Weighted F1"],
                "Test Macro F1": test_metrics["Macro F1"],
                "Test Worst-class F1": test_metrics["Worst-class F1"],
                "Test Worst-class Recall": test_metrics["Worst-class Recall"],
                "Selected Features": ", ".join(selected_features)
            })

            val_per_class["Feature Strategy"] = feature_strategy_name
            val_per_class["No. Features"] = len(selected_features)
            val_per_class["Model"] = model_name
            val_per_class["Dataset"] = "Validation"

            test_per_class["Feature Strategy"] = feature_strategy_name
            test_per_class["No. Features"] = len(selected_features)
            test_per_class["Model"] = model_name
            test_per_class["Dataset"] = "Test"

            per_class_rows.append(val_per_class)
            per_class_rows.append(test_per_class)

            trained_models[model_name] = {
                "model": model,
                "scaler": scaler,
                "features": selected_features
            }

        except Exception as e:
            print(model_name, "failed.")
            print("Error:", e)

    summary_df = pd.DataFrame(summary_rows)

    if len(per_class_rows) > 0:
        per_class_df = pd.concat(per_class_rows, ignore_index=True)
    else:
        per_class_df = pd.DataFrame()

    return summary_df, per_class_df, trained_models


# ============================================================
# SHAP SUPPORT FUNCTIONS
# ============================================================

def create_balanced_shap_sample(X_train, y_train_str):
    print("\nCreating balanced SHAP sample...")

    temp_df = X_train.copy()
    temp_df["Label"] = y_train_str.values

    sampled_dfs = []

    for label, group in temp_df.groupby("Label"):
        sample_size = min(SHAP_SAMPLES_PER_CLASS, len(group))

        sampled_group = group.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )

        sampled_dfs.append(sampled_group)

    shap_df = pd.concat(sampled_dfs, ignore_index=True)

    shap_df = shap_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    y_shap = shap_df["Label"]
    X_shap = shap_df.drop(columns=["Label"])

    print("SHAP sample shape:", X_shap.shape)

    return X_shap, y_shap


def train_rf_for_shap(X_train, y_train, selected_features):
    print("\nTraining Random Forest for SHAP explanation...")

    X_train_selected = X_train[selected_features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected.astype(np.float32))

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    rf_model.fit(X_train_scaled, y_train)

    return rf_model, scaler


def compute_global_and_classwise_shap(
    rf_model,
    scaler,
    X_shap,
    selected_features,
    label_encoder
):
    print("\nComputing SHAP values...")

    X_shap_selected = X_shap[selected_features]
    X_shap_scaled = scaler.transform(X_shap_selected.astype(np.float32))

    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(
        X_shap_scaled,
        check_additivity=False
    )

    class_names = list(label_encoder.classes_)

    class_shap_dict = {}

    if isinstance(shap_values, list):
        for class_index, class_name in enumerate(class_names):
            class_shap_dict[class_name] = shap_values[class_index]

    else:
        shap_values = np.array(shap_values)

        if shap_values.ndim == 3:
            if shap_values.shape[2] == len(class_names):
                for class_index, class_name in enumerate(class_names):
                    class_shap_dict[class_name] = shap_values[:, :, class_index]

            elif shap_values.shape[1] == len(class_names):
                for class_index, class_name in enumerate(class_names):
                    class_shap_dict[class_name] = shap_values[:, class_index, :]

            else:
                raise ValueError("Could not interpret SHAP shape.")

        else:
            raise ValueError("Expected multiclass SHAP values.")

    # ----------------------------------------------------------
    # Global SHAP importance
    # ----------------------------------------------------------
    global_importance = np.zeros(len(selected_features))

    for class_name in class_names:
        global_importance += np.abs(class_shap_dict[class_name]).mean(axis=0)

    global_importance = global_importance / len(class_names)

    global_importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Global_SHAP_Importance": global_importance
    }).sort_values(
        by="Global_SHAP_Importance",
        ascending=False
    ).reset_index(drop=True)

    # ----------------------------------------------------------
    # Class-wise SHAP importance
    # ----------------------------------------------------------
    classwise_rows = []

    for class_name in class_names:
        class_importance = np.abs(class_shap_dict[class_name]).mean(axis=0)

        temp = pd.DataFrame({
            "Class": class_name,
            "Feature": selected_features,
            "Class_SHAP_Importance": class_importance
        }).sort_values(
            by="Class_SHAP_Importance",
            ascending=False
        ).reset_index(drop=True)

        temp["Rank_Within_Class"] = temp.index + 1

        classwise_rows.append(temp)

    classwise_importance_df = pd.concat(classwise_rows, ignore_index=True)

    print("\nTop 20 global SHAP features:")
    print(global_importance_df.head(20))

    print("\nTop 10 class-wise SHAP features per class:")
    for class_name in class_names:
        print("\nClass:", class_name)
        print(
            classwise_importance_df[
                classwise_importance_df["Class"] == class_name
            ].head(10)[["Rank_Within_Class", "Feature", "Class_SHAP_Importance"]]
        )

    # ----------------------------------------------------------
    # Save CSVs to OUTPUT_PATH
    # ----------------------------------------------------------
    save_csv(global_importance_df, "global_shap_importance.csv")
    save_csv(classwise_importance_df, "classwise_shap_importance.csv")

    # ----------------------------------------------------------
    # SHAP GRAPHS
    # ----------------------------------------------------------
    plot_global_shap_bar(global_importance_df)
    plot_classwise_shap_bars(classwise_importance_df, class_names)
    plot_shap_summary(
        class_shap_dict,
        class_names,
        X_shap_scaled,
        selected_features
    )

    return global_importance_df, classwise_importance_df


# ============================================================
# SHAP GRAPH FUNCTIONS
# ============================================================

def plot_global_shap_bar(global_importance_df, top_n=20):
    """Bar chart of top-N global SHAP feature importances."""
    df_plot = global_importance_df.head(top_n).sort_values(
        "Global_SHAP_Importance", ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    ax.barh(df_plot["Feature"], df_plot["Global_SHAP_Importance"], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Global SHAP Feature Importance (Top {top_n})")
    plt.tight_layout()
    save_fig("shap_global_bar.png")


def plot_classwise_shap_bars(classwise_importance_df, class_names, top_n=10):
    """One horizontal bar chart per class, saved as a single multi-panel figure."""
    n_classes = len(class_names)
    ncols = 2
    nrows = (n_classes + 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, nrows * 4)
    )
    axes = axes.flatten()

    for i, class_name in enumerate(class_names):
        df_cls = classwise_importance_df[
            classwise_importance_df["Class"] == class_name
        ].head(top_n).sort_values("Class_SHAP_Importance", ascending=True)

        axes[i].barh(
            df_cls["Feature"],
            df_cls["Class_SHAP_Importance"],
            color="#ff7f0e"
        )
        axes[i].set_title(f"{class_name} — Top {top_n} Features")
        axes[i].set_xlabel("Mean |SHAP Value|")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Class-wise SHAP Feature Importance", fontsize=14, y=1.01)
    plt.tight_layout()
    save_fig("shap_classwise_bars.png")


def plot_shap_summary(class_shap_dict, class_names, X_shap_scaled, selected_features, top_n=15):
    """
    SHAP beeswarm/dot summary plot for each class individually.
    Each class gets its own PNG file.
    """
    import pandas as pd

    X_df = pd.DataFrame(X_shap_scaled, columns=selected_features)

    for class_name in class_names:
        sv = class_shap_dict[class_name]

        # Identify top_n features by mean |SHAP| for this class
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_n]
        top_features = [selected_features[i] for i in top_idx]

        sv_top = sv[:, top_idx]
        X_top = X_df[top_features]

        shap_explanation = shap.Explanation(
            values=sv_top,
            data=X_top.values,
            feature_names=top_features
        )

        plt.figure()
        shap.plots.beeswarm(shap_explanation, show=False, max_display=top_n)
        plt.title(f"SHAP Summary — {class_name}", pad=12)
        plt.tight_layout()

        safe_name = class_name.replace("/", "_").replace(" ", "_")
        save_fig(f"shap_summary_{safe_name}.png")


# ============================================================
# IDENTIFY WEAK CLASSES
# ============================================================

def identify_weak_classes(per_class_df):
    print("\nIdentifying weak classes from All Features Random Forest validation result...")

    rf_val = per_class_df[
        (per_class_df["Feature Strategy"] == "All Features") &
        (per_class_df["Model"] == "Random Forest") &
        (per_class_df["Dataset"] == "Validation")
    ].copy()

    if rf_val.empty:
        raise ValueError("Random Forest validation results not found.")

    rf_val = rf_val.sort_values(by="Validation_F1", ascending=True)

    print("\nRandom Forest validation per-class result:")
    print(
        rf_val[
            ["Class", "Validation_Precision", "Validation_Recall", "Validation_F1"]
        ]
    )

    weak_classes = rf_val[
        rf_val["Validation_F1"] < WEAK_CLASS_F1_THRESHOLD
    ]["Class"].tolist()

    if len(weak_classes) == 0:
        weak_classes = rf_val.head(3)["Class"].tolist()

    print("\nWeak/focus classes:")
    for cls in weak_classes:
        print(" -", cls)

    return weak_classes


# ============================================================
# EXPERIMENT 1: ALL FEATURES
# ============================================================

def experiment_all_features(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    label_encoder
):
    selected_features = X_train.columns.tolist()

    summary_df, per_class_df, trained_models = train_and_evaluate_feature_set(
        feature_strategy_name="All Features",
        selected_features=selected_features,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        label_encoder=label_encoder
    )

    save_csv(summary_df, "results_all_features_summary.csv")
    save_csv(per_class_df, "results_all_features_per_class.csv")

    return summary_df, per_class_df, trained_models


# ============================================================
# EXPERIMENT 2: GLOBAL SHAP TOP 10
# ============================================================

def experiment_global_shap_top_10(
    global_importance_df,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    label_encoder
):
    selected_features = global_importance_df.head(10)["Feature"].tolist()

    summary_df, per_class_df, trained_models = train_and_evaluate_feature_set(
        feature_strategy_name="Global SHAP Top 10",
        selected_features=selected_features,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        label_encoder=label_encoder
    )

    save_csv(summary_df, "results_global_shap_top_10_summary.csv")
    save_csv(per_class_df, "results_global_shap_top_10_per_class.csv")

    return summary_df, per_class_df, trained_models


# ============================================================
# EXPERIMENT 3: GLOBAL SHAP TOP 20
# ============================================================

def experiment_global_shap_top_20(
    global_importance_df,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    label_encoder
):
    selected_features = global_importance_df.head(20)["Feature"].tolist()

    summary_df, per_class_df, trained_models = train_and_evaluate_feature_set(
        feature_strategy_name="Global SHAP Top 20",
        selected_features=selected_features,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        label_encoder=label_encoder
    )

    save_csv(summary_df, "results_global_shap_top_20_summary.csv")
    save_csv(per_class_df, "results_global_shap_top_20_per_class.csv")

    return summary_df, per_class_df, trained_models


# ============================================================
# EXPERIMENT 4: CLASS-WISE SHAP UNION
# ============================================================

def experiment_classwise_shap_union(
    classwise_importance_df,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    label_encoder
):
    selected_features = []

    for class_name in classwise_importance_df["Class"].unique():
        top_features = classwise_importance_df[
            classwise_importance_df["Class"] == class_name
        ].head(CLASSWISE_TOP_N)["Feature"].tolist()

        selected_features.extend(top_features)

    selected_features = list(dict.fromkeys(selected_features))

    summary_df, per_class_df, trained_models = train_and_evaluate_feature_set(
        feature_strategy_name=f"Class-wise SHAP Union Top {CLASSWISE_TOP_N} Each",
        selected_features=selected_features,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        label_encoder=label_encoder
    )

    save_csv(summary_df, "results_classwise_shap_union_summary.csv")
    save_csv(per_class_df, "results_classwise_shap_union_per_class.csv")

    return summary_df, per_class_df, trained_models


# ============================================================
# EXPERIMENT 5: WEAK-CLASS BOOSTED SHAP UNION
# ============================================================

def experiment_weak_class_boosted_shap_union(
    classwise_importance_df,
    weak_classes,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    label_encoder
):
    selected_features = []

    for class_name in classwise_importance_df["Class"].unique():
        top_features = classwise_importance_df[
            classwise_importance_df["Class"] == class_name
        ].head(CLASSWISE_TOP_N)["Feature"].tolist()

        selected_features.extend(top_features)

    for weak_class in weak_classes:
        weak_features = classwise_importance_df[
            classwise_importance_df["Class"] == weak_class
        ].head(WEAK_CLASS_TOP_N)["Feature"].tolist()

        selected_features.extend(weak_features)

    selected_features = list(dict.fromkeys(selected_features))

    summary_df, per_class_df, trained_models = train_and_evaluate_feature_set(
        feature_strategy_name=f"Weak-class Boosted SHAP Union Top {CLASSWISE_TOP_N} Each + Top {WEAK_CLASS_TOP_N} Weak",
        selected_features=selected_features,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        label_encoder=label_encoder
    )

    save_csv(summary_df, "results_weak_class_boosted_shap_union_summary.csv")
    save_csv(per_class_df, "results_weak_class_boosted_shap_union_per_class.csv")

    return summary_df, per_class_df, trained_models


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # --------------------------------------------------------
    # Step 1: Load and preprocess
    # --------------------------------------------------------
    df = load_data()
    df = fix_label_column(df)
    df = clean_data(df)
    df = standardize_labels(df)

    show_distribution(df, "ORIGINAL DISTRIBUTION")

    df = remove_leakage_columns(df)
    df = filter_rare_classes(df)
    df = cap_large_classes(df)

    X, y = prepare_features(df)

    X_train, X_val, X_test, y_train_str, y_val_str, y_test_str = split_data(X, y)

    y_train, y_val, y_test, label_encoder = encode_labels(
        y_train_str, y_val_str, y_test_str
    )

    # --------------------------------------------------------
    # Experiment 1: All Features
    # --------------------------------------------------------
    all_summary_df, all_per_class_df, _ = experiment_all_features(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        label_encoder
    )

    # --------------------------------------------------------
    # Step 2: SHAP on all features
    # --------------------------------------------------------
    all_features = X_train.columns.tolist()

    rf_model, rf_scaler = train_rf_for_shap(X_train, y_train, all_features)

    X_shap, y_shap = create_balanced_shap_sample(X_train, y_train_str)

    global_importance_df, classwise_importance_df = compute_global_and_classwise_shap(
        rf_model=rf_model,
        scaler=rf_scaler,
        X_shap=X_shap,
        selected_features=all_features,
        label_encoder=label_encoder
    )

    # --------------------------------------------------------
    # Step 3: Identify weak classes
    # --------------------------------------------------------
    weak_classes = identify_weak_classes(all_per_class_df)

    # --------------------------------------------------------
    # Experiments 2–5
    # --------------------------------------------------------
    global_10_summary_df, global_10_per_class_df, _ = experiment_global_shap_top_10(
        global_importance_df,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        label_encoder
    )

    global_20_summary_df, global_20_per_class_df, _ = experiment_global_shap_top_20(
        global_importance_df,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        label_encoder
    )

    classwise_summary_df, classwise_per_class_df, _ = experiment_classwise_shap_union(
        classwise_importance_df,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        label_encoder
    )

    weak_boost_summary_df, weak_boost_per_class_df, _ = experiment_weak_class_boosted_shap_union(
        classwise_importance_df,
        weak_classes,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        label_encoder
    )

    # --------------------------------------------------------
    # Final combined results
    # --------------------------------------------------------
    final_summary_df = pd.concat(
        [
            all_summary_df,
            global_10_summary_df,
            global_20_summary_df,
            classwise_summary_df,
            weak_boost_summary_df
        ],
        ignore_index=True
    )

    final_per_class_df = pd.concat(
        [
            all_per_class_df,
            global_10_per_class_df,
            global_20_per_class_df,
            classwise_per_class_df,
            weak_boost_per_class_df
        ],
        ignore_index=True
    )

    save_csv(final_summary_df, "final_all_experiments_summary.csv")
    save_csv(final_per_class_df, "final_all_experiments_per_class.csv")

    print("\n" + "=" * 100)
    print("FINAL SUMMARY SORTED BY TEST MACRO F1")
    print("=" * 100)
    print(
        final_summary_df.sort_values(
            by=["Test Macro F1", "Test Worst-class F1", "Test Accuracy"],
            ascending=False
        )
    )

    print("\n" + "=" * 100)
    print("FINAL SUMMARY SORTED BY TEST WORST-CLASS F1")
    print("=" * 100)
    print(
        final_summary_df.sort_values(
            by=["Test Worst-class F1", "Test Macro F1", "Test Accuracy"],
            ascending=False
        )
    )

    print(f"\nAll outputs saved to: {OUTPUT_PATH}")
    print("\nSaved files:")
    print("  CSVs:")
    print("    1.  results_all_features_summary.csv")
    print("    2.  results_all_features_per_class.csv")
    print("    3.  results_global_shap_top_10_summary.csv")
    print("    4.  results_global_shap_top_10_per_class.csv")
    print("    5.  results_global_shap_top_20_summary.csv")
    print("    6.  results_global_shap_top_20_per_class.csv")
    print("    7.  results_classwise_shap_union_summary.csv")
    print("    8.  results_classwise_shap_union_per_class.csv")
    print("    9.  results_weak_class_boosted_shap_union_summary.csv")
    print("    10. results_weak_class_boosted_shap_union_per_class.csv")
    print("    11. final_all_experiments_summary.csv")
    print("    12. final_all_experiments_per_class.csv")
    print("    13. global_shap_importance.csv")
    print("    14. classwise_shap_importance.csv")
    print("  Graphs:")
    print("    15. shap_global_bar.png          — top-20 global feature importance bar chart")
    print("    16. shap_classwise_bars.png       — per-class top-10 bar charts (multi-panel)")
    print("    17. shap_summary_<ClassName>.png  — beeswarm dot plot per class")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()