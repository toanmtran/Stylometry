import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================
# CONFIGURATION
# ==========================================
FEATURE_RANKING = [
    "punct_apost_rate", "n_tokens", "punct_semicolon_rate", "contraction_rate",
    "punct_paren_rate", "punct_colon_rate", "pos_verb", "fw_which", "honore_r",
    "fw_his", "pos_det", "pos_adj", "n_vocab", "avg_paragraph_len", "fw_the",
    "fw_you", "fw_what", "fw_but", "simpson_d", "fw_who", "n_sentences",
    "fw_not", "std_word_len", "word_len_1_frac", "brunet_w", "pos_adv",
    "cat_amplifier_rate", "hapax_ratio", "pos_prep", "fw_this",
    "punct_dquote_rate", "yule_k", "word_len_3_frac", "fw_he", "fw_about",
    "uppercase_ratio", "pos_pron", "punct_comma_rate", "fw_we",
    "punct_period_rate", "pronoun_start_rate", "fw_me", "median_sent_len",
    "pos_conj", "fw_if", "fw_or", "fw_i", "fw_and", "word_len_4_frac",
    "fw_that", "fw_her", "fw_on", "fw_by", "fw_would", "fw_at", "char_i_rate",
    "fw_from", "fw_their", "fw_do", "char_a_rate", "word_len_2_frac",
    "fw_they", "fw_will", "punct_question_rate", "char_e_rate", "fw_to",
    "fw_of", "cat_conj_rate", "punct_excl_rate", "fw_up", "fw_one",
    "fw_have", "avg_sent_len", "fw_out",
]

NETWORK_CONFIGS = {
    "Depth 1 (64,)":     (64,),
    "Depth 2 (64, 32)":  (64, 32),
    "Depth 3 (64, 64, 64)": (64, 64, 64),
    "Depth 10":          tuple([64] * 10),
    "Depth 50":          tuple([64] * 50),
}

FEATURE_SUBSETS = [15, 30, 50, 74]   # top-k; 74 = full ranked set
METADATA_COLS   = {"author", "passage_id"}


# ==========================================
# HELPERS
# ==========================================
def load_data(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["author"])
    return df, X, y, le


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    kept = []
    for _, group in df.groupby("author"):
        clf = IsolationForest(contamination="auto", random_state=42)
        mask = clf.fit_predict(group[feature_cols].values) == 1
        kept.append(group[mask])
    return pd.concat(kept).reset_index(drop=True)


def get_ordered_features(df: pd.DataFrame) -> list[str]:
    available = set(df.columns) - METADATA_COLS
    return [f for f in FEATURE_RANKING if f in available]


def evaluate(
    df: pd.DataFrame,
    ordered_features: list[str],
    early_stopping: bool,
) -> str:
    all_feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df["author"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    eval_tasks: list[tuple[str, np.ndarray]] = []
    for k in FEATURE_SUBSETS:
        cols = ordered_features[:k]
        eval_tasks.append((f"EVALUATING TOP {len(cols)} FEATURES", df[cols].values))
    eval_tasks.append((f"EVALUATING ALL {len(all_feature_cols)} FEATURES", df[all_feature_cols].values))

    md = "# MLP Classifier Evaluation Results (Subsets & Full Features)\n\n"

    for header, X_raw in eval_tasks:
        print(f"\n{'='*50}\n{header}\n{'='*50}")
        md += f"## {header}\n\n"

        for label, config in NETWORK_CONFIGS.items():
            print(f"\n--- 5-Fold CV: {label} ---")
            md += f"### 5-Fold CV: {label}\n\n"

            fold_reports, train_accs, test_accs = [], [], []

            for train_idx, test_idx in skf.split(X_raw, y):
                # Fit scaler on train only — no data leakage
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_raw[train_idx])
                X_test  = scaler.transform(X_raw[test_idx])

                clf = MLPClassifier(
                    hidden_layer_sizes=config,
                    max_iter=1000,
                    random_state=42,
                    solver="adam",
                    early_stopping=early_stopping,
                )
                clf.fit(X_train, y[train_idx])

                train_accs.append(clf.score(X_train, y[train_idx]))
                test_accs.append(clf.score(X_test,  y[test_idx]))

                report = classification_report(
                    y[test_idx],
                    clf.predict(X_test),
                    target_names=le.classes_,
                    output_dict=True,
                    zero_division=0,
                )
                fold_reports.append(pd.DataFrame(report).T)

            mean_train = np.mean(train_accs)
            mean_test  = np.mean(test_accs)
            print(f"  Mean Training Accuracy: {mean_train:.4f}")
            print(f"  Mean Testing Accuracy:  {mean_test:.4f}")

            md += f"- **Mean Training Accuracy:** {mean_train:.4f}\n"
            md += f"- **Mean Testing Accuracy:** {mean_test:.4f}\n\n"

            avg_stats    = pd.concat(fold_reports).groupby(level=0).mean()
            display_rows = list(le.classes_) + ["macro avg", "weighted avg"]
            table        = avg_stats.loc[display_rows]
            print(table.to_string())
            md += table.to_markdown() + "\n\n"

    return md


# ==========================================
# MAIN
# ==========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train & evaluate MLP stylometric classifier")
    parser.add_argument("--input",  default="author_features_extracted_full.csv",
                        help="Input CSV file (default: author_features_extracted_full.csv)")
    parser.add_argument("--output", default="evaluation_results_full_and_subsets.md",
                        help="Output markdown file (default: evaluation_results_full_and_subsets.md)")
    parser.add_argument("--remove-outliers", action="store_true",
                        help="Remove outliers with Isolation Forest before training")
    parser.add_argument("--early-stopping", action="store_true",
                        help="Enable early stopping during MLP training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    if args.remove_outliers:
        before = len(df)
        df = remove_outliers(df)
        print(f"  Outlier removal: {before} → {len(df)} rows")

    ordered_features = get_ordered_features(df)
    md = evaluate(df, ordered_features, early_stopping=args.early_stopping)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nResults saved to '{args.output}'")


if __name__ == "__main__":
    main()
