"""One-off: does a Random Forest's feature ranking agree with the
univariate-filter FEATURE_RANKING we use for the top-30 / top-50 subsets?

Recreates the user's RF setup exactly:
    y = first column (author);  X = the 107 features
    train_test_split(test_size=0.2, stratify=y, random_state=0)
    RandomForestClassifier(n_estimators=50000, max_depth=5, random_state=0)
then ranks features two ways -- impurity (feature_importances_) and
permutation_importance on the test split -- and reports overlap with the
existing top-30 / top-50.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "data"
CSV = DATA / "author_features_extracted_full.csv"
METADATA = ["author", "passage_id"]

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


def overlap(rf_ranked, n):
    a, b = set(rf_ranked[:n]), set(FEATURE_RANKING[:n])
    return len(a & b), sorted(b - a)


def main():
    df = pd.read_csv(CSV)
    y = df["author"]
    X = df.drop(columns=METADATA)
    print(f"{X.shape[0]} passages x {X.shape[1]} features, {y.nunique()} classes\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)

    rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=0,
                                n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"train acc {rf.score(X_train, y_train):.6f}   "
          f"test acc {rf.score(X_test, y_test):.6f}\n")

    # 1) impurity-based importance
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    imp_ranked = list(imp.sort_values(ascending=False).index)

    # 2) permutation importance on the held-out test split
    perm = permutation_importance(rf, X_test, y_test, random_state=0)
    perm_ser = pd.Series(perm.importances_mean, index=X.columns)
    perm_ranked = list(perm_ser.sort_values(ascending=False).index)

    print("Top-10 by impurity importance:")
    print(imp.sort_values(ascending=False).head(10).to_string(), "\n")
    print("Top-10 by permutation importance:")
    print(perm_ser.sort_values(ascending=False).head(10).to_string(), "\n")

    for name, ranked in [("impurity", imp_ranked), ("permutation", perm_ranked)]:
        print(f"=== {name} vs existing FEATURE_RANKING ===")
        for n in (30, 50):
            ov, missing = overlap(ranked, n)
            exact = ranked[:n] == FEATURE_RANKING[:n]
            print(f"  top-{n}: overlap {ov}/{n} ({ov/n:.0%})   "
                  f"exact-ordered match: {exact}")
            print(f"    in existing top-{n} but NOT in RF top-{n}: {missing}")
        print()

    # dump full orderings so downstream scripts can adopt them
    import json
    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "impurity_ranking": imp_ranked,
        "permutation_ranking": perm_ranked,
        "permutation_importances": {f: float(perm_ser[f]) for f in perm_ranked},
    }
    (out / "rf_rankings.json").write_text(json.dumps(payload, indent=2))
    print("Full permutation ranking (all 107):")
    print(perm_ranked)
    print("\nSaved -> outputs/rf_rankings.json")


if __name__ == "__main__":
    main()
