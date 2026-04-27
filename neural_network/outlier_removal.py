import pandas as pd
from sklearn.ensemble import IsolationForest

INPUT_FILE = "author_features_extracted_full.csv"
OUTPUT_FILE = "feature_extracted_without_outliers.csv"
CONTAMINATION = "auto"
RANDOM_STATE = 42

df = pd.read_csv(INPUT_FILE)

feature_cols = [c for c in df.columns if c not in ("author", "passage_id")]

kept = []
removed_counts = {}

for author, group in df.groupby("author"):
    X = group[feature_cols].values
    clf = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    labels = clf.fit_predict(X)          # 1 = inlier, -1 = outlier
    inliers = group[labels == 1]
    n_removed = (labels == -1).sum()
    removed_counts[author] = n_removed
    kept.append(inliers)

result = pd.concat(kept).reset_index(drop=True)
result.to_csv(OUTPUT_FILE, index=False)

print(f"Input rows : {len(df)}")
print(f"Output rows: {len(result)}")
print()
print("Removed per author:")
for author, n in removed_counts.items():
    total = (df["author"] == author).sum()
    print(f"  {author:<25} {n:>3} / {total} removed")
