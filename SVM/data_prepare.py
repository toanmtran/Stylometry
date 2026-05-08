import pandas as pd
import numpy as np
import random
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_pairs(features, authors_array, selected_authors, max_pairs_per_auth=200):
    author_to_id = {auth: idx for idx, auth in enumerate(selected_authors)}
    
    pos_pairs, pos_groups = [], []
    neg_pairs, neg_groups = [], []
    
    # 1. Generate positive pairs
    for auth in selected_authors:
        idx = np.where(authors_array == auth)[0]
        
        all_combos = list(combinations(idx, 2))
        if len(all_combos) > max_pairs_per_auth:
            all_combos = random.sample(all_combos, max_pairs_per_auth)

        for i, j in all_combos:
            f1, f2 = features[i], features[j]
            
            # Compute the resultal features
            abs_diff = np.abs(f1 - f2)
            sq_diff = (f1 - f2)**2
            mult_feat = f1 * f2

            combined = np.concatenate([abs_diff, sq_diff, mult_feat])
            
            pos_pairs.append(combined)
            pos_groups.append(author_to_id[auth])
            
    # 2.  Generate negative pairs
    num_pos = len(pos_pairs)
    selected_authors_list = list(selected_authors)
    
    for _ in range(num_pos):
        a1, a2 = random.sample(selected_authors_list, 2)
        idx1 = random.choice(np.where(authors_array == a1)[0])
        idx2 = random.choice(np.where(authors_array == a2)[0])
        
        f1, f2 = features[idx1], features[idx2]

        # Compute the resultal features
        abs_diff = np.abs(f1 - f2)
        sq_diff = (f1 - f2)**2
        mult_feat = f1 * f2

        combined = np.concatenate([abs_diff, sq_diff, mult_feat])
        
        neg_pairs.append(combined)
        neg_groups.append(author_to_id[a1]) 
        
    X = np.vstack((pos_pairs, neg_pairs))
    
    # Label: 1 for positive, 0 for negative
    y = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))])
    groups = np.array(pos_groups + neg_groups)
    
    return X, y, groups

def process_and_save_data(csv_path, output_path, seed=42):
    print(f"Processing seed: {seed}...")
    
    # 1. Load Data
    df = pd.read_csv(csv_path)
    authors = df['author'].unique()
    
    # Take features columns
    feature_cols = df.drop(columns=['author', '_label'], errors='ignore').columns
    
    # 2. Split Authors
    random.seed(seed)
    np.random.seed(seed)
    train_authors, test_authors = train_test_split(authors, train_size=int(len(authors) * 4/5), random_state=seed)
    
    train_df = df[df['author'].isin(train_authors)]
    test_df = df[df['author'].isin(test_authors)]
    
    X_train_raw = train_df[feature_cols].values
    y_train_authors = train_df['author'].values
    
    X_test_raw = test_df[feature_cols].values
    y_test_authors = test_df['author'].values
    
    # 3. Generate Pairs
    X_train_pairs, y_train, groups_train = generate_pairs(X_train_raw, y_train_authors, train_authors)
    X_test_pairs, y_test, _ = generate_pairs(X_test_raw, y_test_authors, test_authors)
    
    # 4. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pairs)
    X_test_scaled = scaler.transform(X_test_pairs)
    
    # 6. Save Arrays
    np.savez_compressed(
        output_path,
        X_train=X_train_scaled,
        y_train=y_train,
        groups_train=groups_train,
        X_test=X_test_scaled,
        y_test=y_test
    )
    print(f"Done {output_path}\n")

if __name__ == "__main__":
    for run in range(3):
        current_seed = run + 19
        process_and_save_data(
            csv_path='LesswrongLarge.csv', 
            output_path=f'processed_data_seed_{current_seed}.npz',
            seed=current_seed
        )