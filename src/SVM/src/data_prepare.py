import pandas as pd
import numpy as np
import random
from itertools import combinations
from sklearn.model_selection import train_test_split

# Funtion for generating pairs
def generate_pairs(features, authors_array, selected_authors, max_pairs_per_auth=200):
    # Dictionary author
    author_indices = {auth: np.where(authors_array == auth)[0] for auth in selected_authors}

    pos_pairs, neg_pairs = [], []

    # Generates positive pairs
    for auth in selected_authors:
        idx = author_indices[auth]
        # Generate combinations from an author
        all_combos = list(combinations(idx, 2))
        
        # Limit the number of pairs, if larger -> choose randomly
        if len(all_combos) > max_pairs_per_auth:
            all_combos = random.sample(all_combos, max_pairs_per_auth)

        # For each combo, generate fetures
        for i, j in all_combos:
            f1, f2 = features[i], features[j]
            # Three features: 
            # |f1 - f2|
            # (f1-f2)^2
            # f1*f2
            combined = np.concatenate([np.abs(f1 - f2), (f1 - f2)**2, f1 * f2])
            pos_pairs.append(combined)

    # Generates negative pairs
    num_pos = len(pos_pairs)
    # set -> list
    selected_authors_list = list(selected_authors)

    # len(pos_pairs) = len(neg_pairs) to be balenced
    for _ in range(num_pos):
        # Random 2 authors
        a1, a2 = random.sample(selected_authors_list, 2)
        # Random an article from each author
        idx1 = random.choice(author_indices[a1])
        idx2 = random.choice(author_indices[a2])

        # Generate features
        f1, f2 = features[idx1], features[idx2]
        combined = np.concatenate([np.abs(f1 - f2), (f1 - f2)**2, f1 * f2])
        neg_pairs.append(combined)

    # Build dataset
    # Features
    X = np.vstack((pos_pairs, neg_pairs)).astype(np.float32)
    # Labels
    y = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))]).astype(np.int8)

    return X, y


def process_and_save_data(csv_path, output_path, seed=42):
    print(f"Processing Seed: {seed}...")
    
    # Load data from csv
    df = pd.read_csv(csv_path)
    # Take unique authors
    authors = df['author'].unique()
    # Take feature columns
    feature_cols = df.drop(columns=['author', '_label'], errors='ignore').columns

    # Seed
    random.seed(seed)
    np.random.seed(seed)

    # Split Authors (80% Train, 20% Test)
    train_authors, test_authors = train_test_split(authors, train_size=int(len(authors) * 4/5), random_state=seed)

    # Train authors
    train_df = df[df['author'].isin(train_authors)]
    # Test authors
    test_df = df[df['author'].isin(test_authors)]

    # Raw features (before generating pairs)
    X_train_raw = train_df[feature_cols].values
    # Raw labels (author's name, not 0/1)
    y_train_authors = train_df['author'].values
    # Generate training pairs will be processed later in the next file
    # Because the way we generate training pairs is different from the test pairs

    # Generate pairs for test set
    X_test_raw = test_df[feature_cols].values
    y_test_authors = test_df['author'].values
    X_test_pairs, y_test_pairs = generate_pairs(X_test_raw, y_test_authors, test_authors)

    # Save data
    np.savez_compressed(
        output_path,
        # Raw data for training
        X_train_raw = X_train_raw,  
        # Raw labels          
        y_train_authors = y_train_authors, 
        # Test set  
        X_test_pairs = X_test_pairs,         
        y_test_pairs = y_test_pairs          
    )
    print(f"Saved: {output_path}\n")