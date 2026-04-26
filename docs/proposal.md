1. Introduction:

This project investigates whether individuals can be identified by their unique writing style (“writing fingerprints”) using machine learning.

We extract various stylistic features (e.g., sentence length, punctuation, word distribution) and apply K-means clustering to group texts by author without using content. We analyze factors affecting performance, which may include number of clusters, possible journal editing, writing domain (news vs blogs), and differences before and after AI-assisted writing.

We plan to build additional supervised models:

* Logistic regression to predict if two texts share the same author
* SVM for multi-class author classification
* Neural networks to compare performance with SVM

We evaluate trade-offs in accuracy, data requirements, and feature engineering. Applications include crime investigation (narrowing down suspects based on writing), historical authorship attribution, and academic integrity detection.

2. Data Description

* Lesswrong (5 authors, 150 each)
Author's Name | Total number of articles | Number of articles from 2024 or before | Number of articles from 2025 or after

* Lesswrong (20 authors, 10 each)
Author's Name | Total number of articles | Number of articles from 2022 or before | Number of articles from 2025 or after

* Guardian (5 authos, 10 each)
Author's Name | Total number of articles | Number of articles from 2022 or before | Number of articles from 2025 or after

* Aeon (5 authors, 10 each)
Author's Name | Total number of articles | Number of articles from 2022 or before | Number of articles from 2025 or after


3. Planned Implementation

a. K-means:

* Effects of K
Data: Lesswrong (20-author set)
Vary K, using all 10 articles for each author, and observe changes in performnace (e.g. ARI ?)
May use K-fold or something similar to split the data and obtain the average performance value for each K.

* Effects of possible AI assisted writing [Data: Lesswrong: —2022 | 2025— + AI generated samples]

* Effects of possible journal editing
Data: Aeon, our suspected journal editing vs. Lesswrong (20-author set + ≤ 2022) to serve as the 
Performance comparison is made for K = 20 and K = 5 (possible K-fold to take average performance)

* Effects of writing domain (blog vs. news report)
Data: Guardian (news report) vs Aeon (blog) ()


Effects of possible AI assisted writing [Data: Lesswrong: —2022 | 2025— + AI generated samples]

Binary Logistic Regression (predict Yes | No confidence) [Data: Lesswrong]
Author overfitting: Does it exist?

SVM (less data, more expertise)

Neural Net (more data, less expertise)
