INFORMATION ABOUT THE LESSWRONG_REGULAR DATASET

1. \raw: The raw data collected from lesswrong, containing data for 11 authors, each satisfying:

* at least 20 articles  ≥ 2024
* at least 20 articles  ≤ 2022
* Nominal length (pre-processing) of at least 1500 words

2. \cleaned: The cleaned data for those authors from \raw (after running scripts\clean_lesswrong_regular.py).

After cleaning, the length of some articles could be significantly reduced. Articles might be removed entirely due to various reasons (e.g., duplicates, too short, low sentence-density). Removed articles (with reason for removal) are stored in dataset\lesswrong_regular\cleaned\_dropped.jsonl.

3. cleaned_10: A set curated from \cleaned with extensive outliner/noise removal. The set contains 10 authors with exactly 40 articles each, split evenly between two periods: ≥ 2024, ≤ 2022. The global minimum length is 1526 words.

4. cleaned_4: A set curated from \cleaned with only four authors who have a high number (75-77) of articles with length ≥ 1526 words. All articles shorter than this length have been removed.

5. cleaned_5: A set curated from  \cleaned with only five authors who have a high number (35-57) of articles  with length ≥ 3000 words. All articles shorter than this length have been removed.