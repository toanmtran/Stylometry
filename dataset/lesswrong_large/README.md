INFORMATION ABOUT THE LESSWRONG_LARGE DATA SET

1. \raw: The raw data collected from lesswrong, containing data for 40 authors, each satisfying:

* at least 30 articles
* Nominal length (pre-processing) of at least 1500 words

2. \cleaned: The cleaned data for those authors from \raw

The cleaning process involves:

* Run scripts\clean_lesswrong_regular.py
* Surgically remove outliners and noisy articles using Claude
    Prompt: Please go through the [name] dataset (let's do 10 authors at a time) and surgically remove all articles that are interviews, talks, audio transcripts, articles written by multiple authors, articles where the supposed author is not an author but just a supervisor/adviser, articles with too much noisy content (code, quotes, dense math formulas) that would affect stylometry analysis.
* Manually refine the articles (removing tags, headers, etc.)
* Remove all posts whose length < 1500 words (using scripts/remove_short_articles_clean35.py).

329 articles were removed from the \raw dataset.

3. cleaned_35: A set curated from \cleaned

* Extensively prune outliner articles from \cleaned using scripts\prune_outliners_cleaned25.py (remove articles too different from others in their author group, _GAP_SIGMA = 2.0)

* Remove authors with too few articles (< 27 articles). 

* All 35 authors have at least 27 articles with length ≥1526 words.

4. cleaned_25: A set curated from \cleaned_35

This set is primarily optimized for K-means performance.

* Prune further outliner articles from \cleaned using scripts\prune_outliners_cleaned25.py (_GAP_SIGMA = 1.8)

* 35 authors were ranked by how well they cluster relative to other authors (using scripts\rank_author_sihoutte.py). Most noisy authors were removed. Some other authors were additionally removed if they had two few articles relative to others. All remaining 25 authors have at least 26 articles with length ≥1526 words.
