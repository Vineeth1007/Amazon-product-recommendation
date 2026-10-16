# Amazon Electronics Product Recommendation

A notebook that compares a non-personalised popularity recommender with a personalised SVD collaborative-filtering recommender on a small, dense sample of Amazon Electronics ratings.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

> Type: Jupyter notebook (recommender systems)

## Overview

Online stores such as Amazon rely on recommendation models to suggest different products to different users. This project builds two recommenders for electronics products from user-product ratings: a popularity model that ranks products by how often they were rated, and a model-based collaborative-filtering model that factorises the user-product rating matrix with truncated SVD.

The notebook takes the first 50,000 of the 7,824,482 ratings in the Electronics ratings file, keeps the users who rated at least 15 products within that slice (11 users, 219 ratings, 186 products), and then builds and compares the two recommenders. The popularity model returns the same top-5 list for every user, while the SVD model returns a list per user. The reported RMSE of 0.05854 is not a held-out evaluation (see [Results](#results) and [Limitations](#limitations-and-known-issues)).

## Contents

| File | What it is |
|---|---|
| [amazon-product-recommendation.ipynb](amazon-product-recommendation.ipynb) | The full analysis, committed with its saved outputs (76 cells in 8 numbered sections) |
| [requirements.txt](requirements.txt) | Python packages imported by the notebook |
| `ratings_Electronics.csv` | The input data. **Not committed** - see [Data](#data) |

## Data

- **Source:** Amazon Reviews data, Electronics category, ratings-only file. The data page cited by this project is <http://jmcauley.ucsd.edu/data/amazon/> (UCSD, J. McAuley). Terms of use are set by the data provider; check the data page before reusing it.
- **File the notebook expects:** `ratings_Electronics.csv`, read with `pd.read_csv('ratings_Electronics.csv', names=['userId', 'productId', 'ratings', 'timestamp'])`. The file therefore has no header row and four columns: user ID, product ID, rating (1 to 5) and timestamp.
- **Size:** 7,824,482 rows as loaded by the notebook (about 179 MB in memory with three columns, after `timestamp` is dropped; section 2 output).
- **Committed?** No. The file is large, so it is listed in `.gitignore`. Download it from the data page above and place it **next to the notebook, in the repository root**, because the notebook opens it by bare relative path.

## Approach

The notebook is organised in numbered sections:

1. **Imports** - pandas, NumPy, scikit-learn, SciPy, seaborn and Matplotlib.
2. **Load and inspect the data** - read the CSV, name the columns, drop `timestamp`, and work on the first 50,000 rows (mean rating 4.03524, ratings from 1 to 5, 46,554 unique users and 3,446 unique products in that slice). The missing-value check on the full data finds none.
3. **Make the matrix denser** - keep only users with 15 or more ratings in that slice. This leaves 219 ratings from 11 users, a user-by-product matrix of shape (11, 186) with 10.70% of cells filled.
4. **Split** - random 70/30 train/test split of the 219 ratings (`random_state=0`): 153 training rows and 66 test rows.
5. **Popularity recommender** - the score of a product is the number of ratings it has in the training split; the top 5 by score (ties broken by product ID) are recommended to every user.
6. **Collaborative filtering** - train and test rows are concatenated back into one user-by-product matrix (unrated cells filled with 0), decomposed with `scipy.sparse.linalg.svds` using `k=10`, and multiplied back together to obtain predicted ratings. For a chosen user, the unrated products with the highest predicted rating are recommended.
7. **Evaluation** - RMSE between the average actual and the average predicted rating of each of the 186 products.
8. **Top-K recommendations** - recommendation lists for individual users with K = 5, followed by a short written summary of the two approaches.

## Results

| Recommender | Output | Notebook section |
|---|---|---|
| Popularity (non-personalised) | Identical top 5 for every user: `B00004RC2D` (score 4), `B00002SWHH` (3), `B00004SC3Y` (3), `B00004SCKA` (3), `B00000J3II` (2) | 5 |
| SVD collaborative filtering (`k=10`) | A different top-5 list per user, for example users 4, 6, 8 and 9 receive different product lists | 6 and 8 |
| SVD reconstruction error | RMSE = 0.05854 | 7 |

Caveats for reading these results:

- The RMSE compares the column averages of the actual matrix with the column averages of the SVD reconstruction over all 186 products. The SVD is fitted on train and test ratings together, so this number measures how closely the low-rank reconstruction reproduces average ratings. It is not an accuracy estimate for unseen ratings.
- In the printed recommendation lists, all five products shown for a given user have the same predicted score (users 4 and 8: 1.526746; user 6: 0.038989; user 9: 0.023351), so the order within each list is effectively arbitrary.
- "User 4", "user 6" and so on are 1-based positions in the pivot table (sorted by Amazon user ID), not Amazon user IDs.
- With only 11 users the results illustrate the technique; they are not evidence about how the method performs on the full dataset.

## How to run

1. Create a Python environment. The notebook's saved metadata records Python 3.7.3.
2. Install the dependencies and Jupyter:
   ```bash
   pip install -r requirements.txt
   pip install notebook
   ```
3. Download `ratings_Electronics.csv` (see [Data](#data)) into the repository root, next to the notebook.
4. Open and run the notebook:
   ```bash
   jupyter notebook amazon-product-recommendation.ipynb
   ```

The notebook was left unmodified. It imports `sklearn.externals.joblib` (section 1), which was removed in scikit-learn 0.23, and calls `sns.factorplot` (section 2), which was renamed `catplot` in seaborn 0.9. On current library versions, either use older releases of those two packages or adjust those two lines.

## Project structure

```text
.
├── README.md                              # this file
├── amazon-product-recommendation.ipynb    # the analysis, with saved outputs
├── requirements.txt                       # packages imported by the notebook
├── .gitignore                             # keeps the large CSV and notebook checkpoints out of git
└── ratings_Electronics.csv                # NOT committed - download it and place it here
```

## Limitations and known issues

- The analysis uses a very small sample: the first 50,000 rows of the file (not a random sample), reduced to 11 users and 219 ratings.
- There is no held-out evaluation of the recommenders (see [Results](#results)).
- The notebook has no pinned dependencies and relies on APIs that were removed in later library releases (see [How to run](#how-to-run)).
- Some printed labels and comments in the notebook do not match its code or outputs. For example, section 3 prints "users who have rated 25 or more items" although the filter is 15 or more ratings, its "unique products" line counts users instead of products, and a markdown note says "7 products and 236 users" while the output shape is (11, 186). The notebook was kept as it was; the figures in this README are taken from the code and outputs, not from those comments.

## Credits

- **Data:** Amazon Reviews data from the UCSD collection linked in [Data](#data).
- **Notebook:** the notebook was added to this repository inside a folder named `Amzon-Product-Recommendation-master`. The repository does not record who wrote it, so this README makes no claim of authorship; attribution should be added here if the original source is known.
- **Libraries:** pandas, NumPy, SciPy, scikit-learn, seaborn and Matplotlib.

## License

No license file - all rights reserved by the respective authors. The dataset is subject to its own terms of use.
