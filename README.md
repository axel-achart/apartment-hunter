<p align="center">
  <a href="#english">🇬🇧 English</a> · <a href="#francais">🇫🇷 Français</a>
</p>

<a id="english"></a>

# apartment-hunter

**Estimating the sale price of a house in the Seattle area from 7 characteristics, with a web app that gives the estimate in one click.**

🇬🇧 English version · [🇫🇷 Version française](#francais)

## In short

- **Problem:** how much is a house worth? We want a quick and reliable price estimate from a few simple characteristics (size, quality, location).
- **Data:** 21,613 real house sales in King County (Seattle area, USA), 2014-2015.
- **Method:** data cleaning, exploration, selection of the 7 most useful characteristics, then comparison of 3 machine learning models, each one tuned automatically.
- **Result:** the best model estimates the price with an error of about 13% on average. It is used by a small web application where you fill in a form and get an estimated price.

## Context and problem statement

Buying or selling a house starts with one question: what is it worth? An estimate based on a few characteristics (area, quality, location) helps a buyer or an agency to spot a fair price quickly.

**Problem statement:** can we estimate the sale price of a house in King County (Seattle area, USA) from its characteristics, and how accurate can this estimate be?

To answer, we clean and explore a real sales dataset, study several regression algorithms, train and compare three of them, then use the best one in a Flask web application, containerized with Docker.

## Data

| | |
| --- | --- |
| **Source** | King County house sales dataset (`data/raw/kc_house_data.csv`, description in `data/raw/House sales king county description.txt`) |
| **Period** | Sales between 2014 and 2015 |
| **Size** | 21,613 sales, 21 columns. After cleaning: 21,606 sales (`data/clean/kc_house_data_clean.csv`) |
| **Target** | `price`, the sale price in dollars (modeled as `log(price)`) |
| **Main variables** | living and lot area, number of bedrooms and bathrooms, floors, construction grade, condition, view, waterfront, year built / renovated, zipcode, latitude, longitude, neighbors' areas |

The Madrid files in `data/raw/` were not used.

## Method

```text
   Dataset → Cleaning → Exploration (Power BI) → Feature selection
        → Train / test split → SVR · Random Forest · Gradient Boosting
        → Grid search → Comparison (MAE / RMSE / R²) → Flask application
```

### 1. Data cleaning (`notebook.ipynb`)

- **Date**: converted to a date type, then dropped. Sales only cover 2014-2015, so its impact on the price is very low.
- **Missing values**: none in the dataset, so no imputation is needed.
- **Duplicates**: 3 identical rows (once the date is dropped) are removed, because the same sale would be counted twice.
- **`id`**: dropped, it is an identifier with no predictive value.
- **Inconsistent rows**: 4 rows removed (3 houses with 0 bedrooms but 2.5 bathrooms, and 1 house with 33 bedrooms for 1,620 sqft). They are data entry errors.
- **Date consistency**: no house is renovated before being built, so `yr_built` and `yr_renovated` are kept.
- **Units**: areas in square feet (`sqft_*`) are converted to square meters (`sqm_*`).
- **Outliers**: 217 price outliers are detected by three methods at once (IQR, modified Z-score, percentiles). They are real luxury houses, not errors, so they are kept. Their weight is limited by using `log(price)` as the target and models that are not very sensitive to extreme values.
- **Standardization**: done inside the modeling pipeline, after the train / test split, to avoid data leakage.

### 2. Exploratory analysis (`analysis_explo.pbix`, Power BI)

- The median price is **450,000 $** and the mean is 540,000 $. The maximum is 7,700,000 $. The distribution is strongly skewed to the right (skewness of about 4).
- The price is mostly linked to the **size** and the **quality**: correlation with the living area of 0.70, and with the construction grade of 0.67.
- **Location** matters a lot: median price of 1,400,000 $ for waterfront houses against 450,000 $ for the others. Across the 70 zipcodes, the median goes from 235,000 $ (98002) to 1,892,500 $ (98039).
- The year of construction and the condition have almost no direct link with the price.

### 3. Feature selection (`notebook.ipynb`)

We start with 17 candidate variables. Keeping all of them brings noise, redundancy (`sqm_living` and `sqm_above` say almost the same thing) and a model harder to explain. Four selection methods are compared, on the **train set only**, with `log(price)` as the target. `zipcode` is a code and not a quantity, so it is tested separately at the end.

| Method | Principle | Result |
| --- | --- | --- |
| **Correlation** | How much each variable moves with the price (from -1 to +1). | `grade` (0.70) and `sqm_living` (0.69) are the most linked. `sqm_living` and `sqm_above` are redundant (0.88). |
| **Random Forest importance** | How much each variable helps the trees reduce the error. | `grade`, `lat` and `sqm_living` carry about 82% of the importance. |
| **Boruta** | A variable is kept only if it does better than shuffled copies (pure noise). | Keeps 12 variables out of 17. |
| **Forward selection** | Adds variables one by one, as long as the score improves enough. | Keeps 7 variables. |

We keep the features chosen by **at least 3 methods out of 4**, plus `long` (only 2 votes, but a position needs both coordinates).

**Final features (7):** `grade`, `sqm_living`, `sqm_lot`, `waterfront`, `view`, `lat`, `long`, covering **size**, **quality** and **location**.

Check (R² of a Random Forest, cross-validation on the train set, on `log(price)`):

| Features | R² |
| --- | --- |
| All 17 features | 0.8805 |
| The 7 final features | 0.8771 |
| 6 features, without `long` | 0.8337 |
| 7 final features + `zipcode` (One-Hot) | 0.8778 |

With less than half of the variables, the score only drops by 0.003. `zipcode` adds 0.0007 only, for about 70 extra columns, so `lat` + `long` is kept.

### 4. Models tested (`notebook_conclusion.ipynb`)

Eight regression algorithms were studied (see [Appendix: watch](#watch)). Three were chosen because the price is very skewed, has strong outliers and non-linear links with the variables:

| Algorithm | Principle |
| --- | --- |
| **Support Vector Regression (SVR)** | A regression with an error margin (ε-tube): errors inside the margin are not penalized. Needs standardized data (`StandardScaler` in a `Pipeline`). |
| **Random Forest** | Averages the predictions of many decision trees trained independently on random parts of the data. |
| **Gradient Boosting** | Builds trees one after the other, each one correcting the errors of the previous ones. |

All models use the same split (80 / 20, `random_state=42`): 17,284 houses for training, 4,322 for testing.

- **Choices** (settings, best model) are made with the **cross-validation MAE** on the train set (5 folds), computed on `log(price)`: 0.13 means a prediction about 13% away from the real price.
- The **test set** is only used at the end, once, to give the final scores.

### 5. Grid search

| Model | Values tested | Best combination | Time |
| --- | --- | --- | --- |
| SVR | `C`: 1, 3, 10, 30 · `gamma`: 0.05, 0.15, 0.5 · `epsilon`: 0.05, 0.1 | `C=3`, `gamma=0.5`, `epsilon=0.05` | 515 s |
| Random Forest | `n_estimators`: 100, 300, 500 · `max_depth`: 10, 20, None · `max_features`: 0.33, 0.5, 1.0 | `n_estimators=500`, `max_depth=20`, `max_features=0.5` | 797 s |
| Gradient Boosting | `n_estimators`: 200, 500, 1000 · `learning_rate`: 0.03, 0.1, 0.3 · `max_depth`: 3, 5, 7 | `n_estimators=500`, `learning_rate=0.03`, `max_depth=7` | 455 s |

### 6. Final choice

The **Gradient Boosting tuned by grid search** (500 trees, learning rate 0.03, depth 7) has the best cross-validation score, and it is also the best on the test set. It is saved in `models/best_model.joblib` and used by the Flask application.

## Results

**On houses it had never seen, the model estimates the price with an error of about 13% on average, which is about 68,000 $ per house, and it explains about 89% of the price differences between houses.**

Detailed metrics (test set of 4,322 houses; CV MAE and R² computed on `log(price)`):

| Model | Settings | CV MAE (log) | Test MAE | Test RMSE | Test R² |
| --- | --- | --- | --- | --- | --- |
| SVR | default | 0.1484 | 80,510 $ | 155,399 $ | 0.8572 |
| SVR | grid search | 0.1375 | 76,383 $ | 175,746 $ | 0.8639 |
| Random Forest | default | 0.1312 | 70,357 $ | 129,835 $ | 0.8827 |
| Random Forest | grid search | 0.1283 | 69,374 $ | 128,487 $ | 0.8871 |
| Gradient Boosting | default | 0.1409 | 76,852 $ | 139,662 $ | 0.8708 |
| **Gradient Boosting** | **grid search** | **0.1261** | **68,455 $** | **124,199 $** | **0.8920** |

- **Gradient Boosting** gains the most from the grid search: its test MAE drops from 76,852 $ to 68,455 $ (-11%, i.e. -8,397 $).
- **Random Forest** was already good with its default settings: the gain is small (about 1,000 $).
- **SVR**: the MAE improves, but the RMSE gets worse (175,746 $ instead of 155,399 $): better on most houses, but some bigger mistakes, probably on expensive houses.

## Limits and possible improvements

**Limits**

- A single county and only two years of sales (2014-2015): prices do not follow today's market.
- No information on the interior state or the exact neighborhood.
- Luxury houses are rare, so they are harder to predict.

**Possible improvements**

- More recent data, other models, more location features (distance to the center, schools).
- A prediction interval instead of a single value.
- A wider grid search: several best values are at the edge of their grid (`gamma` and `epsilon` for the SVR, `n_estimators` for the Random Forest, `learning_rate` and `max_depth` for the Gradient Boosting).
- An export of the Power BI dashboard in a readable format (PDF) next to the `.pbix` file.

## Repository structure

| File / folder | Purpose |
| --- | --- |
| `data/raw/` | Raw data (King County and Madrid) and their descriptions |
| `data/clean/` | Cleaned data (`kc_house_data_clean.csv`) and a features file (`kc_house_features.csv`) |
| `data/images/` | Diagrams illustrating the regression algorithms |
| `notebook.ipynb` | Data cleaning, first visualizations and feature selection |
| `analysis_explo.pbix` | Exploratory analysis dashboard (Power BI) |
| `notebook_conclusion.ipynb` | Study of the algorithms, training, grid search and comparison of the 3 models, saving of the best model |
| `models/best_model.joblib` | Best model (Gradient Boosting, 6.8 MB) |
| `app.py` | Flask application: a form that estimates the price of a house with the saved model |
| `templates/index.html` | HTML page of the Flask application |
| `flask.dockerfile` | Dockerfile to run the Flask application in a container |
| `requirements.txt` | Python dependencies |

## Installation and usage

Requires **Python 3.13** (the version used to train the model: the `.joblib` file is only guaranteed to load with the same scikit-learn version, pinned in `requirements.txt`).

```bash
git clone https://github.com/axel-achart/apartment-hunter.git
cd apartment-hunter
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

**Launch the web application:**

```bash
python app.py
```

Then open http://localhost:5000, fill in the 7 characteristics and click to get the estimated price. With the default values of the form, the app returns an estimate of 471,336 $.

**Or launch it with Docker** (no Python installation needed):

```bash
docker build -f flask.dockerfile -t apartment-hunter .
docker run -p 5000:5000 apartment-hunter
```

The application is then available at http://localhost:5000.

**Open the notebooks:** open `notebook.ipynb` then `notebook_conclusion.ipynb` in VS Code or Jupyter (the `ipykernel` kernel is installed by `requirements.txt`). The outputs are saved in the notebooks, so they can be read without running them. Running the 3 grid searches takes about 30 minutes.

## Team

Project made at **LaPlateforme_** (Marseille), **Bachelor AI / Data**, in September and October 2026.

| Name | GitHub |
| --- | --- |
| Axel Achart | [axel-achart](https://github.com/axel-achart) |
| Hippolyte Geslain | [Hippolyte-Geslain](https://github.com/Hippolyte-Geslain) |
| Kylliann Larcher | [kylliann-larcher](https://github.com/kylliann-larcher) |

## Tech stack

- **Language:** Python 3.13
- **Data:** pandas, NumPy
- **Visualization:** matplotlib, seaborn, Power BI
- **Machine learning:** scikit-learn (SVR, Random Forest, Gradient Boosting, GridSearchCV, Pipeline), Boruta, joblib
- **Web application:** Flask, HTML
- **Tools:** Jupyter notebooks, Git / GitHub, Docker

<a id="watch"></a>

## Appendix: watch

### The eight algorithms studied

The full study, with one diagram per algorithm, is in `notebook_conclusion.ipynb`.

| Algorithm | In one line |
| --- | --- |
| Linear regression | Fits a straight line between one variable and the target by minimizing the error. |
| Multiple linear regression | Same idea with several variables, each one with its own weight. |
| Polynomial regression | Adds powers of the variables to draw a curve, at the risk of overfitting. |
| Poisson regression | Specialized in counts (accidents, sales per shop), not in continuous prices. |
| Support Vector Regression | Linear model with an error margin that ignores small errors and resists outliers. |
| Gradient Boosting | Chains small trees, each one correcting the previous ones' errors. |
| Random Forest | Averages many independent decision trees to be robust and accurate. |
| Elastic Net | Linear regression that also shrinks the coefficients (Lasso + Ridge) to remove useless variables. |

**Why SVR, Random Forest and Gradient Boosting:** the price is very skewed with strong outliers (SVR and trees handle it better than a linear model), the links with the variables are not linear (trees capture them directly), and with about 21,600 rows these models train in a reasonable time. Linear, multiple and polynomial regressions are too simple (or overfit), Poisson regression is made for counts, and Elastic Net is linear, so it could at most be a simple baseline.

### Docker

**What it is.** Docker packages an application with everything it needs (Python, libraries, files) so it runs the same way on any computer.

| Term | Meaning |
| --- | --- |
| Dockerfile | Text file with the steps to build the image (here `flask.dockerfile`). |
| Image | Frozen package built from the Dockerfile. |
| Container | A running image. Several containers can be started from the same image. |
| Docker Hub | Online registry of images (for example the official `python` image). |
| Layer | Each Dockerfile instruction creates a layer, kept in cache if nothing changed. |

**Container or virtual machine?** A virtual machine emulates a full computer with its own operating system: heavy (several GB) and slow to start. A container shares the kernel of the host: light (a few hundred MB) and starts in seconds.

**Why Docker in this project.** The `.joblib` model only loads with the same scikit-learn version as the one used for training. The image fixes Python 3.13 and installs `requirements.txt`, so anyone can launch the application with two commands, without installing Python (see [Installation and usage](#installation-and-usage)).

**Good practices applied in `flask.dockerfile`:** a `slim` base image, `requirements.txt` installed before copying the code (cache reuse), `pip install --no-cache-dir`, and only the files needed to predict copied (`app.py`, `templates/`, `models/`).

---

<a id="francais"></a>

# apartment-hunter

**Estimer le prix de vente d'une maison de la région de Seattle à partir de 7 caractéristiques, avec une application web qui donne l'estimation en un clic.**

[🇬🇧 English version](#english) · 🇫🇷 Version française

## En bref

- **Problème :** combien vaut une maison ? Nous voulons une estimation de prix rapide et fiable à partir de quelques caractéristiques simples (taille, qualité, localisation).
- **Données :** 21 613 ventes réelles de maisons dans le comté de King (région de Seattle, États-Unis), en 2014-2015.
- **Méthode :** nettoyage des données, exploration, sélection des 7 caractéristiques les plus utiles, puis comparaison de 3 modèles de machine learning, chacun réglé automatiquement.
- **Résultat :** le meilleur modèle estime le prix avec une erreur d'environ 13 % en moyenne. Il est utilisé par une petite application web : on remplit un formulaire et on obtient un prix estimé.

## Contexte et problématique

Acheter ou vendre une maison commence par une question : combien vaut-elle ? Une estimation fondée sur quelques caractéristiques (surface, qualité, localisation) aide un acheteur ou une agence à repérer rapidement un prix correct.

**Problématique :** peut-on estimer le prix de vente d'une maison du comté de King (région de Seattle, États-Unis) à partir de ses caractéristiques, et avec quelle précision ?

Pour y répondre, nous nettoyons et explorons un vrai jeu de données de ventes, étudions plusieurs algorithmes de régression, entraînons et comparons trois d'entre eux, puis utilisons le meilleur dans une application web Flask, conteneurisée avec Docker.

## Données

| | |
| --- | --- |
| **Source** | Jeu de données des ventes de maisons du comté de King (`data/raw/kc_house_data.csv`, description dans `data/raw/House sales king county description.txt`) |
| **Période** | Ventes entre 2014 et 2015 |
| **Taille** | 21 613 ventes, 21 colonnes. Après nettoyage : 21 606 ventes (`data/clean/kc_house_data_clean.csv`) |
| **Cible** | `price`, le prix de vente en dollars (modélisé sous la forme `log(price)`) |
| **Principales variables** | surface habitable et du terrain, nombre de chambres et de salles de bain, étages, grade de construction, état, vue, front de mer, année de construction / rénovation, code postal, latitude, longitude, surfaces des voisins |

Les fichiers Madrid présents dans `data/raw/` n'ont pas été utilisés.

## Méthode

```text
   Dataset → Nettoyage → Exploration (Power BI) → Sélection des features
        → Séparation train / test → SVR · Random Forest · Gradient Boosting
        → Grid search → Comparaison (MAE / RMSE / R²) → Application Flask
```

### 1. Nettoyage des données (`notebook.ipynb`)

- **Date** : convertie en type date, puis supprimée. Les ventes ne couvrent que 2014-2015, son impact sur le prix est très faible.
- **Valeurs manquantes** : aucune dans le jeu de données, aucune imputation n'est nécessaire.
- **Doublons** : 3 lignes identiques (une fois la date retirée) sont supprimées, car la même vente serait comptée deux fois.
- **`id`** : supprimé, c'est un identifiant sans valeur prédictive.
- **Lignes incohérentes** : 4 lignes supprimées (3 maisons à 0 chambre mais 2,5 salles de bain, et 1 maison de 33 chambres pour 1 620 sqft). Ce sont des erreurs de saisie.
- **Cohérence des dates** : aucune maison n'est rénovée avant sa construction, `yr_built` et `yr_renovated` sont conservées.
- **Unités** : les surfaces en pieds carrés (`sqft_*`) sont converties en mètres carrés (`sqm_*`).
- **Valeurs aberrantes** : 217 prix aberrants sont détectés par trois méthodes à la fois (IQR, Z-score modifié, percentiles). Ce sont de vraies maisons de luxe, pas des erreurs : elles sont conservées. Leur poids est limité en prenant `log(price)` comme cible et des modèles peu sensibles aux valeurs extrêmes.
- **Standardisation** : réalisée dans le pipeline de modélisation, après la séparation train / test, pour éviter toute fuite de données.

### 2. Analyse exploratoire (`analysis_explo.pbix`, Power BI)

- Le prix médian est de **450 000 $** et le prix moyen de 540 000 $. Le maximum atteint 7 700 000 $. La distribution est fortement étalée vers la droite (asymétrie d'environ 4).
- Le prix est surtout lié à la **taille** et à la **qualité** : corrélation de 0,70 avec la surface habitable et de 0,67 avec le grade de construction.
- La **localisation** compte beaucoup : prix médian de 1 400 000 $ pour les maisons en front de mer contre 450 000 $ pour les autres. Selon les 70 codes postaux, le prix médian va de 235 000 $ (98002) à 1 892 500 $ (98039).
- L'année de construction et l'état n'ont presque aucun lien direct avec le prix.

### 3. Sélection des features (`notebook.ipynb`)

Nous partons de 17 variables candidates. Toutes les garder apporte du bruit, de la redondance (`sqm_living` et `sqm_above` disent presque la même chose) et un modèle plus difficile à expliquer. Quatre méthodes de sélection sont comparées, sur le **jeu d'entraînement uniquement**, avec `log(price)` comme cible. `zipcode` est un code et non une quantité : il est testé séparément à la fin.

| Méthode | Principe | Résultat |
| --- | --- | --- |
| **Corrélation** | À quel point chaque variable évolue avec le prix (de -1 à +1). | `grade` (0,70) et `sqm_living` (0,69) sont les plus liées. `sqm_living` et `sqm_above` sont redondantes (0,88). |
| **Importance d'un Random Forest** | Combien chaque variable aide les arbres à réduire l'erreur. | `grade`, `lat` et `sqm_living` portent environ 82 % de l'importance. |
| **Boruta** | Une variable n'est gardée que si elle fait mieux que des copies mélangées (du pur bruit). | Garde 12 variables sur 17. |
| **Forward selection** | Ajoute les variables une par une, tant que le score s'améliore assez. | Garde 7 variables. |

Nous gardons les variables choisies par **au moins 3 méthodes sur 4**, plus `long` (seulement 2 votes, mais une position demande les deux coordonnées).

**Features finales (7) :** `grade`, `sqm_living`, `sqm_lot`, `waterfront`, `view`, `lat`, `long`, qui couvrent la **taille**, la **qualité** et la **localisation**.

Vérification (R² d'un Random Forest, validation croisée sur le jeu d'entraînement, sur `log(price)`) :

| Variables | R² |
| --- | --- |
| Les 17 features | 0,8805 |
| Les 7 features finales | 0,8771 |
| 6 features, sans `long` | 0,8337 |
| 7 features finales + `zipcode` (One-Hot) | 0,8778 |

Avec moins de la moitié des variables, le score ne baisse que de 0,003. `zipcode` n'apporte que 0,0007, pour environ 70 colonnes en plus : `lat` + `long` est donc conservé.

### 4. Modèles testés (`notebook_conclusion.ipynb`)

Huit algorithmes de régression ont été étudiés (voir [Annexe : veille](#veille)). Trois ont été retenus, car le prix est très asymétrique, comporte de fortes valeurs aberrantes et des liens non linéaires avec les variables :

| Algorithme | Principe |
| --- | --- |
| **Support Vector Regression (SVR)** | Régression avec une marge d'erreur (tube ε) : les erreurs dans la marge ne sont pas pénalisées. Demande des données standardisées (`StandardScaler` dans un `Pipeline`). |
| **Random Forest** | Fait la moyenne des prédictions de nombreux arbres de décision entraînés indépendamment sur des parties aléatoires des données. |
| **Gradient Boosting** | Construit des arbres les uns après les autres, chacun corrigeant les erreurs des précédents. |

Tous les modèles utilisent la même séparation (80 / 20, `random_state=42`) : 17 284 maisons pour l'entraînement, 4 322 pour le test.

- Les **choix** (réglages, meilleur modèle) sont faits avec la **MAE en validation croisée** sur le jeu d'entraînement (5 parts), calculée sur `log(price)` : 0,13 veut dire une prédiction à environ 13 % du vrai prix.
- Le **jeu de test** ne sert qu'à la fin, une seule fois, pour donner les scores finaux.

### 5. Grid search

| Modèle | Valeurs testées | Meilleure combinaison | Temps |
| --- | --- | --- | --- |
| SVR | `C` : 1, 3, 10, 30 · `gamma` : 0.05, 0.15, 0.5 · `epsilon` : 0.05, 0.1 | `C=3`, `gamma=0.5`, `epsilon=0.05` | 515 s |
| Random Forest | `n_estimators` : 100, 300, 500 · `max_depth` : 10, 20, None · `max_features` : 0.33, 0.5, 1.0 | `n_estimators=500`, `max_depth=20`, `max_features=0.5` | 797 s |
| Gradient Boosting | `n_estimators` : 200, 500, 1000 · `learning_rate` : 0.03, 0.1, 0.3 · `max_depth` : 3, 5, 7 | `n_estimators=500`, `learning_rate=0.03`, `max_depth=7` | 455 s |

### 6. Choix final

Le **Gradient Boosting réglé par grid search** (500 arbres, learning rate 0,03, profondeur 7) a le meilleur score en validation croisée, et il est aussi le meilleur sur le jeu de test. Il est enregistré dans `models/best_model.joblib` et utilisé par l'application Flask.

## Résultats

**Sur des maisons qu'il n'avait jamais vues, le modèle estime le prix avec une erreur d'environ 13 % en moyenne, soit environ 68 000 $ par maison, et il explique environ 89 % des écarts de prix entre les maisons.**

Métriques détaillées (jeu de test de 4 322 maisons ; MAE CV et R² calculés sur `log(price)`) :

| Modèle | Réglages | MAE CV (log) | MAE test | RMSE test | R² test |
| --- | --- | --- | --- | --- | --- |
| SVR | par défaut | 0,1484 | 80 510 $ | 155 399 $ | 0,8572 |
| SVR | grid search | 0,1375 | 76 383 $ | 175 746 $ | 0,8639 |
| Random Forest | par défaut | 0,1312 | 70 357 $ | 129 835 $ | 0,8827 |
| Random Forest | grid search | 0,1283 | 69 374 $ | 128 487 $ | 0,8871 |
| Gradient Boosting | par défaut | 0,1409 | 76 852 $ | 139 662 $ | 0,8708 |
| **Gradient Boosting** | **grid search** | **0,1261** | **68 455 $** | **124 199 $** | **0,8920** |

- Le **Gradient Boosting** est celui qui gagne le plus avec la grid search : sa MAE de test passe de 76 852 $ à 68 455 $ (-11 %, soit -8 397 $).
- Le **Random Forest** était déjà bon avec ses réglages par défaut : le gain est faible (environ 1 000 $).
- **SVR** : la MAE s'améliore, mais la RMSE se dégrade (175 746 $ au lieu de 155 399 $) : meilleur sur la plupart des maisons, mais quelques erreurs plus grosses, sans doute sur les maisons chères.

## Limites et pistes d'amélioration

**Limites**

- Un seul comté et seulement deux ans de ventes (2014-2015) : les prix ne suivent pas le marché actuel.
- Pas d'information sur l'état intérieur ou le quartier exact.
- Les maisons de luxe sont rares, donc plus difficiles à prédire.

**Pistes d'amélioration**

- Des données plus récentes, d'autres modèles, plus de variables de localisation (distance au centre, écoles).
- Un intervalle de prédiction plutôt qu'une valeur unique.
- Une grid search plus large : plusieurs meilleures valeurs sont au bord de leur grille (`gamma` et `epsilon` pour le SVR, `n_estimators` pour le Random Forest, `learning_rate` et `max_depth` pour le Gradient Boosting).
- Un export du dashboard Power BI dans un format consultable (PDF) à côté du fichier `.pbix`.

## Structure du dépôt

| Fichier / dossier | Rôle |
| --- | --- |
| `data/raw/` | Données brutes (King County et Madrid) et leurs descriptions |
| `data/clean/` | Données nettoyées (`kc_house_data_clean.csv`) et un fichier de features (`kc_house_features.csv`) |
| `data/images/` | Schémas illustrant les algorithmes de régression |
| `notebook.ipynb` | Nettoyage des données, premières visualisations et sélection des features |
| `analysis_explo.pbix` | Dashboard d'analyse exploratoire (Power BI) |
| `notebook_conclusion.ipynb` | Étude des algorithmes, entraînement, grid search et comparaison des 3 modèles, enregistrement du meilleur modèle |
| `models/best_model.joblib` | Meilleur modèle (Gradient Boosting, 6,8 Mo) |
| `app.py` | Application Flask : un formulaire qui estime le prix d'une maison avec le modèle enregistré |
| `templates/index.html` | Page HTML de l'application Flask |
| `flask.dockerfile` | Dockerfile pour lancer l'application Flask dans un conteneur |
| `requirements.txt` | Dépendances Python |

## Installation et exécution

Nécessite **Python 3.13** (la version utilisée pour entraîner le modèle : le fichier `.joblib` n'est garanti qu'avec la même version de scikit-learn, fixée dans `requirements.txt`).

```bash
git clone https://github.com/axel-achart/apartment-hunter.git
cd apartment-hunter
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

**Lancer l'application web :**

```bash
python app.py
```

Ouvrir ensuite http://localhost:5000, remplir les 7 caractéristiques et valider pour obtenir le prix estimé. Avec les valeurs par défaut du formulaire, l'application renvoie une estimation de 471 336 $.

**Ou la lancer avec Docker** (sans installer Python) :

```bash
docker build -f flask.dockerfile -t apartment-hunter .
docker run -p 5000:5000 apartment-hunter
```

L'application est ensuite disponible sur http://localhost:5000.

**Ouvrir les notebooks :** ouvrir `notebook.ipynb` puis `notebook_conclusion.ipynb` dans VS Code ou Jupyter (le noyau `ipykernel` est installé par `requirements.txt`). Les sorties sont enregistrées dans les notebooks : ils se lisent sans être relancés. Relancer les 3 grid searches prend environ 30 minutes.

## Équipe

Projet réalisé à **LaPlateforme_** (Marseille), **Bachelor IA / Data**, en septembre et octobre 2026.

| Nom | GitHub |
| --- | --- |
| Axel Achart | [axel-achart](https://github.com/axel-achart) |
| Hippolyte Geslain | [Hippolyte-Geslain](https://github.com/Hippolyte-Geslain) |
| Kylliann Larcher | [kylliann-larcher](https://github.com/kylliann-larcher) |

## Stack technique

- **Langage :** Python 3.13
- **Données :** pandas, NumPy
- **Visualisation :** matplotlib, seaborn, Power BI
- **Machine learning :** scikit-learn (SVR, Random Forest, Gradient Boosting, GridSearchCV, Pipeline), Boruta, joblib
- **Application web :** Flask, HTML
- **Outils :** notebooks Jupyter, Git / GitHub, Docker

<a id="veille"></a>

## Annexe : veille

### Les huit algorithmes étudiés

L'étude complète, avec un schéma par algorithme, se trouve dans `notebook_conclusion.ipynb`.

| Algorithme | En une ligne |
| --- | --- |
| Régression linéaire | Ajuste une droite entre une variable et la cible en minimisant l'erreur. |
| Régression linéaire multiple | Même idée avec plusieurs variables, chacune avec son propre poids. |
| Régression polynomiale | Ajoute des puissances des variables pour tracer une courbe, au risque de surapprendre. |
| Régression de Poisson | Spécialisée dans les comptages (accidents, ventes par magasin), pas dans les prix continus. |
| Support Vector Regression | Modèle linéaire avec une marge d'erreur qui ignore les petites erreurs et résiste aux valeurs aberrantes. |
| Gradient Boosting | Enchaîne de petits arbres, chacun corrigeant les erreurs des précédents. |
| Random Forest | Fait la moyenne de nombreux arbres de décision indépendants pour être robuste et précis. |
| Elastic Net | Régression linéaire qui réduit aussi les coefficients (Lasso + Ridge) pour écarter les variables inutiles. |

**Pourquoi SVR, Random Forest et Gradient Boosting :** le prix est très asymétrique avec de fortes valeurs aberrantes (le SVR et les arbres s'en accommodent mieux qu'un modèle linéaire), les liens avec les variables ne sont pas linéaires (les arbres les captent directement), et avec environ 21 600 lignes ces modèles s'entraînent en un temps raisonnable. Les régressions linéaire, multiple et polynomiale sont trop simples (ou surapprennent), la régression de Poisson est faite pour des comptages, et Elastic Net est linéaire : il pourrait au mieux servir de point de comparaison.

### Docker

**Ce que c'est.** Docker emballe une application avec tout ce dont elle a besoin (Python, bibliothèques, fichiers) pour qu'elle tourne de la même façon sur n'importe quel ordinateur.

| Terme | Signification |
| --- | --- |
| Dockerfile | Fichier texte qui décrit les étapes pour construire l'image (ici `flask.dockerfile`). |
| Image | Paquet figé construit à partir du Dockerfile. |
| Conteneur | Une image en cours d'exécution. On peut lancer plusieurs conteneurs à partir de la même image. |
| Docker Hub | Registre en ligne d'images (par exemple l'image officielle `python`). |
| Couche (layer) | Chaque instruction du Dockerfile crée une couche, gardée en cache si rien n'a changé. |

**Conteneur ou machine virtuelle ?** Une machine virtuelle émule un ordinateur complet avec son propre système d'exploitation : lourde (plusieurs Go) et lente à démarrer. Un conteneur partage le noyau de la machine hôte : léger (quelques centaines de Mo) et démarre en quelques secondes.

**Pourquoi Docker dans ce projet.** Le modèle `.joblib` ne se charge qu'avec la même version de scikit-learn que celle de l'entraînement. L'image fixe Python 3.13 et installe `requirements.txt` : n'importe qui peut lancer l'application en deux commandes, sans installer Python (voir [Installation et exécution](#installation-et-exécution)).

**Bonnes pratiques appliquées dans `flask.dockerfile` :** une image de base `slim`, `requirements.txt` installé avant la copie du code (réutilisation du cache), `pip install --no-cache-dir`, et seuls les fichiers utiles pour prédire copiés (`app.py`, `templates/`, `models/`).
