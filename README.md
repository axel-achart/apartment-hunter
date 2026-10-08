# apartment-hunter

🇬🇧 English version · [🇫🇷 Version française](#apartment-hunter-1)

## Context

Buying or selling a house starts with one question: what is it worth? An estimate based on a few characteristics (area, quality, location) helps a buyer or an agency to spot a fair price quickly.

**Problem statement:** can we estimate the sale price of a house in King County (Seattle area, USA) from its characteristics, and how accurate can this estimate be?

To answer, we clean and explore a real sales dataset, study several regression algorithms, train and compare three of them, then deploy the best one in a Flask web application, containerized with Docker.

## Project structure

| File / folder | Purpose |
| --- | --- |
| `data/raw/` | Raw data (King County and Madrid) and their descriptions |
| `data/clean/` | Cleaned data (`kc_house_data_clean.csv`) |
| `data/images/` | Diagrams illustrating the regression algorithms |
| `notebook.ipynb` | Data cleaning and first visualizations |
| `analyse.pbix` | Exploratory analysis dashboard made with Power BI |
| `notebook_conclusion.ipynb` | Study of the regression algorithms, then modeling and conclusion |
| `app.py` | Flask application (to be completed) |
| `flask.dockerfile` | Dockerfile of the application (to be completed) |
| `requirements.txt` | Python dependencies |

## Progress

| Step | Status |
| --- | --- |
| Data cleaning | Done |
| Exploratory analysis (Power BI) | Done |
| Algorithm study (watch) | Done (`notebook_conclusion.ipynb`) |
| Feature selection | Done (`notebook.ipynb`) |
| Model training and comparison | To do |
| Grid search | To do |
| Flask application | To do |
| Docker | To do |

## Approach

```text
                    DATASET
                       │
                       ▼
                 Data cleaning
                       │
                       ▼
             Exploratory analysis
                   (Power BI)
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Price ↔      Price ↔      Price ↔
        Area        Quality      Location
          │            │            │
          └────────────┼────────────┘
                       ▼
               Feature selection
                       │
                       ▼
               Train / test split
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         SVR     Random Forest  Gradient Boosting
          │            │            │
          └────────────┼────────────┘
                       ▼
                  Comparison
                MAE / RMSE / R²
                       │
                       ▼
                  Grid search
                       │
                       ▼
             Flask application + Docker
```

## Data and analysis

### The dataset

The dataset is **King County** (`kc_house_data.csv`): 21,613 house sales between 2014 and 2015, with 21 columns (price, bedrooms, bathrooms, living area, lot area, floors, waterfront, view, condition, grade, year built, year renovated, zipcode, latitude, longitude, and neighbors' areas). The Madrid files in `data/raw/` were not used.

### Data cleaning

Cleaning is done in `notebook.ipynb`:

- **Date**: converted to a date type, then dropped. Sales only cover 2014-2015, so its impact on the price is very low.
- **Missing values**: none in the dataset, so no imputation is needed.
- **Duplicates**: 3 identical rows (once the date is dropped) are removed, because the same sale would be counted twice.
- **`id`**: dropped, it is an identifier with no predictive value.
- **Inconsistent rows**: 4 rows removed (3 houses with 0 bedrooms but 2.5 bathrooms, and 1 house with 33 bedrooms for 1,620 sqft). They are data entry errors.
- **Date consistency**: no house is renovated before being built, so `yr_built` and `yr_renovated` are kept.
- **Units**: areas in square feet (`sqft_*`) are converted to square meters (`sqm_*`).
- **Outliers**: 217 price outliers are detected by three methods at once (IQR, modified Z-score, percentiles). They are real luxury houses, not errors, so they are kept. Their weight is limited by using `log(price)` as the target and models that are not very sensitive to extreme values.
- **Standardization**: done inside the modeling pipeline, after the train / test split, to avoid data leakage.

The cleaned file (21,606 rows) is saved in `data/clean/kc_house_data_clean.csv`.

### Exploratory analysis

The exploratory analysis is in the Power BI dashboard `analyse.pbix`. Main facts from the cleaned data:

- The median price is **450,000 $** and the mean is 540,000 $. The maximum is 7,700,000 $. The distribution is strongly skewed to the right (skewness of about 4).
- The price is mostly linked to the **size** and the **quality**: correlation with the living area of 0.70, and with the construction grade of 0.67.
- **Location** matters a lot: median price of 1,400,000 $ for waterfront houses against 450,000 $ for the others. Across the 70 zipcodes, the median goes from 235,000 $ (98002) to 1,892,500 $ (98039).
- The year of construction and the condition have almost no direct link with the price.

An export of the dashboard in a readable format (PDF) is to be added next to the `.pbix` file.

## Algorithms used

Eight regression algorithms were studied (see [Watch](#watch)). Three were chosen:

| Algorithm | Principle |
| --- | --- |
| **Support Vector Regression (SVR)** | A regression with an error margin (ε-tube): errors inside the margin are not penalized, which makes the model less sensitive to outliers. Needs standardized data. |
| **Random Forest** | Averages the predictions of many decision trees trained independently on random parts of the data and features. Robust and little sensitive to extreme values. |
| **Gradient Boosting** | Builds trees one after the other, each one correcting the errors of the previous ones. Often the most accurate on tabular data. |

### Feature selection

- **Target:** `y = price` (modeled as `log(price)`).
- **Excluded:** `price` itself, and `sqm_price_per_living` and `sqm_price_per_lot`, because they are computed from the price and would give the answer to the model.
- **Candidate features:**

| Group | Variables |
| --- | --- |
| Housing characteristics | `bedrooms`, `bathrooms`, `floors`, `sqm_living`, `sqm_lot`, `sqm_above`, `sqm_basement` |
| Quality | `grade`, `condition`, `view`, `waterfront` |
| Age / renovation | `yr_built`, `yr_renovated` |
| Location | `zipcode`, `lat`, `long` |
| Neighborhood | `sqm_living15`, `sqm_lot15` |

#### Why select features

We start with 17 candidate variables, and keeping all of them causes problems: noise (a useless variable can make the model learn coincidences), redundancy (`sqm_living` and `sqm_above` say almost the same thing) and a heavier, harder to explain model. Each selection method looks at the data from a different angle, so none is reliable alone: we compare four of them, then keep the features they agree on.

The selection is done on the **train set only**, with `log(price)` as the target. `zipcode` is left out of the four methods, because it is a code and not a quantity (98039 is not "bigger" than 98002), so it is tested separately at the end.

#### The four methods

| Method | Principle | Result | Limit |
| --- | --- | --- | --- |
| **Correlation** | Measures how much each variable moves with the price (from -1 to +1). | `grade` (0.70) and `sqm_living` (0.69) are the most linked. `condition`, `long` and `yr_built` almost not. `sqm_living` and `sqm_above` are redundant (0.88). | Only sees straight-line links: `long` looks useless although the position matters. |
| **Random Forest importance** | Measures how much each variable helps the trees of a Random Forest to reduce the error. | `grade`, `lat` and `sqm_living` carry about 82% of the importance. | Similar variables share their importance, so one can look useless while its information is in the other. |
| **Boruta** | Creates shuffled copies of each variable (pure noise). A variable is kept only if it does better than the best noisy copy. | Keeps 12 variables out of 17. Rejects `bedrooms`, `floors`, `sqm_basement`, `yr_renovated`, `condition`. | The most permissive: keeps anything better than noise, even if barely useful. |
| **Forward selection** | Starts with no variable and adds, one by one, the one that improves the score the most, until the gain is too small. | Keeps 7 variables: `waterfront`, `view`, `grade`, `lat`, `long`, `sqm_living`, `sqm_lot`. | Slow, and the choice depends on the order of addition. It is the most compact because it avoids redundant variables. |

#### How the final choice was made

Each method votes for the features it keeps. `grade`, `sqm_living` and `lat` get 4 votes out of 4, and `waterfront`, `view` and `sqm_lot` get 3. We keep the features chosen by **at least 3 methods out of 4**, plus `long`: it only has 2 votes, but `lat` alone gives the north-south axis only, and a position needs both coordinates.

**Final features (7):** `grade`, `sqm_living`, `sqm_lot`, `waterfront`, `view`, `lat`, `long`. They cover the three ideas found in the exploration: the **size** (`sqm_living`, `sqm_lot`), the **quality** (`grade`, `view`, `waterfront`) and the **location** (`lat`, `long`).

#### Checks

A good selection must keep the performance with fewer variables. R² of a Random Forest (on the log of the price, cross-validation on the train set; the R² is the share of the price variation explained by the model):

| Features | R² |
| --- | --- |
| All 17 features | 0.8805 |
| The 7 final features | 0.8771 |
| 6 features, without `long` | 0.8337 |

- With less than half of the variables, the score only drops by 0.003.
- Removing `long` costs 0.04, which confirms it was right to keep it.
- **Location:** adding `zipcode` (One-Hot Encoding) gives 0.8778, a gain of 0.0007 only, because `lat` and `long` already describe the location. So `lat` + `long` is kept: simpler, and about 70 columns saved.

### Modeling pipelines

`zipcode` is treated as a categorical variable (One-Hot Encoding). Numerical variables are standardized for the SVR. Tree-based models do not need it.

```text
                    Data
                     │
                     ▼
              Train / Test Split
                     │
                     ▼
              ColumnTransformer
             ┌────────┴────────┐
             │                 │
         Numerical        Categorical
             │                 │
       StandardScaler      OneHotEncoder
             │                 │
             └────────┬────────┘
                      ▼
       SVR / Random Forest / Gradient Boosting
                      │
                      ▼
                 Prediction
```

### Evaluation and comparison

The models are compared with three metrics:

- **MAE**: mean absolute error, in dollars.
- **RMSE**: root mean squared error, more sensitive to large errors.
- **R²**: share of the price variance explained by the model.

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| SVR | to be filled | to be filled | to be filled |
| Random Forest | to be filled | to be filled | to be filled |
| Gradient Boosting | to be filled | to be filled | to be filled |

## Conclusion

*To be written once the models are trained and compared.* It will give:

- **Performance**: the best model, its MAE / RMSE / R², and the gain brought by the grid search.
- **Limits** already known: a single county and only two years of sales (2014-2015), so prices do not follow the market today; no information such as the interior state or the exact neighborhood; luxury houses are rare, so they are harder to predict.
- **Possible improvements**: more recent data, other models, more location features (distance to the center, schools), a prediction interval instead of a single value.

## Watch

The watch is the study of regression algorithms and the choice of the models. The full study, with one diagram per algorithm, is in `notebook_conclusion.ipynb`.

### The eight algorithms studied

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

### Why these three models

SVR, Random Forest and Gradient Boosting were chosen because they fit our data:

- The price is very skewed and has strong outliers: SVR (error margin) and tree-based models handle it better than a simple linear model.
- The links between the price and the variables are not linear (for example `grade`, or the location with `lat` and `long`): trees capture them without having to build new variables.
- With about 21,600 rows, these models train in a reasonable time.

The others were left out: linear, multiple and polynomial regressions are too simple for this non-linear relation (or overfit for the polynomial one), Poisson regression is made for counts and not for prices, and Elastic Net is linear, so it could at most serve as a simple point of comparison.

## Installation and launch

These commands will work once the application and the Dockerfile are finished.

Locally:

```bash
pip install -r requirements.txt
python app.py
```

With Docker:

```bash
docker build -f flask.dockerfile -t apartment-hunter .
docker run -p 5000:5000 apartment-hunter
```

The application is then available at <http://localhost:5000>.

---

# apartment-hunter

[🇬🇧 English version](#apartment-hunter) · 🇫🇷 Version française

## Contexte

Acheter ou vendre une maison commence par une question : combien vaut-elle ? Une estimation fondée sur quelques caractéristiques (surface, qualité, localisation) aide un acheteur ou une agence à repérer rapidement un prix correct.

**Problématique :** peut-on estimer le prix de vente d'une maison du comté de King (région de Seattle, États-Unis) à partir de ses caractéristiques, et avec quelle précision ?

Pour y répondre, nous nettoyons et explorons un vrai jeu de données de ventes, étudions plusieurs algorithmes de régression, entraînons et comparons trois d'entre eux, puis déployons le meilleur dans une application web Flask, conteneurisée avec Docker.

## Structure du projet

| Fichier / dossier | Rôle |
| --- | --- |
| `data/raw/` | Données brutes (King County et Madrid) et leurs descriptions |
| `data/clean/` | Données nettoyées (`kc_house_data_clean.csv`) |
| `data/images/` | Schémas illustrant les algorithmes de régression |
| `notebook.ipynb` | Nettoyage des données et premières visualisations |
| `analyse.pbix` | Dashboard d'analyse exploratoire réalisé avec Power BI |
| `notebook_conclusion.ipynb` | Étude des algorithmes de régression, puis modélisation et conclusion |
| `app.py` | Application Flask (à compléter) |
| `flask.dockerfile` | Dockerfile de l'application (à compléter) |
| `requirements.txt` | Dépendances Python |

## Avancement

| Étape | État |
| --- | --- |
| Nettoyage des données | Fait |
| Analyse exploratoire (Power BI) | Fait |
| Étude des algorithmes (veille) | Faite (`notebook_conclusion.ipynb`) |
| Sélection des features | Faite (`notebook.ipynb`) |
| Entraînement et comparaison des modèles | À faire |
| Grid search | À faire |
| Application Flask | À faire |
| Docker | À faire |

## Démarche

```text
                    DATASET
                       │
                       ▼
              Nettoyage des données
                       │
                       ▼
              Analyse exploratoire
                   (Power BI)
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Prix ↔       Prix ↔       Prix ↔
      Surface       Qualité    Localisation
          │            │            │
          └────────────┼────────────┘
                       ▼
              Sélection des features
                       │
                       ▼
              Séparation train / test
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         SVR     Random Forest  Gradient Boosting
          │            │            │
          └────────────┼────────────┘
                       ▼
                  Comparaison
                MAE / RMSE / R²
                       │
                       ▼
                  Grid search
                       │
                       ▼
             Application Flask + Docker
```

## Données et analyse

### Le jeu de données

Le jeu de données est **King County** (`kc_house_data.csv`) : 21 613 ventes de maisons entre 2014 et 2015, avec 21 colonnes (prix, chambres, salles de bain, surface habitable, surface du terrain, étages, front de mer, vue, état, grade, année de construction, année de rénovation, code postal, latitude, longitude et surfaces des voisins). Les fichiers Madrid présents dans `data/raw/` n'ont pas été utilisés.

### Nettoyage des données

Le nettoyage est fait dans `notebook.ipynb` :

- **Date** : convertie en type date, puis supprimée. Les ventes ne couvrent que 2014-2015, son impact sur le prix est très faible.
- **Valeurs manquantes** : aucune dans le jeu de données, aucune imputation n'est nécessaire.
- **Doublons** : 3 lignes identiques (une fois la date retirée) sont supprimées, car la même vente serait comptée deux fois.
- **`id`** : supprimé, c'est un identifiant sans valeur prédictive.
- **Lignes incohérentes** : 4 lignes supprimées (3 maisons à 0 chambre mais 2,5 salles de bain, et 1 maison de 33 chambres pour 1 620 sqft). Ce sont des erreurs de saisie.
- **Cohérence des dates** : aucune maison n'est rénovée avant sa construction, `yr_built` et `yr_renovated` sont conservées.
- **Unités** : les surfaces en pieds carrés (`sqft_*`) sont converties en mètres carrés (`sqm_*`).
- **Valeurs aberrantes** : 217 prix aberrants sont détectés par trois méthodes à la fois (IQR, Z-score modifié, percentiles). Ce sont de vraies maisons de luxe, pas des erreurs : elles sont conservées. Leur poids est limité en prenant `log(price)` comme cible et des modèles peu sensibles aux valeurs extrêmes.
- **Standardisation** : réalisée dans le pipeline de modélisation, après la séparation train / test, pour éviter toute fuite de données.

Le fichier nettoyé (21 606 lignes) est enregistré dans `data/clean/kc_house_data_clean.csv`.

### Analyse exploratoire

L'analyse exploratoire se trouve dans le dashboard Power BI `analyse.pbix`. Principaux constats sur les données nettoyées :

- Le prix médian est de **450 000 $** et le prix moyen de 540 000 $. Le maximum atteint 7 700 000 $. La distribution est fortement étalée vers la droite (asymétrie d'environ 4).
- Le prix est surtout lié à la **taille** et à la **qualité** : corrélation de 0,70 avec la surface habitable et de 0,67 avec le grade de construction.
- La **localisation** compte beaucoup : prix médian de 1 400 000 $ pour les maisons en front de mer contre 450 000 $ pour les autres. Selon les 70 codes postaux, le prix médian va de 235 000 $ (98002) à 1 892 500 $ (98039).
- L'année de construction et l'état n'ont presque aucun lien direct avec le prix.

Un export du dashboard dans un format consultable (PDF) reste à ajouter à côté du fichier `.pbix`.

## Algorithmes utilisés

Huit algorithmes de régression ont été étudiés (voir [Veille](#veille)). Trois ont été retenus :

| Algorithme | Principe |
| --- | --- |
| **Support Vector Regression (SVR)** | Régression avec une marge d'erreur (tube ε) : les erreurs dans la marge ne sont pas pénalisées, ce qui rend le modèle moins sensible aux valeurs aberrantes. Demande des données standardisées. |
| **Random Forest** | Fait la moyenne des prédictions de nombreux arbres de décision entraînés indépendamment sur des parties aléatoires des données et des variables. Robuste et peu sensible aux valeurs extrêmes. |
| **Gradient Boosting** | Construit des arbres les uns après les autres, chacun corrigeant les erreurs des précédents. Souvent le plus précis sur des données tabulaires. |

### Sélection des features

- **Cible :** `y = price` (modélisée sous la forme `log(price)`).
- **Exclues :** `price` elle-même, ainsi que `sqm_price_per_living` et `sqm_price_per_lot`, car elles sont calculées à partir du prix et donneraient la réponse au modèle.
- **Features candidates :**

| Groupe | Variables |
| --- | --- |
| Caractéristiques du logement | `bedrooms`, `bathrooms`, `floors`, `sqm_living`, `sqm_lot`, `sqm_above`, `sqm_basement` |
| Qualité | `grade`, `condition`, `view`, `waterfront` |
| Ancienneté / rénovation | `yr_built`, `yr_renovated` |
| Localisation | `zipcode`, `lat`, `long` |
| Environnement proche | `sqm_living15`, `sqm_lot15` |

#### Pourquoi sélectionner des features

Nous partons de 17 variables candidates, et toutes les garder pose des problèmes : du bruit (une variable inutile peut faire apprendre au modèle des coïncidences), de la redondance (`sqm_living` et `sqm_above` disent presque la même chose) et un modèle plus lourd et plus difficile à expliquer. Chaque méthode de sélection regarde les données sous un angle différent, donc aucune n'est fiable seule : nous en comparons quatre, puis nous gardons les variables sur lesquelles elles s'accordent.

La sélection est faite sur le **jeu d'entraînement uniquement**, avec `log(price)` comme cible. `zipcode` est mis de côté pour les quatre méthodes, car c'est un code et non une quantité (98039 n'est pas « plus grand » que 98002) ; il est donc testé séparément à la fin.

#### Les quatre méthodes

| Méthode | Principe | Résultat | Limite |
| --- | --- | --- | --- |
| **Corrélation** | Mesure à quel point chaque variable évolue avec le prix (de -1 à +1). | `grade` (0,70) et `sqm_living` (0,69) sont les plus liées. `condition`, `long` et `yr_built` presque pas. `sqm_living` et `sqm_above` sont redondantes (0,88). | Ne voit que les liens en ligne droite : `long` paraît inutile alors que la position compte. |
| **Importance d'un Random Forest** | Mesure combien chaque variable aide les arbres d'un Random Forest à réduire l'erreur. | `grade`, `lat` et `sqm_living` portent environ 82 % de l'importance. | Des variables semblables se partagent l'importance : l'une peut sembler inutile alors que son information est dans l'autre. |
| **Boruta** | Crée des copies mélangées de chaque variable (du pur bruit). Une variable n'est gardée que si elle fait mieux que la meilleure copie bruitée. | Garde 12 variables sur 17. Écarte `bedrooms`, `floors`, `sqm_basement`, `yr_renovated`, `condition`. | La plus permissive : garde tout ce qui est meilleur que du bruit, même si c'est peu utile. |
| **Forward selection** | Part de zéro variable et ajoute, une par une, celle qui améliore le plus le score, jusqu'à ce que le gain soit trop faible. | Garde 7 variables : `waterfront`, `view`, `grade`, `lat`, `long`, `sqm_living`, `sqm_lot`. | Lente, et le choix dépend de l'ordre d'ajout. C'est la plus compacte, car elle évite les variables redondantes. |

#### Comment le choix final a été fait

Chaque méthode vote pour les variables qu'elle garde. `grade`, `sqm_living` et `lat` ont 4 votes sur 4, et `waterfront`, `view` et `sqm_lot` en ont 3. Nous gardons les variables choisies par **au moins 3 méthodes sur 4**, plus `long` : elle n'a que 2 votes, mais `lat` seule ne donne que l'axe nord-sud, et une position demande les deux coordonnées.

**Features finales (7) :** `grade`, `sqm_living`, `sqm_lot`, `waterfront`, `view`, `lat`, `long`. Elles couvrent les trois idées vues dans l'exploration : la **taille** (`sqm_living`, `sqm_lot`), la **qualité** (`grade`, `view`, `waterfront`) et la **localisation** (`lat`, `long`).

#### Vérifications

Une bonne sélection doit garder la performance avec moins de variables. R² d'un Random Forest (sur le logarithme du prix, validation croisée sur le jeu d'entraînement ; le R² est la part de la variation des prix expliquée par le modèle) :

| Variables | R² |
| --- | --- |
| Les 17 features | 0,8805 |
| Les 7 features finales | 0,8771 |
| 6 features, sans `long` | 0,8337 |

- Avec moins de la moitié des variables, le score ne baisse que de 0,003.
- Retirer `long` coûte 0,04, ce qui confirme qu'il fallait le garder.
- **Localisation :** ajouter `zipcode` (One-Hot Encoding) donne 0,8778, soit un gain de 0,0007 seulement, car `lat` et `long` décrivent déjà la localisation. `lat` + `long` est donc conservé : plus simple, et environ 70 colonnes économisées.

### Pipelines de modélisation

`zipcode` est traité comme une variable catégorielle (One-Hot Encoding). Les variables numériques sont standardisées pour le SVR. Les modèles à base d'arbres n'en ont pas besoin.

```text
                    Data
                     │
                     ▼
              Train / Test Split
                     │
                     ▼
              ColumnTransformer
             ┌────────┴────────┐
             │                 │
        Numériques        Catégorielles
             │                 │
       StandardScaler      OneHotEncoder
             │                 │
             └────────┬────────┘
                      ▼
       SVR / Random Forest / Gradient Boosting
                      │
                      ▼
                 Prédiction
```

### Évaluation et comparaison

Les modèles sont comparés avec trois métriques :

- **MAE** : erreur absolue moyenne, en dollars.
- **RMSE** : racine de l'erreur quadratique moyenne, plus sensible aux grosses erreurs.
- **R²** : part de la variance du prix expliquée par le modèle.

| Modèle | MAE | RMSE | R² |
| --- | --- | --- | --- |
| SVR | à compléter | à compléter | à compléter |
| Random Forest | à compléter | à compléter | à compléter |
| Gradient Boosting | à compléter | à compléter | à compléter |

## Conclusion

*À rédiger une fois les modèles entraînés et comparés.* Elle donnera :

- **Les performances** : le meilleur modèle, ses MAE / RMSE / R², et le gain apporté par le grid search.
- **Les limites** déjà connues : un seul comté et seulement deux ans de ventes (2014-2015), donc des prix qui ne suivent pas le marché actuel ; pas d'information sur l'état intérieur ou le quartier exact ; les maisons de luxe sont rares, donc plus difficiles à prédire.
- **Les pistes d'amélioration** : des données plus récentes, d'autres modèles, plus de variables de localisation (distance au centre, écoles), un intervalle de prédiction plutôt qu'une valeur unique.

## Veille

La veille est l'étude des algorithmes de régression et le choix des modèles. L'étude complète, avec un schéma par algorithme, se trouve dans `notebook_conclusion.ipynb`.

### Les huit algorithmes étudiés

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

### Pourquoi ces trois modèles

SVR, Random Forest et Gradient Boosting ont été choisis parce qu'ils conviennent à nos données :

- Le prix est très asymétrique et comporte de fortes valeurs aberrantes : le SVR (marge d'erreur) et les modèles à base d'arbres s'en accommodent mieux qu'un simple modèle linéaire.
- Les liens entre le prix et les variables ne sont pas linéaires (par exemple `grade`, ou la localisation avec `lat` et `long`) : les arbres les captent sans qu'il faille construire de nouvelles variables.
- Avec environ 21 600 lignes, ces modèles s'entraînent en un temps raisonnable.

Les autres ont été écartés : les régressions linéaire, multiple et polynomiale sont trop simples pour cette relation non linéaire (ou surapprennent pour la polynomiale), la régression de Poisson est faite pour des comptages et non pour des prix, et Elastic Net est linéaire, donc il pourrait au mieux servir de simple point de comparaison.

## Installation et lancement

Ces commandes seront valables une fois l'application et le Dockerfile terminés.

En local :

```bash
pip install -r requirements.txt
python app.py
```

Avec Docker :

```bash
docker build -f flask.dockerfile -t apartment-hunter .
docker run -p 5000:5000 apartment-hunter
```

L'application est ensuite accessible sur <http://localhost:5000>.
