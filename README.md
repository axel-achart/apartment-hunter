# apartment-hunter

🇬🇧 English version · [🇫🇷 Version française](#apartment-hunter-1)

Project to predict real estate prices.

The model estimates the price of a house from its characteristics (living area, number of bedrooms, number of floors, quality, location, etc.). Several regression algorithms are trained and compared, and the best one is deployed in a Flask application, containerized with Docker.

The chosen dataset is **King County** (`kc_house_data.csv`, 21,613 sales). The Madrid files in `data/raw/` were not used.

## Project structure

| File / folder | Purpose |
| --- | --- |
| `data/raw/` | Raw data (King County and Madrid) and their descriptions |
| `data/clean/` | Cleaned data (`kc_house_data_clean.csv`) |
| `data/images/` | Diagrams illustrating the regression algorithms |
| `notebook.ipynb` | Data cleaning and first visualizations |
| `analyse.pbix` | Exploratory analysis done with Power BI |
| `notebook_conclusion.ipynb` | Documentation of the regression algorithms (with diagrams) |
| `app.py` | Flask application (to be completed) |
| `flask.dockerfile` | Dockerfile of the application (to be completed) |
| `requirements.txt` | Python dependencies |

## Progress

| Step | Status |
| --- | --- |
| Data cleaning | Being finalized |
| Exploratory analysis (Power BI) | Done |
| Algorithm documentation | Done (`notebook_conclusion.ipynb`) |
| Feature selection | To do |
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

## Data cleaning

Cleaning is done in `notebook.ipynb`:

- **Date**: converted to a date type, then dropped. Sales only cover 2014-2015, so its impact on the price is very low.
- **Missing values**: none in the dataset, so no imputation is needed.
- **Duplicates**: detected and removed.
- **`id`**: dropped, it is an identifier with no predictive value.
- **Date consistency**: no house is renovated before being built, so `yr_built` and `yr_renovated` are kept.
- **Units**: areas in square feet (`sqft_*`) are converted to square meters (`sqm_*`).
- **Outliers**: detected with three methods (IQR, modified Z-score, percentiles). The treatment (keep or remove) and its justification are still to be finalized.
- **Standardization**: done inside the modeling pipeline, after the train / test split, to avoid data leakage.

The cleaned file is saved in `data/clean/kc_house_data_clean.csv`.

## Regression algorithms

The description of eight algorithms, with one diagram each, is in `notebook_conclusion.ipynb`: linear, multiple linear, polynomial, Poisson, SVR, Gradient Boosting, Random Forest and Elastic Net regressions.

The three algorithms chosen for prediction are:

- Support Vector Regression (SVR)
- Gradient Boosting
- Random Forest

## Feature selection

### Target

The variable to predict is `y = price`.

### Excluded variables

- `price`: it is the target, it cannot be used as an explanatory variable.
- `price_per_living_sqft`, `price_per_lot_sqft`, `sqm_price_per_living`, `sqm_price_per_lot`: they are computed from the price, so they would give the answer to the model.
- The original `sqft_*` columns: they are replaced by their `sqm_*` equivalents.

### Candidate features

| Group | Variables |
| --- | --- |
| Housing characteristics | `bedrooms`, `bathrooms`, `floors`, `sqm_living`, `sqm_lot`, `sqm_above`, `sqm_basement` |
| Quality | `grade`, `condition`, `view`, `waterfront` |
| Age / renovation | `yr_built`, `yr_renovated` |
| Location | `zipcode`, `lat`, `long` |
| Neighborhood | `sqm_living15`, `sqm_lot15` |

### Location configurations

`zipcode`, `lat` and `long` all describe the location. Two configurations will be tested:

- Model 1: `lat`, `long`
- Model 2: `lat`, `long`, `zipcode`

The planned selection methods are correlation, Random Forest feature importance, Boruta and forward feature selection.

## Modeling pipelines

`zipcode` is treated as a categorical variable (One-Hot Encoding) and not as a number. Numerical variables are standardized for the SVR. Tree-based models do not need it.

SVR pipeline:

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
                     SVR
                      │
                      ▼
                 Prediction
```

Random Forest and Gradient Boosting pipeline:

```text
                    Data
                     │
                     ▼
              Train / Test Split
                     │
                     ▼
               Preprocessing
                     │
                     ▼
        Random Forest / Gradient Boosting
                     │
                     ▼
                 Prediction
```

## Evaluation

The models are compared with three metrics:

- **MAE**: mean absolute error, in dollars.
- **RMSE**: root mean squared error, more sensitive to large errors.
- **R²**: share of the price variance explained by the model.

Results will be added here once the models are trained.

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

Projet de prédiction du prix de biens immobiliers.

Le modèle estime le prix d'une maison à partir de ses caractéristiques (surface, nombre de chambres, nombre d'étages, qualité, localisation, etc.). Plusieurs algorithmes de régression sont entraînés puis comparés, et le meilleur est déployé dans une application Flask, conteneurisée avec Docker.

Le jeu de données choisi est celui de **King County** (`kc_house_data.csv`, 21 613 ventes). Les fichiers Madrid présents dans `data/raw/` n'ont pas été utilisés.

## Structure du projet

| Fichier / dossier | Rôle |
| --- | --- |
| `data/raw/` | Données brutes (King County et Madrid) et leurs descriptions |
| `data/clean/` | Données nettoyées (`kc_house_data_clean.csv`) |
| `data/images/` | Schémas illustrant les algorithmes de régression |
| `notebook.ipynb` | Nettoyage des données et premières visualisations |
| `analyse.pbix` | Analyse exploratoire réalisée avec Power BI |
| `notebook_conclusion.ipynb` | Documentation des algorithmes de régression (avec schémas) |
| `app.py` | Application Flask (à compléter) |
| `flask.dockerfile` | Dockerfile de l'application (à compléter) |
| `requirements.txt` | Dépendances Python |

## Avancement

| Étape | État |
| --- | --- |
| Nettoyage des données | En cours de finalisation |
| Analyse exploratoire (Power BI) | Fait |
| Documentation des algorithmes | Faite (`notebook_conclusion.ipynb`) |
| Sélection des features | À faire |
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

## Nettoyage des données

Le nettoyage est fait dans `notebook.ipynb` :

- **Date** : convertie en type date, puis supprimée. Les ventes ne couvrent que 2014-2015, son impact sur le prix est très faible.
- **Valeurs manquantes** : aucune dans le jeu de données, aucune imputation n'est nécessaire.
- **Doublons** : détectés et supprimés.
- **`id`** : supprimé, c'est un identifiant sans valeur prédictive.
- **Cohérence des dates** : aucune maison n'est rénovée avant sa construction, `yr_built` et `yr_renovated` sont conservées.
- **Unités** : les surfaces en pieds carrés (`sqft_*`) sont converties en mètres carrés (`sqm_*`).
- **Valeurs aberrantes** : détectées avec trois méthodes (IQR, Z-score modifié, percentiles). Le traitement (conservation ou suppression) et sa justification restent à finaliser.
- **Standardisation** : réalisée dans le pipeline de modélisation, après la séparation train / test, pour éviter toute fuite de données.

Le fichier nettoyé est enregistré dans `data/clean/kc_house_data_clean.csv`.

## Algorithmes de régression

La description de huit algorithmes, avec un schéma pour chacun, se trouve dans `notebook_conclusion.ipynb` : régression linéaire, linéaire multiple, polynomiale, Poisson, SVR, Gradient Boosting, Random Forest et Elastic Net.

Les trois algorithmes retenus pour la prédiction sont :

- Support Vector Regression (SVR)
- Gradient Boosting
- Random Forest

## Sélection des features

### Cible

La variable à prédire est `y = price`.

### Variables exclues

- `price` : c'est la cible, elle ne peut pas servir de variable explicative.
- `price_per_living_sqft`, `price_per_lot_sqft`, `sqm_price_per_living`, `sqm_price_per_lot` : elles sont calculées à partir du prix, elles révéleraient la réponse au modèle.
- Les colonnes `sqft_*` d'origine : elles sont remplacées par leurs équivalents `sqm_*`.

### Features candidates

| Groupe | Variables |
| --- | --- |
| Caractéristiques du logement | `bedrooms`, `bathrooms`, `floors`, `sqm_living`, `sqm_lot`, `sqm_above`, `sqm_basement` |
| Qualité | `grade`, `condition`, `view`, `waterfront` |
| Ancienneté / rénovation | `yr_built`, `yr_renovated` |
| Localisation | `zipcode`, `lat`, `long` |
| Environnement proche | `sqm_living15`, `sqm_lot15` |

### Configurations de localisation

`zipcode`, `lat` et `long` décrivent tous la localisation. Deux configurations seront testées :

- Modèle 1 : `lat`, `long`
- Modèle 2 : `lat`, `long`, `zipcode`

Les méthodes de sélection prévues sont la corrélation, l'importance des variables d'un Random Forest, Boruta et la forward feature selection.

## Pipelines de modélisation

`zipcode` est traité comme une variable catégorielle (One-Hot Encoding) et non comme un nombre. Les variables numériques sont standardisées pour le SVR. Les modèles à base d'arbres n'en ont pas besoin.

Pipeline SVR :

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
                     SVR
                      │
                      ▼
                 Prédiction
```

Pipeline Random Forest et Gradient Boosting :

```text
                    Data
                     │
                     ▼
              Train / Test Split
                     │
                     ▼
              Pré-traitement
                     │
                     ▼
        Random Forest / Gradient Boosting
                     │
                     ▼
                 Prédiction
```

## Évaluation

Les modèles sont comparés avec trois métriques :

- **MAE** : erreur absolue moyenne, en dollars.
- **RMSE** : racine de l'erreur quadratique moyenne, plus sensible aux grosses erreurs.
- **R²** : part de la variance du prix expliquée par le modèle.

Les résultats seront ajoutés ici une fois les modèles entraînés.

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
