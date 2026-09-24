# apartment-hunter

Construire un modèle qui estime le prix de biens immobiliers. Entraîné avec plusieurs modèle de prédiction.
A partir de plusieurs facteurs comme la superficie, nombre d'étage, nombre de salle, rue et ville etc...
The choice was made to pick the King county files.

#### The *Data* folder is a folder that has both the raw and cleaned .csv files

#### **app.py** is the Flask app file that when launched build the app in the browser.

#### **flask.dockerfile** is the dockerfile that enables the dockerisation of the app.

#### **.gitignore** is the classic git file.

#### **notebook_conclusion.ipynb** is the final jupyter notebook file that converges our research and development.

#### **notebook.ipnyb** is the jupyter file that has the EDA.

#### **README.md** is the file that explains our project.

-------------------------------------------------------------------------------------------------------------------------------------

                    DATASET
                       │
                       ▼
              Analyse exploratoire
                   Power BI
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Prix ↔      Prix ↔       Prix ↔
      Surface      Qualité    Localisation
          │            │            │
          └────────────┼────────────┘
                       ▼
              Sélection des features
                       │
                       ▼
              Préparation des données
                       │
                       ▼
                 Train / Test Split
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
         Elastic Net   RF    Gradient Boosting
              │        │        │
              └────────┼────────┘
                       ▼
                Comparaison
             MAE / RMSE / R²

Première étape : définir clairement la cible
X→price

Les variables à exclure immédiatement:

price
bedrooms
bathrooms
floors
waterfront
view
condition
grade
yr_built
yr_renovated
zipcode
lat
long
sqm_living
sqm_lot
sqm_basement
sqm_living15
sqm_price_per_lot
sqm_price_per_living
sqm_lot15
sqm_above

Les features candidates:

candidate_features = [
    "bedrooms",
    "bathrooms",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqm_living",
    "sqm_lot",
    "sqm_basement",
    "sqm_living15",
    "sqm_lot15",
    "sqm_above"
]

Notre première sélection pourrait être:

Caractéristiques du logement:
bedrooms
bathrooms
floors
sqm_living
sqm_lot
sqm_above
sqm_basement

Qualité

grade
condition
view
waterfront

Ancienneté / rénovation

yr_built
yr_renovated

Localisation

zipcode
lat
long

Environnement proche

sqm_living15
sqm_lot15

Attention à zipcode:

on a:

zipcode
lat
long

on testera donc deux configurations

Modèle 1 
lat
long

Modele 2
lat /lon/ zipcode

13. Pipeline Elastic Net

Pour Elastic Net, les variables numériques doivent généralement être standardisées.

Et zipcode doit être traité comme une variable catégorielle, pas comme un nombre continu.

Conceptuellement :

                    Data
                     │
                     ▼
              Train / Test Split
                     │
                     ▼
              ColumnTransformer
             ┌────────┴────────┐
             │                 │
        Numerical          Categorical
             │                 │
        StandardScaler    OneHotEncoder
             │                 │
             └────────┬────────┘
                      ▼
                 ElasticNet
                      │
                      ▼
                  Prediction

Pipeline Random Forest

Pour Random Forest :
                    Data
                     │
                     ▼
              Train / Test Split
                     │
                     ▼
              Preprocessing
                     │
                     ▼
               Random Forest
                     │
                     ▼
                  Prediction

Pipeline Gradient Boosting

Même logique :
                    Data
                     │
                     ▼
              Train / Test Split
                     │
                     ▼
              Preprocessing
                     │
                     ▼
             Gradient Boosting
                     │
                     ▼
                  Prediction