import numpy as np
import pandas as pd
import joblib # Permet d'enregistrer et de recharger le modèle entraîné
from sklearn.model_selection import train_test_split, GridSearchCV #Sépare les données / cherche les meilleurs paramètres
from sklearn.pipeline import Pipeline# Permet de créer un pipeline de transformations et de modèle
from sklearn.preprocessing import StandardScaler# Permet de standardiser les features
from sklearn.svm import SVR# Modèle de régression par machines à vecteurs de support
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor #Modèle de régression gb & rf

#Modèle de régression SVR pour prédire le prix des maisons

#Etape 1 import des donées
# Données nettoyées
print("Modèle SVR")
df = pd.read_csv("data/clean/kc_house_data_clean.csv")

# Les 7 features retenues à la sélection
final_features = ["grade", "sqm_living", "sqm_lot", "waterfront", "view", "lat", "long"]

X = df[final_features]
y = np.log(df["price"])  # cible : log du prix

# Même split que dans notebook.ipynb
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Etape 2 création du pipeline SVR avec standardisation des features
# Création du pipeline SVR avec standardisation des features

svr = Pipeline([
    ("scaler", StandardScaler()),#Il remplace chaque valeur par (valeur - moyenne) / écart-type. Toutes les colonnes se retrouvent à la même échelle. 
    ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1)),#kernel="rbf" : la forme de la courbe.
    # C=1.0 : la sévérité envers les erreurs qui sortent du tube.
    # epsilon=0.1 : la largeur du tube autour de la prédiction où les erreurs ne sont pas pénalisées.
])

# Paramètres utilisés avant optimisation
svr_model = svr.named_steps["svr"]
print("Paramètres du SVR :")
print("  kernel  :", svr_model.kernel, "-> la forme de la courbe (rbf = courbe souple)")
print("  C       :", svr_model.C, "-> la sévérité envers les erreurs (grand = colle plus aux données)")
print("  epsilon :", svr_model.epsilon, "-> la marge d'erreur tolérée sans pénalité")
print("  gamma   :", svr_model.gamma, "-> la souplesse de la courbe (grand = risque de surapprentissage)")

#Etape 3 entraînement du modèle
svr.fit(X_train, y_train)

y_pred_log = svr.predict(X_test)

y_pred = np.exp(y_pred_log)   # prédiction en $
y_true = np.exp(y_test)       # vrai prix en $

#Etape 4 évaluation du modèle

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_test, y_pred_log)   # R² sur le log, comme dans la sélection des features

print("MAE  :", round(mae), "$")
print("RMSE :", round(rmse), "$")
print("R²   :", round(r2, 4))

"""********************************************"""

"""Modèle de régression Random Forest"""

"""Données déjà importées as *df[final features]*"""

print("Modèle Random Forest")

# On garde le même split que le SVR pour comparer les modèles sur les mêmes maisons
rf= RandomForestRegressor(random_state=42)

# Paramètres utilisés avant optimisation (valeurs par défaut de scikit-learn)
print("Paramètres du Random Forest :")
print("  n_estimators     :", rf.n_estimators, "-> le nombre d'arbres dans la forêt")
print("  max_depth        :", rf.max_depth, "-> la profondeur max d'un arbre (None = pas de limite)")
print("  min_samples_leaf :", rf.min_samples_leaf, "-> le nombre minimum de maisons dans une feuille")
print("  max_features     :", rf.max_features, "-> la part des features testées à chaque question (1.0 = toutes)")

rf.fit(X_train,y_train)

y_pred_log = rf.predict(X_test)
y_pred = np.exp(y_pred_log)
y_true = np.exp(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_test,y_pred_log)

print("MAE  :", round(mae), "$")
print("RMSE :", round(rmse), "$")
print("R²   :", round(r2, 4))


"""********************************************"""

"""Modèle de régression Gradient Boosting"""

"""Données déjà importées as *df[final features]*"""

print("Modèle Gradient Boosting")

# Même split que les deux autres modèles
gb = GradientBoostingRegressor(random_state=42)

# Paramètres utilisés avant optimisation (valeurs par défaut de scikit-learn)
print("Paramètres du Gradient Boosting :")
print("  n_estimators  :", gb.n_estimators, "-> le nombre d'arbres, construits l'un après l'autre")
print("  learning_rate :", gb.learning_rate, "-> la taille de la correction apportée par chaque arbre")
print("  max_depth     :", gb.max_depth, "-> la profondeur max de chaque arbre")
print("  subsample     :", gb.subsample, "-> la part des maisons vue par chaque arbre (1.0 = toutes)")

gb.fit(X_train,y_train)

y_pred_log = gb.predict(X_test)
y_pred = np.exp(y_pred_log)
y_true = np.exp(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_test,y_pred_log)

print("MAE  :", round(mae), "$")
print("RMSE :", round(rmse), "$")
print("R²   :", round(r2, 4))


"""********************************************"""

"""Optimisation des paramètres avec Grid Search"""

# Le Grid Search teste toutes les combinaisons de la grille en cross-validation
# sur le train set uniquement : le test set ne sert qu'à la note finale.

def evaluate(model):
    # On remet les prédictions en $ pour MAE et RMSE, le R² reste sur le log
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_test)
    print("MAE  :", round(mean_absolute_error(y_true, y_pred)), "$")
    print("RMSE :", round(np.sqrt(mean_squared_error(y_true, y_pred))), "$")
    print("R²   :", round(r2_score(y_test, y_pred_log), 4))


# SVR : petite grille, car le SVR est lent sur 17 000 maisons
# Le préfixe "svr__" indique que le paramètre appartient à l'étape "svr" du pipeline
print("\nGrid Search SVR")
svr_grid = {
    "svr__C": [1, 10, 100],
    "svr__epsilon": [0.05, 0.1],
    "svr__gamma": ["scale", 0.1],
}
svr_search = GridSearchCV(svr, svr_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
svr_search.fit(X_train, y_train)
print("Meilleurs paramètres :", svr_search.best_params_)
print("MAE en cross-validation (log) :", round(-svr_search.best_score_, 4))
evaluate(svr_search.best_estimator_)


# Random Forest : on teste la taille de la forêt et la profondeur des arbres
print("\nGrid Search Random Forest")
rf_grid = {
    "n_estimators": [100, 300],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 3],
}
rf_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
rf_search.fit(X_train, y_train)
print("Meilleurs paramètres :", rf_search.best_params_)
print("MAE en cross-validation (log) :", round(-rf_search.best_score_, 4))
evaluate(rf_search.best_estimator_)


# Gradient Boosting : plus d'arbres avec un learning_rate plus petit est souvent meilleur
print("\nGrid Search Gradient Boosting")
gb_grid = {
    "n_estimators": [200, 500],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
}
gb_search = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
gb_search.fit(X_train, y_train)
print("Meilleurs paramètres :", gb_search.best_params_)
print("MAE en cross-validation (log) :", round(-gb_search.best_score_, 4))
evaluate(gb_search.best_estimator_)


"""********************************************"""

"""Enregistrement du meilleur modèle"""

# On choisit le modèle avec la plus petite erreur (MAE) en cross-validation,
# le test set ne sert pas à choisir pour ne pas biaiser la note finale
searches = {
    "SVR": svr_search,
    "Random Forest": rf_search,
    "Gradient Boosting": gb_search,
}
# scikit-learn renvoie la MAE en négatif : le plus grand score est donc la plus petite erreur
best_name = "SVR"
for name in searches:
    if searches[name].best_score_ > searches[best_name].best_score_:
        best_name = name
best_model = searches[best_name].best_estimator_
print("\nMeilleur modèle :", best_name)

# Même format que la branche feat/axel : le modèle seul, que l'app Flask charge directement.
# Il attend les 7 features dans l'ordre de final_features et prédit log(price) :
# l'app doit appliquer exp() pour avoir un prix en $
joblib.dump(best_model, "models/best_model.joblib")
print("Modèle enregistré dans models/best_model.joblib")

# Vérification : le modèle rechargé doit donner exactement les mêmes scores
saved_model = joblib.load("models/best_model.joblib")
print("Vérification du modèle rechargé")
evaluate(saved_model)