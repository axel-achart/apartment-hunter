# Flask application: estimate the price of a house in King County (Seattle area)

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.joblib"))

FEATURES = ["grade", "sqm_living", "sqm_lot", "waterfront", "view", "lat", "long"]

DEFAULT_VALUES = {
    "grade": "7",
    "sqm_living": "184",
    "sqm_lot": "683",
    "waterfront": "0",
    "view": "0",
    "lat": "47.7088",
    "long": "-122.202",
}


@app.route("/", methods=["GET", "POST"])
def index():
    values = dict(DEFAULT_VALUES)
    price = None
    error = None

    if request.method == "POST":
        for feature in FEATURES:
            values[feature] = request.form.get(feature, "").strip()

        try:
            house = pd.DataFrame([{feature: float(values[feature]) for feature in FEATURES}])
            log_price = model.predict(house)[0]
            price = f"{np.exp(log_price):,.0f}"
        except ValueError:
            error = "Every field must contain a number (use a dot for decimals: 47.57)."

    return render_template("index.html", values=values, price=price, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split #Sépare les données en ensembles d'entraînement et de test
# from sklearn.pipeline import Pipeline# Permet de créer un pipeline de transformations et de modèle
# from sklearn.preprocessing import StandardScaler# Permet de standardiser les features
# from sklearn.svm import SVR# Modèle de régression par machines à vecteurs de support
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor #Modèle de régression gb & rf
#
# #Modèle de régression SVR pour prédire le prix des maisons
#
# #Etape 1 import des donées
# # Données nettoyées
# print("Modèle SVR")
# df = pd.read_csv("data/clean/kc_house_data_clean.csv")
#
# # Les 7 features retenues à la sélection
# final_features = ["grade", "sqm_living", "sqm_lot", "waterfront", "view", "lat", "long"]
#
# X = df[final_features]
# y = np.log(df["price"])  # cible : log du prix
#
# # Même split que dans notebook.ipynb
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#
# #Etape 2 création du pipeline SVR avec standardisation des features
# # Création du pipeline SVR avec standardisation des features
#
# svr = Pipeline([
#     ("scaler", StandardScaler()),#Il remplace chaque valeur par (valeur - moyenne) / écart-type. Toutes les colonnes se retrouvent à la même échelle. 
#     ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1)),#kernel="rbf" : la forme de la courbe.
#     # C=1.0 : la sévérité envers les erreurs qui sortent du tube.
#     # epsilon=0.1 : la largeur du tube autour de la prédiction où les erreurs ne sont pas pénalisées.
# ])
#
# #Etape 3 entraînement du modèle
# svr.fit(X_train, y_train)
#
# y_pred_log = svr.predict(X_test)
#
# y_pred = np.exp(y_pred_log)   # prédiction en $
# y_true = np.exp(y_test)       # vrai prix en $
#
# #Etape 4 évaluation du modèle
#
# mae = mean_absolute_error(y_true, y_pred)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# r2 = r2_score(y_test, y_pred_log)   # R² sur le log, comme dans la sélection des features
#
# print("MAE  :", round(mae), "$")
# print("RMSE :", round(rmse), "$")
# print("R²   :", round(r2, 4))
#
# --------------------------------------------
#
# Modèle de régression Random Forest
#
# Données déjà importées as *df[final features]*
#
# print("Modèle Random Forest")
#
# X = df[final_features]
# y = np.log(df["price"])
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#
# rf= RandomForestRegressor()
# rf.fit(X_train,y_train)
#
# y_pred_log = rf.predict(X_test)
# y_pred = np.exp(y_pred_log)
# y_true = np.exp(y_test)
#
# mae = mean_absolute_error(y_true, y_pred)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# r2 = r2_score(y_test,y_pred_log)
#
# print("MAE  :", round(mae), "$")
# print("RMSE :", round(rmse), "$")
# print("R²   :", round(r2, 4))
#
#
# --------------------------------------------
#
# Modèle de régression Gradient Boosting
#
# Données déjà importées as *df[final features]*
#
# print("Modèle Gradient Boosting")
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#
# gb = GradientBoostingRegressor()
# gb.fit(X_train,y_train)
#
# y_pred_log = gb.predict(X_test)
# y_pred = np.exp(y_pred_log)
# y_true = np.exp(y_test)
#
# mae = mean_absolute_error(y_true, y_pred)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# r2 = r2_score(y_test,y_pred_log)
#
# print("MAE  :", round(mae), "$")
# print("RMSE :", round(rmse), "$")
# print("R²   :", round(r2, 4))
