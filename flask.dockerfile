# Même version de Python que nos .venv
FROM python:3.13-slim

WORKDIR /app

# On installe les dépendances avec les mêmes versions que pour l'entraînement,
# sinon le modèle .joblib risque de ne pas se charger
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On copie seulement ce dont l'app a besoin pour prédire
COPY app.py .
COPY templates/ templates/
COPY models/ models/

# Port utilisé par Flask
EXPOSE 5000

CMD ["python", "app.py"]
