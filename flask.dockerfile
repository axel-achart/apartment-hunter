# Our python version
FROM python:3.13-slim

WORKDIR /app

# We install the dependencies using the same versions as those used for training;
# otherwise, the .joblib model might fail to load.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We only copy what the app needs to make predictions
COPY app.py .
COPY templates/ templates/
COPY models/ models/

# Port used by Flask
EXPOSE 5000

CMD ["python", "app.py"]