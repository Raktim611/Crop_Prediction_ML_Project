# CropSense — Crop Recommendation System

A complete machine-learning crop recommendation web application built with:

- Python
- Pandas
- Scikit-learn
- Flask
- HTML/CSS

## Dataset

`dataset/Crop_recommendation.csv`

Features:

- N — Nitrogen
- P — Phosphorus
- K — Potassium
- temperature
- humidity
- ph
- rainfall

Target:

- label — recommended crop

## Project structure

```text
crop_recommendation_app/
├── app.py
├── train_model.py
├── requirements.txt
├── dataset/
│   └── Crop_recommendation.csv
├── model/
│   ├── crop_model.pkl
│   └── model_metrics.txt
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use:

```powershell
venv\Scripts\activate
```

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## 3. Train the model

Run:

```powershell
python train_model.py
```

The script:

1. Loads the CSV.
2. Validates required columns.
3. Splits data into train/test sets.
4. Trains seven classification algorithms.
5. Compares their test accuracy.
6. Selects the best model automatically.
7. Saves it to `model/crop_model.pkl`.
8. Saves evaluation information to `model/model_metrics.txt`.

## 4. Start Flask

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 5. Health check

Open:

```text
http://127.0.0.1:5000/health
```

It returns the model name and test accuracy as JSON.

## Deployment on Render

Use:

**Build Command**
```text
pip install -r requirements.txt
```

**Start Command**
```text
gunicorn app:app
```

Make sure `model/crop_model.pkl` is committed to GitHub, or alternatively change the deployment workflow to train the model during the build.

## Important ML note

The model is a decision-support system based on the variables contained in the supplied dataset. A high benchmark accuracy on this dataset does not guarantee the same accuracy on real-world farms. Real deployment should consider local agronomic knowledge, field measurements, season, location, crop variety and current weather.
