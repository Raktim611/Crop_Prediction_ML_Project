import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")

app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "crop_model.pkl not found. Run `python train_model.py` first."
    )
    
    

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
MODEL_NAME = bundle["model_name"]
MODEL_ACCURACY = bundle["test_accuracy"]


CROP_INFO = {
    "apple": ("🍎", "Apple", "A fruit crop that generally prefers well-drained soil and a cool-to-moderate climate."),
    "banana": ("🍌", "Banana", "A tropical crop that generally performs well with warm temperatures, humidity and adequate water."),
    "blackgram": ("🌱", "Black Gram", "A pulse crop commonly grown in warm conditions with suitable moisture."),
    "chickpea": ("🫘", "Chickpea", "A pulse crop that generally prefers relatively cool and dry growing conditions."),
    "coconut": ("🥥", "Coconut", "A tropical crop that benefits from warm conditions, humidity and substantial rainfall."),
    "coffee": ("☕", "Coffee", "A plantation crop associated with warm, humid conditions and adequate rainfall."),
    "cotton": ("🌿", "Cotton", "A warm-season crop that needs suitable temperature and soil nutrients."),
    "grapes": ("🍇", "Grapes", "A fruit crop that benefits from suitable temperature, soil conditions and controlled moisture."),
    "jute": ("🌾", "Jute", "A fiber crop that generally favors warm, humid conditions and sufficient rainfall."),
    "kidneybeans": ("🫘", "Kidney Beans", "A legume crop that grows under suitable temperature, moisture and nutrient conditions."),
    "lentil": ("🌱", "Lentil", "A pulse crop generally suited to cooler growing conditions and moderate moisture."),
    "maize": ("🌽", "Maize", "A cereal crop requiring appropriate nutrients, temperature and water availability."),
    "mango": ("🥭", "Mango", "A tropical fruit crop that generally prefers warm conditions and suitable soil moisture."),
    "mothbeans": ("🌱", "Moth Beans", "A drought-tolerant pulse crop commonly associated with warm and relatively dry conditions."),
    "mungbean": ("🌱", "Mung Bean", "A short-duration pulse crop that generally prefers warm conditions and adequate moisture."),
    "muskmelon": ("🍈", "Muskmelon", "A warm-season fruit crop that prefers suitable temperature, soil and moisture."),
    "orange": ("🍊", "Orange", "A citrus fruit crop requiring suitable soil conditions, temperature and moisture."),
    "papaya": ("🥭", "Papaya", "A tropical fruit crop that generally prefers warm temperatures and adequate moisture."),
    "pigeonpeas": ("🌱", "Pigeon Peas", "A pulse crop that can perform well under warm conditions with suitable rainfall."),
    "pomegranate": ("🍎", "Pomegranate", "A fruit crop that can tolerate relatively dry conditions when soil and climate are suitable."),
    "rice": ("🌾", "Rice", "A major cereal crop generally associated with warm temperatures, humidity and substantial water availability."),
    "watermelon": ("🍉", "Watermelon", "A warm-season fruit crop that benefits from adequate heat, suitable soil and moisture.")
}


def validate_number(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid value for {name}.")


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        prediction=None,
        error=None,
        model_name=MODEL_NAME,
        model_accuracy=MODEL_ACCURACY
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {
            feature: validate_number(request.form.get(feature), feature)
            for feature in FEATURES
        }

        input_df = pd.DataFrame([values], columns=FEATURES)

        prediction = model.predict(input_df)[0]
        probabilities = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            pairs = sorted(
                zip(model.classes_, probs),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            probabilities = [
                {"crop": crop, "probability": round(float(prob) * 100, 2)}
                for crop, prob in pairs
            ]

        icon, display_name, description = CROP_INFO.get(
            str(prediction).lower(),
            ("🌱", str(prediction).title(), "The model recommends this crop based on the supplied conditions.")
        )

        return render_template(
            "index.html",
            prediction=display_name,
            icon=icon,
            description=description,
            probabilities=probabilities,
            submitted=values,
            error=None,
            model_name=MODEL_NAME,
            model_accuracy=MODEL_ACCURACY
        )

    except ValueError as exc:
        return render_template(
            "index.html",
            prediction=None,
            error=str(exc),
            model_name=MODEL_NAME,
            model_accuracy=MODEL_ACCURACY
        ), 400


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "test_accuracy": round(float(MODEL_ACCURACY), 6)
    }


if __name__ == "__main__":
    app.run(debug=True)
