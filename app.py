from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file
)

from datetime import date
import pandas as pd
import numpy as np
import joblib
import os

from werkzeug.utils import secure_filename

from config import MODEL_PATH, KEY

from utils import (
    create_sections,
    get_feature_types,
    FEATURE_CONFIG
)
# ==========================
# Flask Configuration
# ==========================

app = Flask(__name__)
app.secret_key = KEY

UPLOAD_FOLDER = "uploads"
PREDICTION_FOLDER = "predictions"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    PREDICTION_FOLDER,
    exist_ok=True
)

# ==========================
# Load Model
# ==========================
try:
    model = joblib.load(MODEL_PATH)
    print("Model Loaded Successfully")
except Exception as e:
    raise RuntimeError(
        f"Model loading failed: {e}"
    )
    
# ==========================
# Home Page
# ==========================
@app.route("/")
def home():
    numerical_features, categorical_features = get_feature_types()
    sections = create_sections()
    return render_template(
        "index.html",
        sections=sections,
        feature_config=FEATURE_CONFIG,
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )
columns_to_drop = [
    # After HouseAge, RemodelAge, GarageAge
    "YrSold",
    "YearBuilt",
    "YearRemodAdd",
    "GarageYrBlt",

    # After TotalBathrooms
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",

    # After TotalSF
    "1stFlrSF",
    "2ndFlrSF",

    # Basement area components
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "LowQualFinSF",

    # After TotalPorchArea
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",

    # After HasBasement
    "TotalBsmtSF"
]
    
# ==========================
# Single Prediction
# ==========================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
    data = {}
    numerical_features, categorical_features = get_feature_types()
    sections = create_sections()
    # --------------------------
    # Numerical Features
    # --------------------------
    for feature in numerical_features:
        value = request.form.get(feature)
        if value == "" or value is None:
            data[feature] = np.nan
        else:
            data[feature] = float(value)
    # --------------------------
    # Categorical Features
    # --------------------------
    for feature in categorical_features:
        value = request.form.get(feature)
        if value == "":
            value = np.nan
        data[feature] = value
    # Create DataFrame
    input_df = pd.DataFrame(
        [data]
    )
    # Engineered Features
    input_df["HouseAge"] = date.today().year - input_df["YearBuilt"]
    input_df["RemodelAge"] = date.today().year - input_df["YearRemodAdd"]

    if pd.isna(input_df["GarageYrBlt"].iloc[0]):
        input_df["GarageAge"] = input_df["HouseAge"]
    else:
        input_df["GarageAge"] = date.today().year - input_df["GarageYrBlt"]

    input_df["TotalBathrooms"] = (
        input_df["FullBath"]
        + 0.5 * input_df["HalfBath"]
        + input_df["BsmtFullBath"]
        + 0.5 * input_df["BsmtHalfBath"]
    )

    input_df["TotalSF"] = (
        input_df["GrLivArea"]
        + input_df["TotalBsmtSF"]
    )

    input_df["TotalPorchArea"] = (
        input_df["OpenPorchSF"]
        + input_df["EnclosedPorch"]
        + input_df["3SsnPorch"]
        + input_df["ScreenPorch"]
    )

    input_df["HasGarage"] = (
        input_df["GarageCars"] > 0
    ).astype(int)

    input_df["HasBasement"] = (
        input_df["TotalBsmtSF"] > 0
    ).astype(int)

    input_df["HasFireplace"] = (
        input_df["Fireplaces"] > 0
    ).astype(int)
    input_df.drop(columns=columns_to_drop, inplace=True)
    try:
        # Maintain training feature order
        input_df = input_df[
            model.feature_names_in_
        ]
        prediction = model.predict(
            input_df
        )
        # Reverse log transformation
        # Remove if model was trained without log1p
        prediction = np.expm1(
            prediction
        )[0]
    except Exception as e:
        flash(
            f"Prediction failed: {e}",
            "danger"
        )

        return render_template(
            "index.html",
            sections=sections,
            feature_config=FEATURE_CONFIG,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )
    return render_template(
        "index.html",
        prediction=round(prediction,2),
        sections=sections,
        feature_config=FEATURE_CONFIG,
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

# ==========================
# Batch Prediction
# ==========================

@app.route(
    "/batch_predict",
    methods=["POST"]
)
def batch_predict():
    if "file" not in request.files:
        flash(
            "No file uploaded",
            "warning"
        )

        return redirect(
            url_for("home")
        )

    file = request.files["file"]
    if file.filename == "":
        flash(
            "No file selected",
            "warning"
        )

        return redirect(
            url_for("home")
        )

    filename = secure_filename(
        file.filename
    )

    if not filename.endswith(".csv"):
        return "Only CSV files allowed"
    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(
        upload_path
    )
    try:
        df = pd.read_csv(
            upload_path
        )
        df["HouseAge"] = date.today().year - df["YearBuilt"]
        df["RemodelAge"] = date.today().year - df["YearRemodAdd"]

        if pd.isna(df["GarageYrBlt"].iloc[0]):
            df["GarageAge"] = df["HouseAge"]
        else:
            df["GarageAge"] = date.today().year - df["GarageYrBlt"]

        df["TotalBathrooms"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

        df["TotalSF"] = (
            df["GrLivArea"]
            + df["TotalBsmtSF"]
        )

        df["TotalPorchArea"] = (
            df["OpenPorchSF"]
            + df["EnclosedPorch"]
            + df["3SsnPorch"]
            + df["ScreenPorch"]
        )

        df["HasGarage"] = (
            df["GarageCars"] > 0
        ).astype(int)

        df["HasBasement"] = (
            df["TotalBsmtSF"] > 0
        ).astype(int)

        df["HasFireplace"] = (
            df["Fireplaces"] > 0
        ).astype(int)
        df.drop(columns=columns_to_drop)
        expected_columns = list(
            model.feature_names_in_
        )
        missing_columns = (
            set(expected_columns)
            -
            set(df.columns)
        )
        if missing_columns:
            return (
                f"Missing Columns: "
                f"{missing_columns}"
            )
        # Maintain training order
        df = df[
            expected_columns
        ]
        prediction = model.predict(
            df
        )
        # Reverse log transformation
        prediction = np.expm1(
            prediction
        )
        df["PredictedPrice"] = prediction
        output_path = os.path.join(
            PREDICTION_FOLDER,
            "prediction.csv"
        )
        df.to_csv(
            output_path,
            index=False
        )
        return send_file(
            output_path,
            as_attachment=True
        )
    except Exception as e:
        return (
            f"Batch Prediction Failed: {e}"
        )

@app.route("/about")
def about():
    return render_template("about.html")

# ==========================
# Run Application
# ==========================
if __name__ == "__main__":
    app.run(
        debug=True
    )