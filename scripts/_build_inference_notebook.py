"""One-off generator for notebooks/03_inference_demo.ipynb.

Not part of the regular pipeline -- run once to (re)create the notebook,
then it's committed and executed like the other two. Kept out of
scripts/*.sh since it's a notebook-authoring tool, not something the
lab run depends on.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        "# Inference Demo — Heart Disease Risk Classifier\n"
        "\n"
        "This notebook demonstrates loading the packaged model artifact "
        "(`models/heart_model.joblib` + `models/model_metadata.json`) and "
        "running inference directly -- no API server needed. It mirrors "
        "exactly what `src/heart_mlops/api/main.py`'s `/predict` endpoint "
        "does internally (same feature-column order, same `float64` cast, "
        "same threshold-based decision rule), so this is a faithful "
        "standalone demonstration of the serving path, not a different "
        "implementation of it.\n"
        "\n"
        "See `docs/DECISIONS.md` D5.1/D5.2 for why the joblib artifact "
        "(not the MLflow registry) is what actually ships."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "import json\n"
        "\n"
        "import joblib\n"
        "import pandas as pd\n"
        "\n"
        "from heart_mlops.config import (\n"
        "    CLEANED_DATA_PATH,\n"
        "    MODEL_FEATURE_COLUMNS,\n"
        "    MODELS_DIR,\n"
        "    PROJECT_ROOT,\n"
        ")\n"
        "\n"
        'pipeline = joblib.load(MODELS_DIR / "heart_model.joblib")\n'
        'metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text())\n'
        "\n"
        "print(f\"Model: {metadata['model_name']} (registry v{metadata['model_version']})\")\n"
        "print(f\"Operating threshold: {metadata['threshold']:.4f}\")\n"
        'print(f"Feature columns, in the exact order the pipeline expects: {MODEL_FEATURE_COLUMNS}")'
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## Single-record inference\n"
        "\n"
        "Using the same sample payloads the API's own container smoke test "
        "sends (`test/sample_requests/`), so this notebook and the live API "
        "are demonstrably exercising the identical model on identical input."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "def predict_one(record: dict) -> dict:\n"
        '    """Mirrors api/main.py\'s _features_to_row + _risk_to_response exactly."""\n'
        '    row = pd.DataFrame([record])[MODEL_FEATURE_COLUMNS].astype("float64")\n'
        "    risk_probability = float(pipeline.predict_proba(row)[0, 1])\n"
        '    threshold = metadata["threshold"]\n'
        "    prediction = int(risk_probability >= threshold)\n"
        '    label = "Disease" if prediction == 1 else "No disease"\n'
        "    confidence = risk_probability if prediction == 1 else 1.0 - risk_probability\n"
        "    return {\n"
        '        "prediction": prediction,\n'
        '        "label": label,\n'
        '        "confidence": round(confidence, 4),\n'
        '        "risk_probability": round(risk_probability, 4),\n'
        "    }\n"
        "\n"
        "\n"
        'sample_dir = PROJECT_ROOT / "test" / "sample_requests"\n'
        'for name in ["valid_high_risk.json", "valid_low_risk.json"]:\n'
        "    record = json.loads((sample_dir / name).read_text())\n"
        "    result = predict_one(record)\n"
        '    print(f"{name}: {result}")'
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## Batch inference on real patients, with ground truth\n"
        "\n"
        "A handful of real rows from the cleaned, combined 4-site dataset "
        "(fixed seed, so this is reproducible), run through the same "
        "pipeline in one vectorised call -- the same approach "
        "`POST /predict/batch` uses -- compared against their actual "
        "recorded diagnosis."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "df = pd.read_csv(CLEANED_DATA_PATH)\n"
        "sample = df.dropna(subset=MODEL_FEATURE_COLUMNS).sample(8, random_state=42)\n"
        "\n"
        'X = sample[MODEL_FEATURE_COLUMNS].astype("float64")\n'
        "risk_probabilities = pipeline.predict_proba(X)[:, 1]\n"
        'threshold = metadata["threshold"]\n'
        "predictions = (risk_probabilities >= threshold).astype(int)\n"
        "\n"
        'results = sample[["site", "target"]].copy()\n'
        'results["risk_probability"] = risk_probabilities.round(4)\n'
        'results["predicted"] = predictions\n'
        'results["correct"] = results["predicted"] == results["target"]\n'
        "results"
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## Takeaway\n"
        "\n"
        "This is the exact artifact and decision rule running in production "
        "(the Docker image, the Kubernetes deployment, and the live "
        "`/predict` endpoint all load this same `heart_model.joblib` and "
        'apply this same threshold) -- there is no separate "notebook '
        'version" of the model to drift out of sync with what actually '
        "serves traffic."
    )
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

with open("notebooks/03_inference_demo.ipynb", "w") as f:
    nbf.write(nb, f)

print("Wrote notebooks/03_inference_demo.ipynb")
