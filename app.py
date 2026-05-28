
import json
import gzip
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier


st.set_page_config(
    page_title="GV-SHR-SA-AKI Mortality Prediction",
    layout="wide"
)

st.title("GV-SHR-SA-AKI Mortality Prediction")
st.markdown(
    "Online prediction tool based on the final SHAP-simplified CatBoost model."
)


@st.cache_resource
def load_model():
    models_dir = Path("models")
    gz_path = models_dir / "simplified_catboost.cbm.gz"
    part_paths = sorted(models_dir.glob("simplified_catboost.cbm.gz.part*"))

    tmp_dir = Path(tempfile.mkdtemp())
    merged_gz_path = tmp_dir / "simplified_catboost.cbm.gz"
    cbm_path = tmp_dir / "simplified_catboost.cbm"

    if gz_path.exists():
        shutil.copyfile(gz_path, merged_gz_path)

    elif part_paths:
        with open(merged_gz_path, "wb") as merged:
            for part in part_paths:
                with open(part, "rb") as pf:
                    shutil.copyfileobj(pf, merged)

    else:
        raise FileNotFoundError(
            "No model file found. Expected simplified_catboost.cbm.gz or split part files."
        )

    with gzip.open(merged_gz_path, "rb") as f_in:
        with open(cbm_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    model = CatBoostClassifier()
    model.load_model(str(cbm_path))

    return model


model = load_model()

with open("models/selected_features.json", "r", encoding="utf-8") as f:
    selected_features = json.load(f)

with open("models/feature_defaults.json", "r", encoding="utf-8") as f:
    feature_defaults = json.load(f)


st.header("Single-patient prediction")

input_data = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(selected_features):
    default_value = float(feature_defaults.get(feature, 0.0))

    with col1 if i % 2 == 0 else col2:
        input_data[feature] = st.number_input(
            label=feature,
            value=default_value,
            format="%.6f"
        )

input_df = pd.DataFrame([input_data], columns=selected_features)

pred_proba = model.predict_proba(input_df)[0, 1]
threshold = 0.5
pred_class = int(pred_proba >= threshold)

if pred_proba < 0.2:
    risk_level = "Low risk"
elif pred_proba < 0.5:
    risk_level = "Intermediate risk"
else:
    risk_level = "High risk"

st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Predicted probability", f"{pred_proba:.3f}")

with c2:
    st.metric("Predicted class", pred_class)

with c3:
    st.metric("Risk category", risk_level)

if risk_level == "Low risk":
    st.success(f"Predicted risk category: {risk_level}")
elif risk_level == "Intermediate risk":
    st.warning(f"Predicted risk category: {risk_level}")
else:
    st.error(f"Predicted risk category: {risk_level}")

st.divider()
st.caption("This tool is intended for research use only.")
