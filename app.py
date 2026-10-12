import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Asthma Detection | Active Learning", page_icon="🫁", layout="wide")

FEATURES = [
    "age", "gender", "cough", "wheezing", "shortness_of_breath",
    "chest_tightness", "family_history", "smoking"
]
TARGET = "asthma"
BINARY_COLS = ["cough", "wheezing", "shortness_of_breath", "chest_tightness", "family_history", "smoking"]

st.markdown("""
<style>
.main {background-color:#f7fbff;}
.hero {padding: 1.3rem 1.5rem; border-radius: 16px; background: linear-gradient(110deg,#dff4ff,#eef7ff); border:1px solid #c9e7ff;}
.hero h1 {color:#123c70; margin-bottom:0.25rem;}
.small-note {font-size:0.9rem;color:#52657d;}
div[data-testid="stMetric"] {background:white; border:1px solid #dcecff; padding:12px; border-radius:12px;}
</style>
<div class="hero">
<h1>🫁 Asthma Detection using Active Learning</h1>
<p><b>Basics of Machine Learning</b> · Educational prototype for learning classification and uncertainty sampling</p>
<p class="small-note">This app is for academic demonstration only. It is not a medical device and cannot diagnose asthma.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🫁 Asthma ML")
    page = st.radio("Navigation", ["Home & Predict", "Train Model", "Active Learning", "About"])
    st.caption("Basics of Machine Learning")

def normalize_gender(series):
    return series.astype(str).str.strip().str.lower().map({
        "male": 1, "m": 1, "man": 1,
        "female": 0, "f": 0, "woman": 0
    }).fillna(0).astype(int)

def prepare_data(df, require_target=True):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {
        "shortness_of_breath_(0/1)": "shortness_of_breath",
        "chest_tightness_(0/1)": "chest_tightness",
        "family_history_(0/1)": "family_history",
        "cough_(0/1)": "cough",
        "wheezing_(0/1)": "wheezing",
        "smoking_(0/1)": "smoking",
        "gender_(male=1)": "gender"
    }
    df = df.rename(columns=aliases)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df["gender"] = normalize_gender(df["gender"])
    for c in ["age"] + BINARY_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if require_target and TARGET not in df.columns:
        raise ValueError("Your CSV must include a target column named 'asthma' (0 = No, 1 = Yes).")
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=FEATURES + ([TARGET] if require_target else []))
    return df

def build_demo_model():
    # Synthetic data is deliberately only for demonstrating app flow, not medical use.
    rng = np.random.default_rng(21)
    n = 700
    age = rng.integers(8, 80, n)
    gender = rng.integers(0, 2, n)
    cough = rng.binomial(1, .35, n)
    wheeze = rng.binomial(1, .22, n)
    breath = rng.binomial(1, .20, n)
    tight = rng.binomial(1, .18, n)
    family = rng.binomial(1, .24, n)
    smoking = rng.binomial(1, .20, n)
    score = -2.1 + .9*cough + 1.35*wheeze + 1.2*breath + .75*tight + .8*family + .35*smoking + .003*age
    prob = 1/(1+np.exp(-score))
    y = rng.binomial(1, prob)
    X = pd.DataFrame({
        "age":age, "gender":gender, "cough":cough, "wheezing":wheeze,
        "shortness_of_breath":breath, "chest_tightness":tight,
        "family_history":family, "smoking":smoking
    })
    model = RandomForestClassifier(n_estimators=120, random_state=21, class_weight="balanced")
    model.fit(X, y)
    return model

if "model" not in st.session_state:
    st.session_state.model = build_demo_model()
    st.session_state.model_source = "Illustrative synthetic demo model"
    st.session_state.trained = False

if page == "Home & Predict":
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.subheader("Enter sample details")
        st.info("Enter 0 for No and 1 for Yes in symptom fields. This is an educational example, not a clinical assessment.")
        with st.form("predict_form"):
            age = st.number_input("Age", min_value=1, max_value=110, value=25)
            gender = st.selectbox("Gender (encoded for model)", ["Male", "Female"])
            c1, c2 = st.columns(2)
            with c1:
                cough = st.selectbox("Cough", [0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
                wheezing = st.selectbox("Wheezing", [0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
                breath = st.selectbox("Shortness of breath", [0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
            with c2:
                tight = st.selectbox("Chest tightness", [0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
                family = st.selectbox("Family history of asthma", [0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
                smoking = st.selectbox("Smoking exposure", [0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
            submitted = st.form_submit_button("🔎 Run demo prediction", use_container_width=True)
        if submitted:
            row = pd.DataFrame([{
                "age":age, "gender":1 if gender=="Male" else 0, "cough":cough,
                "wheezing":wheezing, "shortness_of_breath":breath,
                "chest_tightness":tight, "family_history":family, "smoking":smoking
            }])
            proba = st.session_state.model.predict_proba(row)[0]
            classes = list(st.session_state.model.classes_)
            asthma_prob = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
            st.session_state.last_input = row
            st.session_state.last_prob = asthma_prob
    with right:
        st.subheader("Prediction result")
        if "last_prob" in st.session_state:
            p = st.session_state.last_prob
            if p >= .5:
                st.warning(f"Model output: higher predicted class-1 score ({p:.1%})")
            else:
                st.success(f"Model output: lower predicted class-1 score ({p:.1%})")
            st.caption("This percentage is a model score from a demo model, not a real-world medical risk.")
        else:
            st.markdown("### 🩺 Ready to explore")
            st.write("Fill in the sample details and select **Run demo prediction**.")
        st.markdown("---")
        st.markdown("#### Key features")
        st.write("✅ Classification with Random Forest")
        st.write("✅ CSV-based model training")
        st.write("✅ Uncertainty sampling demonstration")
        st.write("✅ Beginner-friendly interface")
        st.caption("Current model: " + st.session_state.model_source)

elif page == "Train Model":
    st.subheader("Train a model with your own labelled dataset")
    st.warning("Only use a dataset you are permitted to use. The model is for coursework/demo purposes and is not validated for clinical use.")
    st.markdown("CSV columns required: `age, gender, cough, wheezing, shortness_of_breath, chest_tightness, family_history, smoking, asthma`")
    st.markdown("Use 0/1 for yes/no fields and `Male`/`Female` for gender. Target `asthma`: 0 = No, 1 = Yes.")
    file = st.file_uploader("Upload labelled CSV", type=["csv"])
    if file is not None:
        try:
            data = prepare_data(pd.read_csv(file), require_target=True)
            if data[TARGET].nunique() < 2:
                st.error("The asthma target column must contain both 0 and 1 classes.")
            else:
                st.write("Preview", data.head())
                st.metric("Usable rows", len(data))
                if st.button("Train Random Forest model", type="primary"):
                    X = data[FEATURES]
                    y = data[TARGET].astype(int)
                    if len(data) >= 10 and y.value_counts().min() >= 2:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=.2, random_state=42, stratify=y
                        )
                    else:
                        X_train, X_test, y_train, y_test = X, X, y, y
                    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
                    model.fit(X_train, y_train)
                    score = accuracy_score(y_test, model.predict(X_test))
                    st.session_state.model = model
                    st.session_state.model_source = f"User-trained model ({len(data)} rows)"
                    st.session_state.trained = True
                    st.success("Model trained and now used by the prediction page.")
                    st.metric("Holdout accuracy" if len(data) >= 10 and y.value_counts().min() >= 2 else "Training-set accuracy (small dataset)", f"{score:.1%}")
                    st.caption("Accuracy alone does not establish clinical safety or validity.")
        except Exception as e:
            st.error(str(e))

elif page == "Active Learning":
    st.subheader("Active Learning: uncertainty sampling")
    st.write("Active learning asks which unlabelled examples would be most useful for a human to label next. This page ranks samples where the model is least certain.")
    st.markdown("Upload an **unlabelled CSV** with the eight feature columns. The app will return rows with predictions and uncertainty scores.")
    st.caption("Gender can be Male/Female; binary symptom columns should be 0 or 1.")
    file = st.file_uploader("Upload unlabelled CSV", type=["csv"], key="active_csv")
    if file is not None:
        try:
            raw = pd.read_csv(file)
            raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
            prepared = prepare_data(raw, require_target=False)
            X = prepared[FEATURES]
            probs = st.session_state.model.predict_proba(X)
            classes = list(st.session_state.model.classes_)
            if 1 in classes:
                p1 = probs[:, classes.index(1)]
            else:
                p1 = probs[:, -1]
            out = raw.loc[prepared.index].copy()
            out["predicted_asthma_class"] = (p1 >= .5).astype(int)
            out["model_score_class_1"] = p1.round(4)
            out["uncertainty"] = (1 - np.abs(2*p1 - 1)).round(4)
            out = out.sort_values("uncertainty", ascending=False)
            st.success("Most uncertain samples are shown first.")
            st.dataframe(out.head(20), use_container_width=True)
            st.download_button("Download ranked samples CSV", out.to_csv(index=False), "active_learning_ranked_samples.csv", "text/csv")
        except Exception as e:
            st.error(str(e))

else:
    st.subheader("About this project")
    st.write("""
    **Project title:** Asthma Detection using Active Learning – Basics of Machine Learning

    **Learning goals**
    - Understand a supervised classification workflow.
    - Explore a Random Forest classifier.
    - See how uncertainty sampling can prioritize examples for labelling.
    - Learn why dataset quality, validation, bias, and privacy matter.

    **Important limitations**
    - The built-in model uses synthetic, randomly generated data only to demonstrate the interface.
    - Its outputs are not medically meaningful and must not be used for diagnosis, treatment, or emergency decisions.
    - For academic experiments, use a reputable, appropriately consented dataset and clearly report its source and limitations.
    - If someone has breathing difficulty or severe asthma symptoms, seek urgent medical care.
    """)
    st.markdown("**Required packages:** Streamlit, pandas, numpy, scikit-learn.")
