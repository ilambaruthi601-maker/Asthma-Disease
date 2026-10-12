# Asthma Detection using Active Learning – Basics of Machine Learning

A beginner-friendly Streamlit educational prototype.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start the app:
   `streamlit run app.py`

## Deploy online
1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Open https://share.streamlit.io/
3. Connect GitHub, choose this repository and `app.py`, then deploy.
4. Streamlit will provide a public app URL.

## CSV format for model training
Required columns:
`age,gender,cough,wheezing,shortness_of_breath,chest_tightness,family_history,smoking,asthma`

- `gender`: Male/Female
- symptom and exposure columns: 0 = No, 1 = Yes
- `asthma`: 0 = No, 1 = Yes

## Important medical disclaimer
The built-in demo model uses synthetic data solely to demonstrate the user interface. It is not medically meaningful, not clinically validated, and must not be used for diagnosis or treatment. Use a suitable, responsibly sourced dataset for academic experiments.
