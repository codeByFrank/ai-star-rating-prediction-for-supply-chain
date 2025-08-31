# AI Star Rating Prediction for Supply Chain

Predict 1–5★ ratings from Trustpilot review text (Temu focus) using classic ML and Deep Learning.
Includes a Streamlit app for live predictions, model comparison, and a 100-sample quick evaluation.

Live demo: https://supply-chain-ai-star-prediction.streamlit.app

# What this project does

* Uses Trustpilot reviews (English) and cleans/normalizes the text
* Engineers features (TF-IDF + light numeric text signals + optional VADER)
* Trains & compares multiple models (LogReg, LinearSVC, XGBoost, Voting/Stacking + DL variants)
* Serves a Streamlit app for instant star-rating prediction and model inspection
* More visuals & PDFs live in /documentation/.
  
## Quick preview

# Mashine learning
<p align="center">
  <img src="documentation/100_sample%20evaluation_%20color_coded_results.png" width="100%" />
</p>
<p align="center">
  <img src="documentation/ml_live_prediction_sentiment_3_%20penalty.png" width="100%" />
  <img src="documentation/ml_live_prediction_chart.png" width="100%" />
</p>
<p align="center">
    <img src="documentation/ml_history.png" width="100%" />
</p>

# Deep learning
<p align="center">
  <img src="documentation/dl_live_prediction.png" width="100%" />
  <img src="documentation/dl_live_prediction_chart.png" width="100%" />
</p>
