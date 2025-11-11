# predict.py

import joblib
import pandas as pd

# Load the saved pipeline object from the file
PIPELINE_PATH = 'text_only_model.joblib'
try:
    model_pipeline = joblib.load(PIPELINE_PATH)
    print(f"✅ Model pipeline loaded successfully from '{PIPELINE_PATH}'")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at '{PIPELINE_PATH}'. Please run modeltrain.py first.")
    exit()

def predict_news(statement: str, subject: str, speaker: str, party: str, history: list):
    """
    Makes a prediction on a single news statement.
    """
    data = {
        'statement': [statement],
        'subject': [subject],
        'speaker': [speaker],
        'party_affiliation': [party],
        'barely_true_counts': [history[0]],
        'false_counts': [history[1]],
        'half_true_counts': [history[2]],
        'mostly_true_counts': [history[3]],
        'pants_on_fire_counts': [history[4]]
    }

    input_df = pd.DataFrame(data)

    prediction = model_pipeline.predict(input_df)[0]

    probabilities = model_pipeline.predict_proba(input_df)[0]
    confidence = dict(zip(model_pipeline.classes_, probabilities))

    return {
        'prediction': prediction,
        'confidence_scores': {
            'fake': round(confidence.get('fake', 0.0), 4),
            'real': round(confidence.get('real', 0.0), 4)
        }
    }

if __name__ == "__main__":
    # --- EXAMPLE USAGE ---

    statement_1 = "Social Security is a Ponzi scheme and is going bankrupt."
    history_1 = [20, 30, 15, 10, 5]
    result_1 = predict_news(
        statement=statement_1,
        subject='social-security',
        speaker='rick-perry',
        party='republican',
        history=history_1
    )
    print("\n--- Prediction 1 ---")
    print(f"Statement: '{statement_1}'")
    print(f"Predicted Label: {result_1['prediction'].upper()}")
    print(f"Confidence: {result_1['confidence_scores']}")

    statement_2 = "The unemployment rate has decreased over the last quarter."
    history_2 = [2, 1, 5, 25, 0]
    result_2 = predict_news(
        statement=statement_2,
        subject='economy,jobs',
        speaker='barack-obama',
        party='democrat',
        history=history_2
    )
    print("\n--- Prediction 2 ---")
    print(f"Statement: '{statement_2}'")
    print(f"Predicted Label: {result_2['prediction'].upper()}")
    print(f"Confidence: {result_2['confidence_scores']}")
