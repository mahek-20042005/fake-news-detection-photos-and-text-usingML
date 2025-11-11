from preprocessing import PreprocessorTextOnly
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# for saving the image without opening a window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train_text_only_model():
    # 1. --- Load and Preprocess Data ---
    pre = PreprocessorTextOnly(
        data_path=r"C:/Users/singh/OneDrive/Desktop/Fake_news/ml_text/data/"
    )
    X_train, y_train, X_test, y_test = pre.get_data()

    if X_train is None:
        return

    print("\n" + "="*50)
    print("      Training a TEXT-ONLY Binary Classification Model")
    print("="*50)

    # Define label mappings
    real_labels = ['mostly-true', 'true']
    fake_labels = ['false', 'pants-fire', 'barely-true']

    # Convert labels to binary and filter out 'half-true'
    y_train_binary = y_train.apply(lambda x: 'real' if x in real_labels else ('fake' if x in fake_labels else None))
    X_train_binary = X_train[y_train_binary.notna()]
    y_train_binary = y_train_binary.dropna()

    y_test_binary = y_test.apply(lambda x: 'real' if x in real_labels else ('fake' if x in fake_labels else None))
    X_test_binary = X_test[y_test_binary.notna()]
    y_test_binary = y_test_binary.dropna()

    # 3. --- Define and Train the Model Pipeline ---
    text_only_pipeline = Pipeline(steps=[
        ('vectorizer', pre.get_vectorizer()),
        ('classifier', LogisticRegression(
            C=0.5,
            solver='liblinear',
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        ))
    ])

    print("\nTraining the text-only binary model...")
    text_only_pipeline.fit(X_train_binary, y_train_binary)
    print("Training complete.")

    # 4. --- Evaluate the Model ---
    print("\n--- Evaluating on the Test Set ---")
    y_pred = text_only_pipeline.predict(X_test_binary)
    accuracy = accuracy_score(y_test_binary, y_pred)

    print(f"\nModel Accuracy (Binary, Text-Only): {accuracy:.4f}")
    print("\nBinary Classification Report (Text-Only):")
    print(classification_report(y_test_binary, y_pred, digits=3))

    # 4a. --- Confusion Matrix (saved as image) ---
    labels = ['fake', 'real']  # ensure consistent order
    cm = confusion_matrix(y_test_binary, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix — Text-Only (Binary)')
    plt.tight_layout()
    out_img = 'confusion_matrix.png'   # saved in current working directory
    plt.savefig(out_img, dpi=160)
    plt.close(fig)
    print(f"\n📷 Confusion matrix image saved to '{out_img}'")

    # 5. --- Save the Model ---
    model_filename = 'text_only_model.joblib'
    joblib.dump(text_only_pipeline, model_filename)
    print("\n" + "="*50)
    print(f"✅ Text-only model pipeline saved successfully to '{model_filename}'")
    print("="*50)


if __name__ == '__main__':
    train_text_only_model()
