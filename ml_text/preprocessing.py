import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class PreprocessorTextOnly:
    """
    Loads and prepares the LIAR dataset, but ONLY uses the 'statement' text
    as the feature for the model.
    """

    def __init__(self, data_path: str = 'data/'):
        self.data_path = data_path
        self.columns = [
            'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
            'state_info', 'party_affiliation', 'barely_true_counts',
            'false_counts', 'half_true_counts', 'mostly_true_counts',
            'pants_on_fire_counts', 'context'
        ]

    def _load_single_file(self, filename: str) -> pd.DataFrame:
        filepath = os.path.join(self.data_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error: File not found at '{filepath}'")
        return pd.read_csv(filepath, sep='\t', header=None, names=self.columns)

    def get_data(self):
        """
        Loads the data and returns the text statements (X) and labels (y).
        """
        try:
            train_df = self._load_single_file('train.tsv')
            valid_df = self._load_single_file('valid.tsv')
            test_df = self._load_single_file('test.tsv')
        except FileNotFoundError as e:
            print(e)
            return None, None, None, None

        # Combine training + validation for more samples
        df_train_full = pd.concat([train_df, valid_df], ignore_index=True)

        # Prepare data
        X_train = df_train_full['statement'].fillna('')
        y_train = df_train_full['label']
        X_test = test_df['statement'].fillna('')
        y_test = test_df['label']

        print("Text-only data loaded successfully.")
        return X_train, y_train, X_test, y_test

    def get_vectorizer(self) -> TfidfVectorizer:
        """Returns a configured TF-IDF vectorizer."""
        return TfidfVectorizer(
            stop_words='english',
            max_features=7000,
            ngram_range=(1, 2)
        )
