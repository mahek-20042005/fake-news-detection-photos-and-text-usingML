
# ==============================================================================
# STEP 3: BUILD THE HYBRID MODEL
# This code should be added to the end of the script from Step 2.
# ==============================================================================

from transformers import TFBertModel
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.metrics import classification_report

print("\n--- Starting Hybrid Model Construction ---")

# --- 1. Define the Model's Inputs ---
# These input layers must match the keys in our dataset dictionary.

# BERT inputs
input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
token_type_ids = Input(shape=(128,), dtype=tf.int32, name='token_type_ids')
attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')

# Metadata input
# The shape must match the number of columns in our processed metadata.
# X_train_meta.shape[1] will be 6768
metadata_input = Input(shape=(X_train_meta.shape[1],), dtype=tf.float32, name='metadata')


# --- 2. Define the BERT Branch (Text Processing) ---
# We load the base BERT model without its classification head.
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_model.trainable = True # We want to fine-tune the BERT layers

# Pass the inputs to the BERT model
bert_output = bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

# The most meaningful output from BERT for classification is the '[CLS]' token's embedding.
# It's the first token's output, so we take the slice [:, 0, :].
text_features = bert_output.last_hidden_state[:, 0, :]


# --- 3. Define the Metadata Branch (MLP) ---
# A simple Multi-Layer Perceptron (MLP) to process the structured data.
# We add Dropout for regularization to prevent overfitting.
x = Dense(64, activation='relu')(metadata_input)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
metadata_features = Dropout(0.3)(x)


# --- 4. Concatenate (Combine) the Branches ---
# We combine the features extracted from the text and the metadata.
combined_features = Concatenate()([text_features, metadata_features])


# --- 5. Add the Final Classifier Head ---
# We add a few more layers to make a final prediction based on the combined features.
final_dense = Dense(64, activation='relu')(combined_features)
final_dropout = Dropout(0.5)(final_dense)
output_layer = Dense(6, activation='softmax', name='output')(final_dropout) # 6 labels, so 6 neurons


# --- 6. Create the Final Model ---
# The model is defined by its inputs and its final output layer.
model = Model(
    inputs={
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'metadata': metadata_input
    },
    outputs=output_layer
)

# --- 7. Compile the Model ---
optimizer = Adam(learning_rate=3e-5) # A smaller learning rate is good for fine-tuning BERT
loss = SparseCategoricalCrossentropy()
metric = SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Print a summary of the model architecture
model.summary()
print("--- Model Construction Complete ---")


# --- 8. Train the Model ---
print("\n--- Starting Model Training ---")
BATCH_SIZE = 16
EPOCHS = 3 # Start with 2-3 epochs for fine-tuning

# Prepare the datasets for training by batching and prefetching
train_dataset_batched = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
valid_dataset_batched = valid_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset_batched = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

history = model.fit(
    train_dataset_batched,
    epochs=EPOCHS,
    validation_data=valid_dataset_batched
)
print("--- Model Training Complete ---")


# --- 9. Evaluate the Model ---
print("\n--- Starting Model Evaluation on Test Set ---")
results = model.evaluate(test_dataset_batched)
print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")

# Get predictions to generate a detailed classification report
predictions = model.predict(test_dataset_batched)
predicted_labels = np.argmax(predictions, axis=1)

# Get the label names from our label_map dictionary
target_names = list(label_map.keys())

print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels, target_names=target_names, digits=4))