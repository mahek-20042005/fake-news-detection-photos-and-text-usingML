
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from tensorflow.keras.layers import Dropout
# --- 1. Configuration and Constants ---

# Image dimensions for MobileNetV2 (smaller and faster)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
BATCH_SIZE = 32 # Number of images to process at a time. Reduce if you run out of memory.
EPOCHS = 15     # Number of times to train on the entire dataset.

# Construct absolute paths from the script's location to make it robust
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, '..', 'dataset', 'final_dataset')
SAVED_MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'saved_models')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True) # Ensure the saved_models directory exists

TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VALIDATION_DIR = os.path.join(DATASET_DIR, 'validation')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

# Path where the final model will be saved (using a general name)
MODEL_FILE_PATH = os.path.join(SAVED_MODELS_DIR, 'deepfake_detector_v1two.h5')


# --- 2. Data Preparation and Augmentation ---

# Create a data generator for the training set with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values from [0, 255] to [0, 1]
    rotation_range=20,       # Randomly rotate images
    width_shift_range=0.1,   # Randomly shift images horizontally
    height_shift_range=0.1,  # Randomly shift images vertically
    shear_range=0.1,         # Apply shear transformations
    zoom_range=0.1,          # Randomly zoom in on images
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode='nearest'
)

# Create a data generator for the validation set (NO augmentation, just rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create the data loaders that will feed images from directories to the model
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'  # For real vs. fake classification
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)


# --- 3. Model Building (Transfer Learning with MobileNetV2) ---

# Load the pre-trained MobileNetV2 model, without its final classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))

# Freeze the layers of the base model.
for layer in base_model.layers:
    layer.trainable = False

# Add our custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Flattens the feature maps
x = Dense(512, activation='relu')(x) # A fully-connected layer
x = Dropout(0.5)(x) # ADD THIS LINE - Dropout layer for regularization
predictions = Dense(1, activation='sigmoid')(x) # The final output layer
# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model, defining the optimizer, loss function, and metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Model Summary (MobileNetV2):")
model.summary()


# --- 4. Initial Training (with frozen base) ---

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
model_checkpoint = ModelCheckpoint(MODEL_FILE_PATH, monitor='val_loss', save_best_only=True)

print("\n--- Starting Initial Training (Phase 1) ---")
# Train the model with the base layers frozen
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)
print("--- Initial Training Finished ---")


# --- 5. Fine-Tuning (Unfreeze and train again) ---

print("\n--- Starting Fine-Tuning (Phase 2) ---")

# First, load the best model we saved from the initial training phase
model = tf.keras.models.load_model(MODEL_FILE_PATH)

# Unfreeze the top layers of the model
# Let's unfreeze the last 40 layers. You can experiment with this number.
for layer in base_model.layers[-20:]:
    layer.trainable = True

# We MUST re-compile the model for the changes to take effect.
# It's crucial to use a very low learning rate during fine-tuning.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Very low learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Model Summary after Unfreezing for Fine-Tuning:")
model.summary()

# Continue training for a few more epochs with the unfrozen layers
# We use the same callbacks to save the new best version
fine_tune_epochs = 10
total_epochs = EPOCHS + fine_tune_epochs

history_fine_tune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], # Start from where the last training finished
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)
print("--- Fine-Tuning Finished ---")


# --- 6. Final Evaluation on the Test Set ---

print("\n--- Evaluating Final Model on Test Set ---")
# Create a generator for the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Load the best model that was saved during the entire process
best_model = tf.keras.models.load_model(MODEL_FILE_PATH)

# Evaluate the model on the unseen test data
loss, accuracy = best_model.evaluate(test_generator)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")
print(f"Final Test Loss: {loss:.4f}")