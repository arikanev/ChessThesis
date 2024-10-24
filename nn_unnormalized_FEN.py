import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import keras_tuner as kt  # Import Keras Tuner
from tensorflow.keras.layers import TextVectorization, Embedding, MultiHeadAttention, LayerNormalization, Dense, Dropout, Input, Concatenate

# Load the dataset
df = pd.read_csv('chess_dataset_big_normalized.csv')

# Filter out rows with NaN values in 'CPL_normalized' and 'FEN'
df = df.dropna(subset=['CPL', 'FEN'])

# Prepare the features (X) and labels (y)
X_num = df[['CPL', 'ELO']].values  # Numerical features
X_fen = df['FEN'].values  # FEN strings
y = df['Cheat'].values  # Labels

# Tokenizer for FEN
vectorizer = TextVectorization(output_mode='int', max_tokens=100, output_sequence_length=64)  # Adjust max length if needed
vectorizer.adapt(X_fen)

# Tokenize the FEN feature
X_fen_tokenized = vectorizer(X_fen)

# Convert the tokenized FEN tensor to a NumPy array
X_fen_tokenized = X_fen_tokenized.numpy()

# Split into training and test sets
X_num_train, X_num_test, X_fen_train, X_fen_test, y_train, y_test = train_test_split(
    X_num, X_fen_tokenized, y, test_size=0.2, random_state=42)

# Now continue with the model-building and training process
# Standardize the numerical features
scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num_train)
X_num_test = scaler.transform(X_num_test)

# Define Transformer-based block for FEN processing
def transformer_block(inputs, embed_dim, num_heads, ff_dim, rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ff_output = Dense(ff_dim, activation="relu")(out1)
    ff_output = Dropout(rate)(ff_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ff_output)

# Define the combined model
def build_combined_model(hp):
    # FEN input and embedding
    fen_input = Input(shape=(64,), dtype=tf.int32, name='fen_input')  # Adjust max length as needed
    fen_embedding = Embedding(input_dim=100, output_dim=64)(fen_input)  # 100 tokens, 64-dim embeddings
    transformer_output = transformer_block(fen_embedding, embed_dim=64, num_heads=4, ff_dim=64)
    transformer_output = keras.layers.GlobalAveragePooling1D()(transformer_output)
    
    # Numerical input (CPL_normalized, ELO)
    num_input = Input(shape=(2,), name='num_input')
    num_dense = Dense(units=hp.Int('units_dense_1', min_value=8, max_value=64, step=8),
                      activation='relu')(num_input)
    
    # Combine the transformer and dense output
    combined = Concatenate()([transformer_output, num_dense])
    combined = Dense(units=hp.Int('units_dense_2', min_value=8, max_value=64, step=8), activation='relu')(combined)
    
    # Output layer (binary classification)
    output = Dense(1, activation='sigmoid')(combined)

    # Compile the model
    model = keras.Model(inputs=[fen_input, num_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the Keras Tuner
tuner = kt.RandomSearch(
    build_combined_model,  # Pass in the model-building function
    objective='val_accuracy',  # Optimize for validation accuracy
    max_trials=10,  # Perform 10 different trials
    executions_per_trial=1,  # Number of times to repeat each trial for robustness
    directory='tuning_unnormalized_with_FEN',  # Directory to store tuning results
    project_name='chess_hyperparam_tuning_with_fen'
)

# Search for the best hyperparameters
tuner.search([X_fen_train, X_num_train], y_train, epochs=100, validation_split=0.2, batch_size=32)

# Retrieve the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameters
print("Best Hyperparameters:")
print(best_hyperparameters.values)

# Train the best model for more epochs
history = best_model.fit([X_fen_train, X_num_train], y_train, epochs=130, batch_size=32, validation_split=0.2)

# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate([X_fen_test, X_num_test], y_test)
print(f"Best Model Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate the classification report after the model is trained
y_pred_probs = best_model.predict([X_fen_test, X_num_test])
y_pred = (y_pred_probs > 0.5).astype(int)

classification_report_nn = classification_report(y_test, y_pred)
print("\nClassification Report:\n", classification_report_nn)

# Plot the training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
