import pandas as pd
import os

from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
import model  # This imports model.py

# Directory containing all CSV files
data_dir = './dataset'

# Load all CSV files into a list of DataFrames
dfs = [pd.read_csv(os.path.join(data_dir, csv_file)) for csv_file in os.listdir(data_dir) if csv_file.endswith('.csv')]

# Concatenate all DataFrames into one
df = pd.concat(dfs, ignore_index=True)

# Create binary labels
# Remove the '%'
df['Budget recovered'] = df['Budget recovered'].str.replace('%', '')

# Convert to float, setting errors='coerce' will turn invalid parsing into NaN
df['Budget recovered'] = pd.to_numeric(df['Budget recovered'], errors='coerce')

# Handle NaN values by replacing them with 0 or another default value
df['Budget recovered'].fillna(0, inplace=True)

# Apply the lambda function
df['label'] = df['Budget recovered'].apply(lambda x: 1 if x > 120 else 0)

# Tokenize titles and split data
model.tokenizer.fit_on_texts(df['Film'])
sequences = model.tokenizer.texts_to_sequences(df['Film'])
data = pad_sequences(sequences)
X_train, X_test, y_train, y_test = train_test_split(data, df['label'], test_size=0.2)

# Assuming all sequences are of the same length after padding
sequence_length = data.shape[1]

with open('sequence_length.txt', 'w') as file:
    file.write(str(sequence_length))

# Load or create model
model_path = 'movie_name_decider_model'
trained_model = model.load_model(model_path)
if trained_model is None:
    print("No pre-trained model found. Starting training process.")
    trained_model = model.create_model(input_length=sequence_length)

    # Compile and train the model
    trained_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    trained_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the trained model
    trained_model.save(model_path)
else:
    print("Pre-trained model loaded successfully.")

