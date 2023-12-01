import tkinter as tk
from keras.src.utils import pad_sequences
from model import tokenizer, load_model

with open('sequence_length.txt', 'r') as file:
    SEQUENCE_LENGTH = int(file.read().strip())

# Load the saved model
model_path = 'movie_name_decider_model'
model = load_model(model_path)

def predict():
    title = entry.get()
    sequence = tokenizer.texts_to_sequences([title])
    data = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    prediction = model.predict(data)
    if prediction > 0.5:
        result = "Hit"
    else:
        result = "Flop"
    result_label.config(text=f"The movie is likely to be a {result}")


root = tk.Tk()
root.title("Movie Success Predictor")
root.geometry("400x250")  # Adjust the size of the window as per your requirement

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

label = tk.Label(frame, text="Enter Movie Title:")
label.pack(pady=10)

entry = tk.Entry(frame, width=50)
entry.pack(pady=10)

button = tk.Button(frame, text="Predict", command=predict)
button.pack(pady=10)

# Label to display the prediction result
result_label = tk.Label(frame, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
