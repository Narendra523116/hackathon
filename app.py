from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load your trained model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get the user's input from the form
        user_input = request.form['user_input']
        
        # Preprocess the input if needed
        # For example, if your model requires vectorization or any other preprocessing
        # Example: user_input_transformed = vectorizer.transform([user_input])

        # Make a prediction using the model
        prediction = model.predict([user_input])[0]  # Assuming the model expects a list of inputs
        
        # Render the template with the prediction
        return render_template('index.html', prediction=prediction)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
