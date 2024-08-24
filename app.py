from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained MultinomialNB model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the correctly fitted vectorizer
with open('TfidfVectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['user_input']
        
        # Transform the input using the loaded vectorizer
        user_input_transformed = vectorizer.transform([user_input])
        
        # Reshape if necessary
        # user_input_transformed = user_input_transformed.toarray().reshape(1, -1)
        
        # Make a prediction using the transformed input
        prediction = model.predict(user_input_transformed)[0]
        encoded = {0: 'Business', 1: 'Crime', 2: 'Education', 3: 'Entertainment', 4: 'Environment', 5: 'Health', 6: 'Nation',7: 'Politics', 8: 'Religion', 9: 'Science', 10: 'Sports', 11: 'Technology', 12: 'Travel', 13: 'World'}
        
        return render_template('index.html', prediction=encoded[prediction])
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
