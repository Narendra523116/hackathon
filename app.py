from flask import Flask, render_template, request, jsonify
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
        print(user_input)
        
        # Transform the input using the loaded vectorizer
        user_input_transformed = vectorizer.transform([user_input])
        
        
        prediction = model.predict(user_input_transformed)[0]
        print(prediction)
        encoded = {0: 'Business', 1: 'Crime', 2: 'Education', 3: 'Entertainment', 4: 'Environment', 5: 'Health', 6: 'Nation',7: 'Politics', 8: 'Religion', 9: 'Science', 10: 'Sports', 11: 'Technology', 12: 'Travel', 13: 'World'}
        
        return jsonify(prediction=encoded[prediction])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
