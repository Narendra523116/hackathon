import pickle

# Load the fitted vectorizer
with open('path_to_your_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load your trained MultinomialNB model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Test the vectorizer to ensure it's fitted
sample_text = ["This is a test sentence."]
try:
    # Transforming the text should work without errors if the vectorizer is fitted
    transformed_text = vectorizer.transform(sample_text)
    print("Transformation successful:", transformed_text.shape)
except Exception as e:
    print("Error during transformation:", str(e))

# Ensure that the transformed text can be passed to the model
try:
    prediction = model.predict(transformed_text)
    print("Prediction successful:", prediction)
except Exception as e:
    print("Error during prediction:", str(e))
