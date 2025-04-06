import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string

# Load the vectorizer and model
vector = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Preprocessing function
def text_transform(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text into words
    text = nltk.word_tokenize(text)
    
    # Extract only alphanumeric values
    y = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    # Return the preprocessed text as a single string
    return ' '.join(y)  # Join the list of words back into a single string

# Streamlit UI
st.title('Sentiment Analysis App')

# Input from the user
user_input = st.text_area("Enter a longer message:")

# Process the input when the button is clicked
if st.button('Analyze'):
    # Preprocess the input text
    transformed_text = text_transform(user_input)
    
    # Vectorize the input text
    vectore = vector.transform([transformed_text])
    
    # Predict sentiment using the model
    prediction = model.predict(vectore)[0]
    
    # Display the result
    #st.write(f"Predicted Sentiment: {prediction}")
    if prediction == 0:
        st.write('Prediction sentiment is Negitive')
    elif prediction == 1:
        st.write('Prediction is Positive')
    elif prediction == 2:
        st.write('Prediction is Neutral')
    elif prediction == 3:
        st.write("Prediction is Irrelvant")
