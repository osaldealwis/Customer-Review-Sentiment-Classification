
#  Customer Review Sentiment Classification

##  About the Project
This project is focused on **Sentiment Analysis** of textual data, specifically designed to classify **customer reviews** as either **Positive** or **Negative**. 

Sentiment Analysis is widely used in businesses to automatically analyze customer feedback, product reviews, and social media mentions. This helps companies understand user sentiments, improve their products, and make data-driven decisions.

For this project, a **Logistic Regression model** was trained on a large dataset of tweets (which are short texts like customer reviews) to predict sentiments accurately.


##  What Has Been Done
- Data Cleaning and Preprocessing using **NLTK**  
- Feature Extraction using **TF-IDF Vectorizer**  
- Model Training using **Logistic Regression** from **Scikit-learn**  
- Model Evaluation - Achieved **~79% accuracy**  
- Model and Vectorizer saved as `.sav` files using **Pickle**  
- A Python function is created to load the model, preprocess new reviews, and predict sentiment


##  Dataset Used
- **Name:** [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size:** 1.6 Million tweets
- **Description:** Contains tweets labeled as positive or negative 
- **Reason for Use:** Tweets are short text similar to customer reviews, making it ideal for training a model to classify small review texts.


##  How to Use the Project
You can use the trained model to **predict the sentiment of any customer review** by following these steps:

### 1. Clone the Repository and Install Requirements
```bash
git clone <[your-repository-link](https://github.com/osaldealwis/Customer-Review-Sentiment-Classification.git)>
cd sentiment-analysis-project
pip install -r requirements.txt
```

### 2. Load the Model and Vectorizer
```python
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

# Load the saved model and vectorizer
model = pickle.load(open('twitter_model.sav', 'rb'))
vectorizer = pickle.load(open('twitter_vectorizer.sav', 'rb'))
```

### 3. Define the Preprocessing Function
```python
def preprocess_text(text):
    port_stem = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters and numbers
    text = text.lower()
    text = text.split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)
```

### 4. Predict Sentiment of a Sample Review
```python
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([processed_review])
    prediction = model.predict(vectorized_review)
    return "Positive" if prediction[0] == 1 else "Negative"

# Example Usage
sample_review = "I love this product! It's amazing and works perfectly."
print("Predicted Sentiment:", predict_sentiment(sample_review))
```


##  Project Outcome
- The model can now predict the sentiment of any new customer review text


##  Future Improvements
- Add **Neutral** sentiment classification
- Deploy as an API using **Flask/Django**
- Visualize results with sentiment percentages
- Use deep learning models like **LSTM** or **BERT** for higher accuracy

---


