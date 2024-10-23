import ssl
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, \
    precision_recall_fscore_support, precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

# Secure a proper certificate
ssl._create_default_https_context = ssl._create_unverified_context

# Global level, data importation

labelled_path = "C:/Users/dell XPS 15/PycharmProjects/NLPTextAnalytics/talkLabelled.txt"
sticker_path = 'C:/Users/dell XPS 15/PycharmProjects/NLPTextAnalytics/hpy.jpeg'
sticker_path1 = 'C:/Users/dell XPS 15/PycharmProjects/NLPTextAnalytics/net.jpeg'
sticker_path2 = 'C:/Users/dell XPS 15/PycharmProjects/NLPTextAnalytics/neg.jpeg'
cleaning_image_path = 'C:/Users/dell XPS 15/PycharmProjects/NLPTextAnalytics/cleaners.jpeg'
preprocessing_image_path = 'C:/Users/dell XPS 15/PycharmProjects/NLPTextAnalytics/processing.jpeg'
analysis_image_path = 'C:/Users/dell XPS 15/PycharmProjects/NLPTextAnalytics/analysis.jpeg'
evaluation_image_path = 'C:/Users/dell XPS 15/PycharmProjects/NLPTextAnalytics/evaluation.jpg'
background_image_path = 'C:/Users/dell XPS 15/PycharmProjects/NLPTextAnalytics/bakgd.jpg'

# Open the file and read each line
with open(labelled_path, 'r', encoding="utf-8") as file:
    lines = file.readlines()

# Text cleaning
def clean_text(lines):
    cleaned_lines = []
    for line in lines:
        # Convert to lowercase
        text = line.lower()

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        cleaned_lines.append(text)

    return cleaned_lines

lines = clean_text(lines)

# Downloading NLTK resources for text preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')  # this is used for Part of Speech tagging
nltk.download('wordnet')  # for lemmatization

# Set global background image
background_image = f"""
    <style>
    .stApp {{
        background-image: url({background_image_path});
        background-size: cover;
        background-repeat: no-repeat;
    }}
    </style>
"""
st.markdown(background_image, unsafe_allow_html=True)

def text_cleaning():
    st.header("Text Cleaning")

    # Display image on top
    image = Image.open(cleaning_image_path)
    st.image(image, width=100)  # Resize image to width 100px

    st.write("Data after cleaning:")
    st.write(lines)
    dataset = pd.DataFrame({'talk': lines})
    st.write(dataset)

def text_preprocessing():
    st.header("Text Preprocessing")

    # Display image on top
    image = Image.open(preprocessing_image_path)
    st.image(image, width=100)  # Resize image to width 100px

    # Tokenize text
    words = word_tokenize(' '.join(lines))
    tok = pd.DataFrame({'talk': words})
    st.write("Tokenized words:", tok)

    # Displaying stop words
    stop_words = set(stopwords.words('english'))
    st.write("Stopwords:", stop_words)

    # Removal of stop words
    filtered = [word for word in words if word.lower() not in stop_words]
    st.write("Filtered text:", filtered)

    # Stem the text
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in filtered]
    st.write("Stemmed words:", stemmed)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize the stemmed words
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    st.write("Lemmatized words:", lemmatized)

    # POS Tagging
    pos_tags = pos_tag(lemmatized)
    st.write("POS tags:", pos_tags)

    # Train Word2Vec model
    model = Word2Vec([lemmatized], vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = model.wv
    st.write("Word2Vec model has been trained.")

def sentiment_analysis():
    st.header("Sentiment Analysis")

    # Display image on top
    image = Image.open(analysis_image_path)
    st.image(image, width=100)  # Resize image to width 100px

    # Reading the data
    labelled_df = pd.read_csv(labelled_path, delimiter='\t')

    # Separate the predictor variable from the class variable. Assume X for the text and y for the class
    X = labelled_df['messages']
    y = labelled_df['labels']

    # Tokenize text and create Word2Vec model
    X_tokenized = [word_tokenize(text.lower()) for text in X]
    w2v_model = Word2Vec(sentences=X_tokenized, vector_size=50, window=10, min_count=1, workers=20)
    word_vectors = w2v_model.wv

    # Average Word2Vec vectors for each text
    def get_avg_word2vec(tokens_list, vector, k=100):
        if len(tokens_list) < 1:
            return np.zeros(k)
        vec = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
        return np.mean(vec, axis=0)

    X_vectorized = np.array([get_avg_word2vec(tokens, word_vectors) for tokens in X_tokenized])

    # Train test approach with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=40, stratify=y)

    # Handle imbalanced data using SMOTE
    smote = SMOTE(random_state=35)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train the Naive Bayes Classifier
    nb = GaussianNB()
    nb.fit(X_train_res, y_train_res)

    # Predict the test results
    test_predict = nb.predict(X_test)

    # Store models and vectors in session state
    st.session_state['nb'] = nb
    st.session_state['word_vectors'] = word_vectors
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    st.session_state['test_predict'] = test_predict

    # Text input for user prediction
    user_input = st.text_area("Enter text here 💬")
    if st.button("Predict"):
        user_input_tokens = word_tokenize(user_input.lower())
        user_input_vector = get_avg_word2vec(user_input_tokens, word_vectors).reshape(1, -1)
        prediction = nb.predict(user_input_vector)
        st.write(f"Predicted sentiment: {prediction[0]}")

    # File upload for user prediction
    file_upload = st.file_uploader("Import your file here", type=["txt", "csv"])

    if file_upload is not None:
        file_data = file_upload.read()
        user_input = file_data.decode("utf-8")

    if st.button('Predict from File 🔮'):
        user_input_tokens = word_tokenize(user_input.lower())
        user_input_vector = get_avg_word2vec(user_input_tokens, word_vectors).reshape(1, -1)
        prediction = nb.predict(user_input_vector)

        if prediction[0] == 'Positive':
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Sentiment is: 😊 {prediction[0]}")
            with col2:
                image = Image.open(sticker_path)
                st.image(image, width=100)  # Resize image to width 100px
        elif prediction[0] == 'Neutral':
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Sentiment is: 😐 {prediction[0]}")
            with col2:
                image = Image.open(sticker_path1)
                st.image(image, width=100)  # Resize image to width 100px
        else:
            col3, col4 = st.columns(2)
            with col3:
                st.write(f"Sentiment is: ☹️ {prediction[0]}")
            with col4:
                image = Image.open(sticker_path2)
                st.image(image, width=100)  # Resize image to width 100px

def model_evaluation():
    st.header("Model Evaluation")

    # Display image on top
    image = Image.open(evaluation_image_path)
    st.image(image, width=100)  # Resize image to width 100px

    nb = st.session_state['nb']
    word_vectors = st.session_state['word_vectors']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    test_predict = st.session_state['test_predict']

    # Evaluate
    st.write(f"The accuracy of the model is 📊 {accuracy_score(y_test, test_predict) * 100:.2f}%")
    report = classification_report(y_test, test_predict)
    st.text("Classification Report:")
    st.text(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_predict)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Neutral', 'Negative'],
                yticklabels=['Positive', 'Neutral', 'Negative'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

    # ROC and AUC curve
    y_test_bin = pd.get_dummies(y_test)
    y_pred_prob = nb.predict_proba(X_test)
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, label in enumerate(y_test_bin.columns):
        fpr[label], tpr[label], _ = roc_curve(y_test_bin[label], y_pred_prob[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])

    plt.figure()
    for label in y_test_bin.columns:
        plt.plot(fpr[label], tpr[label], lw=2, label=f'{label} (AUC = {roc_auc[label]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot()

    # Precision-Recall curve
    precision = {}
    recall = {}
    pr_auc = {}
    for i, label in enumerate(y_test_bin.columns):
        precision[label], recall[label], _ = precision_recall_curve(y_test_bin[label], y_pred_prob[:, i])
        pr_auc[label] = auc(recall[label], precision[label])

    plt.figure()
    for label in y_test_bin.columns:
        plt.plot(recall[label], precision[label], lw=2, label=f'{label} (AUC = {pr_auc[label]:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    st.pyplot()

    # Calculate precision, recall, and f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_predict, average='weighted')
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")

# Main app layout
st.title("Sentiment Analysis System")
tabs = st.tabs(["Home", "Text Cleaning", "Text Preprocessing", "Sentiment Analysis", "Model Evaluation"])

with tabs[0]:
    st.header("Welcome to the Sentiment Analysis System")
    st.write("""
    This application allows you to clean, preprocess, analyze sentiment, and evaluate models on textual data.
    Use the tabs above to navigate through the different stages of text analysis.
    """)
    image = Image.open(background_image_path)
    st.image(image, width=700)  # Resize image to width 700px

with tabs[1]:
    text_cleaning()

with tabs[2]:
    text_preprocessing()

with tabs[3]:
    sentiment_analysis()

with tabs[4]:
    model_evaluation()
