import csv
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import layers, models

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Read passage and questions from CSV file
passage_file = 'passage.csv'
questions_file = 'questions.csv'

passage = ''
with open(passage_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        passage += ' '.join(row)

questions = []
with open(questions_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        questions.append(row[0])

# Preprocess the passage and questions
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

preprocessed_passage = preprocess_text(passage)
preprocessed_questions = [preprocess_text(question) for question in questions]

# Vectorize the passage and questions
vectorizer = TfidfVectorizer()
passage_vector = vectorizer.fit_transform([preprocessed_passage])
question_vectors = vectorizer.transform(preprocessed_questions)

# Calculate cosine similarity between passage and questions
similarities = cosine_similarity(passage_vector, question_vectors)

# Prepare training data
X_train = np.array(similarities[0]).reshape(-1, 1)
y_train = np.arange(len(questions)).reshape(-1, 1)

# Build the neural network model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(1,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(len(questions), activation='softmax'))

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Answer the questions
predictions = model.predict(X_train)
predicted_answers = [passage.split('.')[int(np.argmax(prediction))] for prediction in predictions]

# Print the predicted answers
for question, answer in zip(questions, predicted_answers):
    print(f"Question: {question}")
    print(f"Answer: {answer.strip()}")
    print()
