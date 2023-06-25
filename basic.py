import csv
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Rest of the code remains the same

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

# Answer the questions based on the highest similarity
for question, similarity in zip(questions, similarities[0]):
    print(f"Question: {question}")
    print(f"Answer: {passage.split('.')[int(similarity * 10) - 1].strip()}")
    print()
