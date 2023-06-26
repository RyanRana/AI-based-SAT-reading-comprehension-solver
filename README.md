# AI-based-SAT-reading-comprehension-solver
This code uses the nltk library for text preprocessing and the sklearn library for vectorizing the text and calculating cosine similarity. It first preprocesses the passage and questions by tokenizing the text, removing stop words, and converting everything to lowercase. Then, it vectorizes the passage and questions using TF-IDF (Term Frequency-Inverse Document Frequency). Finally, it calculates the cosine similarity between the passage and each question and selects the answer from the passage based on the highest similarity.

Please note that the code assumes that the answer to each question can be found in a separate sentence within the passage. You may need to modify the code or use more advanced techniques if the answers require more complex reasoning or if the answers span multiple sentences.

Remember to adapt and improve this code according to your specific requirements and data.


In this code, we first preprocess the passage and questions as before. Then, we vectorize the passage and questions using TF-IDF and calculate cosine similarity.

Next, we prepare the training data by reshaping the similarities array and creating target labels.

We then build a neural network model using the Keras Sequential API. The model consists of three dense layers with ReLU activation. The input shape is (1,) because we have only one feature (cosine similarity) as input.

We compile the model with the Adam optimizer and sparse categorical cross-entropy loss, and train it using the training data.

Finally, we use the trained model to predict the answers for the questions and print the predicted answers.

Please note that this is a simplified example, and you may need to adjust the architecture, hyperparameters, and data representation based on your specific requirements and dataset.
