# Spam or Ham Classifier

## Project Overview

This project is a machine learning-based system for classifying SMS messages as either "spam" or "ham" (not spam). The project involves data preprocessing, exploratory data analysis (EDA), and building a deep learning model using PyTorch for classification. The model is trained to distinguish between spam and ham messages based on the text content of the messages.

## Project Structure

- **Data Preprocessing**: The raw text data is cleaned, tokenized, and vectorized to be used as input for the model. This involves removing unwanted characters, converting text to lowercase, lemmatization, and transforming the text into a numerical format using `TfidfVectorizer`.
  
- **Exploratory Data Analysis (EDA)**: Plotly is used to visualize the distribution of message lengths, the frequency of common words in spam vs. ham messages, and other relevant features.

- **Model Training**: A deep learning model is built using PyTorch. The model consists of three fully connected layers with ReLU activation functions. The final layer uses a sigmoid function to output probabilities for the binary classification task.

- **Model Evaluation**: The model's performance is evaluated using accuracy, loss, and other relevant metrics on both the training and validation datasets.

- **Prediction**: A function is provided to predict whether a new SMS message is spam or ham using the trained model.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- scikit-learn
- Plotly
- pandas
- numpy
- nltk (for text preprocessing)

### Installing Required Packages

You can install the required packages using pip:

```bash
pip install torch scikit-learn plotly pandas numpy nltk
```

## Usage

### 1. Data Preprocessing

The preprocessing steps clean the text, lemmatize it, and vectorize it using `TfidfVectorizer`. The processed data is then split into training and validation datasets.

```python
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)
```

### 2. Exploratory Data Analysis (EDA)

Using Plotly, various visualizations are created to understand the distribution of data and the relationship between different features.

```python
fig_critics_hist = px.histogram(
    df, 
    x='Critics Score', 
    nbins=20, 
    title='Distribution of Critics Scores',
    labels={'Critics Score': 'Critics Score (%)'},
    color_discrete_sequence=['red']
)
fig_critics_hist.show()
```

### 3. Model Training

The `SpamClassifier` model is trained on the processed data using PyTorch.

```python
class SpamClassifier(nn.Module):
    def __init__(self, input_size):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output layer for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
```

### 4. Model Evaluation

The model is evaluated on the validation set to check its accuracy and other performance metrics.

```python
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
```

### 5. Making Predictions

You can use the trained model to predict whether a new SMS message is spam or ham.

```python
message = "Congratulations! You've won a free ticket to the Bahamas!"
prediction = predict_message(message, model, vectorizer)
print(f"The message is classified as: {prediction}")
```

## Conclusion

This project demonstrates how to build and train a spam classifier using deep learning. The model is capable of identifying spam messages with a good degree of accuracy after proper preprocessing and training. This project can be further improved by tuning the model, experimenting with different text preprocessing techniques, and using more advanced deep learning architectures.

## License

This project is licensed under the MIT License.
