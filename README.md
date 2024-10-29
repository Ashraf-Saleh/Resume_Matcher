# Resume Job Matching Model

This project is designed to match resumes with job categories based on the resume's content, specifically targeting roles like **Data Science, Database, DevOps Engineer, DotNet Developer, Java Developer, Python Developer, Testing, and Web Designing**. It uses Natural Language Processing (NLP) techniques to clean, vectorize, and classify resume text into job categories with an optional grouping of less relevant categories as "Other."

## Table of Contents

- [Resume Job Matching Model](#resume-job-matching-model)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Data Source](#data-source)

## Project Overview

The main goal of this project is to create a machine learning model capable of predicting the most relevant job category for a given resume text. It uses a **Naive Bayes classifier with One-vs-Rest strategy** for multiclass classification. The resumes are first cleaned and preprocessed, and then transformed into feature vectors to be used by the model.

## Dataset

The dataset used consists of resumes classified into various job categories. The dataset file should be named `UpdatedResumeDataSet.csv` and stored in a `data/` directory. Each resume entry includes:
- **Category**: The target job category.
- **Resume**: Text content of the resume.

## Installation

To set up the environment, clone the repository and install the dependencies from `requirements.txt`:

```bash
git clone https://github.com/your_username/Resume-Job-Matching.git
cd Resume-Job-Matching
pip install -r requirements.txt
```

Additionally, download the NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```
## Usage

To run the code, ensure that your dataset is loaded, then execute each of the code cells in the notebook. Here’s an outline of steps:

1. **Load Dataset**: Load `UpdatedResumeDataSet.csv` into a pandas DataFrame.
   
   ```python
    final_df = pd.read_csv('data/UpdatedResumeDataSet.csv')
    ```
2. **Clean Resumes**: Use the `clean_resume` function to remove unnecessary characters, URLs, and stopwords from the resumes to ensure the text is suitable for analysis and model training. This function will process each resume by stripping out irrelevant information, punctuation, and common words that do not contribute to the meaning of the text.
   ```python
    final_df['Resume'] = final_df['Resume'].apply(clean_resume)
    ```

3. **Encode Job Categories**: Encode the job categories using `LabelEncoder` from scikit-learn to convert categorical labels into numeric format. This step is essential for machine learning models, which require numerical input for training.
   ```python
    label_encoder = LabelEncoder()
    final_df['Category'] = label_encoder.fit_transform(final_df['Category'])
    ```

4. **Vectorize Text Data**: Transform the cleaned resume text into numerical feature vectors using `CountVectorizer`. This process converts the text data into a matrix of token counts, enabling the model to understand and analyze the text.
   ```python
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(final_df['Resume'])
    y = final_df['Category']
    ```

5. **Split Data**: Divide the dataset into training and testing sets using `train_test_split`. This step is critical for evaluating the model's performance, allowing you to train on one portion of the data while validating on another.
   ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    ```

6. **Train the Model**: Utilize a One-vs-Rest `MultinomialNB` classifier to train the model on the resume vectors. This classification approach allows the model to learn how to distinguish between multiple job categories based on the features derived from the resumes.
   ```python
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.multiclass import OneVsRestClassifier

    model = OneVsRestClassifier(MultinomialNB())
    model.fit(X_train, y_train)
    ```

7. **Evaluate the Model**: Make predictions on the test set and assess the model's accuracy and performance using metrics such as accuracy score and classification report. This evaluation helps determine how well the model generalizes to unseen data.
   ```python
    from sklearn.metrics import accuracy_score, classification_report

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    ```

8. **Predict on New Data**: Test the trained model with a new resume to obtain predicted job category matches. The function will output the match percentages for each category, providing insights into the most suitable job roles for the candidate based on their resume.
    ```python
    new_resume = "Your resume text here..."
    predication_func(new_resume)
    ```

## Code Description
The code is structured to perform resume categorization using machine learning techniques. Below is a detailed description of the main functions and their respective inputs and outputs.

### 1. `clean_resume(resume_text: str) -> str`
- **Description**: Cleans the resume text by removing URLs, unnecessary characters, stop words, and punctuation. It also converts the text to lowercase and removes any numeric values.
- **Input**: 
  - `resume_text`: A string containing the raw resume text.
- **Output**: 
  - A cleaned string of text suitable for further processing.

### 2. Data Preparation
- **Description**: This section loads the dataset, applies the cleaning function to the resumes, and encodes the job categories. The resumes are then vectorized using `CountVectorizer` to convert them into a numerical format suitable for machine learning models.
- **Input**: 
  - A CSV file containing the resumes and their respective categories.
- **Output**: 
  - Vectorized features (`X`) and encoded labels (`y`).

### 3. `train_test_split`
- **Description**: Splits the dataset into training and testing subsets to ensure the model can be evaluated on unseen data.
- **Input**: 
  - Vectorized features (`X`) and encoded labels (`y`).
- **Output**: 
  - `X_train`, `X_test`, `y_train`, `y_test`: Data subsets for training and testing the model.

### 4. Model Training
- **Description**: Trains a One-vs-Rest `MultinomialNB` model on the training data.
- **Input**: 
  - `X_train`: Training features.
  - `y_train`: Training labels.
- **Output**: 
  - A trained model ready for predictions.

### 5. `predication_func(new_resume: str)`
- **Description**: Takes a new resume as input, cleans it, vectorizes it, and generates category probabilities. It then calculates the percentage match for specified categories and groups others under "Other".
- **Input**: 
  - `new_resume`: A string containing the resume text to be predicted.
- **Output**: 
  - A printed list of category matches with their corresponding probabilities, including an "Other" category for non-specified roles.


### Vectorization
CountVectorizer is used to convert text data into numerical feature vectors.

### Label Encoding
Converts job categories to numeric labels for training.

### Model
MultinomialNB classifier in a One-vs-Rest configuration.

### Evaluation
The model is evaluated with an accuracy score and classification report.

## Model Evaluation
The model provides an accuracy score and classification report on the test set. Accuracy and additional metrics (precision, recall, F1-score) can be printed out to assess performance.

## Sample Prediction
When given a new resume, the code will output the match percentage for each specified job category. Categories outside the specified list will be grouped as "Other."

EXAMPLE output:
```
Top category matches:
Data Science: 75.20%
Other: 24.80%

```
## Data Source
The data used for training the model is sourced from Kaggle. The dataset file is named `UpdatedResumeDataSet.csv`. You can find it at the following link: [Updated Resume Dataset](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset).
