import sys
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
import re
import nltk
#nltk.download('omw-1.4')
import pickle

from sqlalchemy import create_engine
from datetime import datetime
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV


def load_data(database_filepath):
    """
    load_data - loads the complaints data from the SQL lite DB file to a dataframe
                and return the features.
    input: SQL lite DB file
    output: features dataframe
    """
    print("load_data - loading file {}".format(database_filepath))

    engine = create_engine('sqlite:///'+database_filepath)

    complaints_df = pd.read_sql_query("SELECT * FROM complaints", con = engine)
    training_complaints_data = complaints_df[['complaint_what_happened', 'Topic']]
    # training_complaints_data = training_complaints_data.set_index("Topic")
    X = training_complaints_data["complaint_what_happened"]
    Y = training_complaints_data["Topic"]

    return X, Y, [0, 1, 2, 3, 4]


def tokenize(text):
    """
    tokenize - tokenize a string
    input: the string to be processed and tokenized
    output: the tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(X_training_set, Y_training_set):
    """
    build_model - trained and tune an nlp pipeline
    input: Training dataset and labels
    output: nlp pipeline
    """

    nlp_pipeline_simple_lr = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multiclassifier',LogisticRegression(random_state=42))
    ])

    print("build_model - model {} \n\n on X_train {} and Y_train {}".format(nlp_pipeline_simple_lr, X_training_set.shape, Y_training_set.shape))

    logreg_grid = {"multiclassifier__C": [100, 10, 5, 4, 3, 2, 1, 1.0, 0.1, 0.01],
                   "multiclassifier__solver": ["liblinear"]}

    # Setup grid hyperparameter search for LogisticRegression
    logreg_hpt = GridSearchCV(nlp_pipeline_simple_lr, param_grid=logreg_grid, cv=5, verbose=True, n_jobs=1)

    print(nlp_pipeline_simple_lr.get_params().keys())

    # Fit random hyperparameter search model
    logreg_hpt.fit(X_training_set, Y_training_set)

    # summarize result
    print('"build_model - best Score: %s' % logreg_hpt.best_score_)
    print('"build_model - best Hyperparameters: %s' % logreg_hpt.best_params_)

    return logreg_hpt.best_estimator_


def evaluate_model(model, X_test, Y_test):
    '''
    evaluate_model - display classification report of the pipeline provided
    input: pipeline, traing and label sets as well as the labels
    output: N/A
    '''

    print("evaluate_model - model {}".format(model))

    predictions = model.predict(X_test)
    print("evaluate_model - classification report")
    print(classification_report(Y_test, predictions))


def save_model(model, model_filepath):
    '''
    save_model - dsave sthe trained pipleine and model to a pickle file
    input: pipleine and path to pickle file where the model will be saved
    output: N/A
    '''

    print("save_model - model {} to {}".format(model, model_filepath))
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    #     To run ML pipeline that trains classifier and saves
    #     `python scripts/train_complaints_classifier.py data/FinancialComplaints.db models/complaints_classifier.pkl`

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print("X -> {} Y -> {} Category - {}".format(X.shape, Y.shape, category_names))

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

        print('Building & Training model...')
        model = build_model(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()