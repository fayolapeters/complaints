import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify

model_file_name = "complaints_classifier.pkl"
model_path = "../models/"+model_file_name

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load model
print("loading model {} ...".format(model_path))
model = pickle.load(open(model_path, 'rb'))


# index webpage receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # create the dictionary of Topic names and Topics
    topic_names = {0: "Bank account services",
                   1: "Credit card / Prepaid card",
                   2: "Others",
                   3: "Theft/Dispute reporting",
                   4: "Mortgages/loans"}

    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    print("generating classification prediction for message {}...".format(query))
    classification_labels = model.predict([query])[0]
    classification_labels = topic_names[classification_labels]
    print("labels {}...".format(classification_labels))
    # classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the index.html Please see that file.
    return render_template(
        'index.html',
        query=query,
        model=model[-1],
        classification_labels=classification_labels
    )


def main():
    app.run(host='0.0.0.0', port=8000, debug=True)


if __name__ == '__main__':
    main()