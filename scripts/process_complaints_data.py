import sys
import json
import re
import pandas as pd
import numpy as np
import en_core_web_sm
nlp = en_core_web_sm.load()
from sqlalchemy import create_engine
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


def clean_text(sent):
    """
    clean_text - remove punctuation and alpha numeric text from each complaint
    input: raw text
    output: clean text
    """

    sent = sent.lower() # Text to lowercase
    pattern = '[^\w\s]' # Removing punctuation
    sent = re.sub(pattern, '', sent) # Replace punctuation with empty string
    pattern = '\w*\d\w*' # Removing words with numbers in between
    sent = re.sub(pattern, '', sent) # Replace words with numbers in between with empty string
    return sent


def lemmmatize_text(text):
    """
    lemmmatize_text - generalize text from each complaint
    input: raw text
    output: lemmmatized text
    """

    sent = []
    doc = nlp(text)
    for token in doc:
        sent.append(token.lemma_)
    return " ".join(sent)


def get_POS_tags(text):
    """
    get_POS_tags - get the parts of speech tag for each word.
    input: raw text
    output: tagged text
    """
    sent = []
    blob = TextBlob(text)
    sent = [word for (word,tag) in blob.tags if tag=='NN']
    return " ".join(sent)


def load_data(complaints_filepath):
    """
    load_data - reads the raw json file and loads this into a dataframes
    input: a json file and it's path : data/complaints.json
    output: dataframe
    """

    print("load_data - loading data files {}".format(complaints_filepath))
    f = open(complaints_filepath)

    # returns JSON object as a dictionary
    data = json.load(f)
    complaints_df = pd.json_normalize(data)

    return complaints_df


def enhance_data(df):
    """
    enhance_data - reads dataframe and cleans the complaints and removes rows with no data
    input: dataframe
    output: dataframe
    """

    # text pre-processing
    complaints_df = df[["_id", "_source.complaint_what_happened"]]
    complaints_df.columns = ["id", "complaint_what_happened"]

    # assign nan in place of blanks in the complaints column
    complaints_df[complaints_df.loc[:, "complaint_what_happened"] == ""] = np.nan

    # remove all rows where complaints column is nan
    complaints_df = complaints_df[~complaints_df["complaint_what_happened"].isnull()]

    # convert complaint_what_happened column to string for performing text operations
    complaints_df["complaint_what_happened"] = complaints_df["complaint_what_happened"].astype(str)

    clean_complaints_df = pd.DataFrame(complaints_df["complaint_what_happened"].apply(clean_text))

    # create a dataframe('clean_complaints_df') that will have only the complaints and the lemmatized complaints
    clean_complaints_df['complaint_lemmatized'] = clean_complaints_df['complaint_what_happened'].apply(lemmmatize_text)

    # Extract Complaint after removing POS tags
    # python -m textblob.download_corpora
    clean_complaints_df['complaint_POS_removed'] = clean_complaints_df['complaint_lemmatized'].apply(get_POS_tags)

    # Removing -PRON- from the text corpus
    clean_complaints_df['Complaint_clean'] = clean_complaints_df['complaint_POS_removed'].str.replace('-PRON-', '')

    # The personal details of customer has been masked in the dataset with xxxx.
    # Let's remove the masked text as this will be of no use for our analysis
    clean_complaints_df['Complaint_clean'] = clean_complaints_df['Complaint_clean'].str.replace('xxxx','')

    return clean_complaints_df


def label_data(df):
    """
    label_data - reads dataframe and adds label to each complaint using topic modeling
    input: dataframe
    output: dataframe
    """

    # Write your code here to initialise the TfidfVectorizer
    tfidf = TfidfVectorizer(min_df=2, max_df=0.95, stop_words='english')

    # Write your code here to create the Document Term Matrix by transforming the complaints column present in df_clean.
    dtm = tfidf.fit_transform(df['Complaint_clean'])

    # Load your nmf_model with the n_components i.e 5
    num_topics = 5

    # keep the random_state =40
    nmf_model = NMF(n_components=num_topics, random_state=40)

    nmf_model.fit(dtm)

    # Create the best topic for each complaint in terms of integer value 0,1,2,3 & 4
    topic_results = nmf_model.transform(dtm)

    # Assign the best topic to each of the cmplaints in Topic Column
    df['Topic'] = topic_results.argmax(axis=1)

    return df


def save_data(df, database_filename):
    """
    save_data - saves a dataframe to a sql lite db file and table
    input: dataframe and the sql lite db file
    output: N/A
    """
    print("save_data - write dataframe to database {}".format(database_filename))
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('complaints', engine, index=False, if_exists='replace')

    # Check Connection and table
    check_query_df = pd.read_sql_query("SELECT * FROM complaints", con = engine).head()
    print("save_data - checking database access and table complaints {}".format(check_query_df.shape))


def main():
    #   To run ETL pipeline that load data from file and stores in database
    #   `python scripts/process_complaints_data.py data/complaints.json data/FinancialComplaints.db`
    if len(sys.argv) == 3:

        complaints_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    COMPLAINTS: {}'
        .format(complaints_filepath))
        complaints_df = load_data(complaints_filepath)

        print("Enhancing data...\n  DataFrame: {}".format("complaints_df"))
        clean_complaints_df = enhance_data(complaints_df)

        print("Labeling data...\n  DataFrame: {}".format("clean_complaints_df"))
        clean_complaints_df = label_data(clean_complaints_df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(clean_complaints_df, database_filepath)

        print('Loaded data from json file and saved to database!')

    else:
        print('Please provide the filepaths of the complaints ' \
              'dataset as the first argument, as ' \
              'well as the filepath of the database to save the loaded data ' \
              'to as the third argument. \nExample: python process_compalaints_data.py ' \
              'complaints.json ' \
              'FinancialComplaints.db')


if __name__ == '__main__':
    main()