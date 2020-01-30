# import libraries
import sys
import pandas as pd
import sqlalchemy
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - filepath of file within sql database data/DisasterResponseDb.db
    OUTPUT:
    X - array of message data (input data for model)
    Y - array of categorisation of messages (target variables)
    category_names - names of target variables

    Loads data from sql database and extracts information for modelling
    '''
    engine = sqlalchemy.create_engine('sqlite:///data/DisasterResponseDb.db')
    df = pd.read_sql_table(database_filepath, engine)

    X = df['message']
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns
    return X, Y, category_names

def tokenize(text):
    '''
    INPUT:
    text - string of text
    OUTPUT:
    lemmed_words - list of words in input string, prepared for the next stages of NLP

    Prepares a string for NLP
        - removes punctuation
        - all characters to lower case
        - splits sentence into words
        - common english words removed
        - words returned to stem form
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed_words


def build_model():
    '''
    OUTPUT:
    NLP model

    Defines a pipeline consisting of NLP processing techniques (including a vectorizior for preparation and
    transformer for feature extraction) and creation of a Random Forest classifier.
    Performs grid search to find optimal parameters for pipeline.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])
    parameters = {
         'vect__ngram_range': ((1, 1), (1, 2)),
         #'vect__max_df': (0.5, 0.75, 1.0),
         #'vect__max_features': (None, 5000, 10000),
         #'tfidf__use_idf': (True, False),
         #'clf__estimator__max_features': ('auto', 'none'),
         #'clf__estimator__n_estimators': [10, 20],
         'clf__estimator__min_samples_leaf': [1, 10]
     }
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - model to apply to testing dataset
    X_test - test input data
    Y_test - test output data
    category_names - names of target variables

    Uses classifer to predict categorisations of input message data. Evaluates predictions for all messages vs actual
    categorisation in terms of precision, recall and R1 scores
    '''
    predictions = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i], classification_report(Y_test[:, i], predictions[:, i]))


def save_model(model, model_filepath):
    '''
    INPUT:
    model - model to save in given filepath
    model_filepath - desired location to save model

    Saves model to given filepath
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    INPUT:
    database_filepath - filepath of file within sql database data/DisasterResponseDb.db
    model_filepath - desired location to save model

    Loads prepared data and trains classifier which gets saved to desired location
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()