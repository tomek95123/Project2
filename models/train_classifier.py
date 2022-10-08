import sys
import pandas as pd
import numpy as np
import sklearn
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import ne_chunk
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):

    ''' 

    Loads data from a specified database 
        
    Inputs:
        - database_filepath: path to a SQLite database
        
    Returns:
        - X: Pandas dataframe with text message
        - Y: Pandas dataframe with categories
        - Y.columns: category names
    '''
    
    # create engine object
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # load messages table from SQLite database
    df = pd.read_sql_table('messages_table', con=engine)
    
    # split table into messages and category labels
    X = df['message'] 
    Y = df.drop(columns=['message', 'id', 'original', 'genre'])
    
    return X, Y, Y.columns

def tokenize(text):

    '''

    Performs cleaning operations: standardization, tokenizationon, 
    stop words removal and stemming a given text data
    
    Inputs:
        -text: Pandas series containing text data
    
    Returns:
        -clean: Pandas series containing clean text data
    '''
    
    # normalize text data
    text = text.lower()
    
    # tokenize text 
    tokens = word_tokenize(text)
    
    # remove stopwirds from text
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # apply stemming 
    stemmed = [PorterStemmer().stem(t) for t in tokens]
    
    # apply lammatization
    clean = [WordNetLemmatizer().lemmatize(s) for s in stemmed]
    
    return(clean)


def build_model():
    
    '''

    Initializes multiclass classification model
    
    Inputs:
        - None

    Returns:
        - Initialized model
    '''
    
    # create pipeline: CountVectorizer, TfidfTransformer and MultiOutputClassifier model based on RandomForestClassifier
    pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    
    Evaluates trained model. Prints classification report for each category column to the console.
    
    Inputs:
        - model: a classification model
        - X_test: tokenized messages
        - Y_test: labels

    Returns:
        - None
    '''
    
    # create predictions
    predicted = model.predict(X_test)

    # for each possible category print classification report to the console
    for i, col in enumerate(category_names):
        print(col, classification_report(Y_test[col], predicted[:,i]))
    

def save_model(model, model_filepath):
    
    '''
    
    Saves model parameters into pickle file.
    
    Inputs:
    
    - model: a model to save
    - model_filepath: path to a file where the model will be stored
    
    Returns:
        - None
    
    '''
    # create filename
    filename = 'final_model.sav'
    
    # save model into pkl file in the provided path
    pickle.dump(model, open(model_filepath, 'wb'))


def main():

    '''

    Loads datatable from the database. Splits data into training and test sets. Initializes, trains and 
    evaluates the model. Saves the model into a pickle file.
    
    '''
    
    # check whether a user provided correct number of arguments
    if len(sys.argv) == 3:
        
        # extract database_filepath and model_filepath from the provided arguments
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        # load messages dataframe and split into messages and categories
        X, Y, category_names = load_data(database_filepath)
        
        # split messages and categories into train and test set
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        # build model
        model = build_model()
        
        print('Training model...')
        # train model using train set
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        # evalueate model using evaluation set
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        # save model
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
