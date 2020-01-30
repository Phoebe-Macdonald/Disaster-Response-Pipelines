# Disaster Response Pipeline Project

## Motivation:
The following files enable the user to build a ETL pipeline and then build a model which classifies messages received 
during a disaster. The model can then be used to categorise messages in an app to aid visualisation.
The use case for this is to enable disaster response teams to identify relevant messages to their specialism
and respond to them quickly. 

## Getting Started:
In order to run the following code you will need to:
- Install all prerequisites below
- Navigate to the project's root directory:
    - Run the following code in the terminal. This will initiate the ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - Run the following code in the terminal. This will run the ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
- Navigate to app directory:
    - Run the following code in the terminal. This will create an app which classifies of messages and visualises results
    `python run.py`
- In a web browser, go to http://0.0.0.0:3001/ to view the app output

### Prerequisites:

#### Manage the system
import sys

#### Data preparation
import pandas as pd

#### Importing and exporting data from databases
import sqlalchemy
from sqlalchemy import create_engine

#### NLP processing techniques
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#### Modelling and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

#### Managing text data preparation and modelling steps
from sklearn.pipeline import Pipeline

#### Importing and exporting model
import pickle


### Files:
This repo contains the following folders and files
- Data - this folder contains files for the ETL pipeline
    - disaster_messages.csv - contains message ids and messages
    - disaster_categories.csv - contains message ids and message categorisation
    - process_data.py - running this file will run ETL pipeline
    
- Models - this folder contains file for the ML pipeline
    - train_classifier.py - running this fill will run ML pipeline
    
- App - this folder contains file to create the app to visualise model outputs
    - templates - this folder contains template files to aid creation of the app
    - run.py - this file creates the app




