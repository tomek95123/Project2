# Disaster Response Pipeline Project

## Project motivation

The aim of this project is to analyse disaster data from Appen (formally Figure 8) in order to build a model for an API that classifies diseaster messages provided by the user. The app also presents visualization of the distribution of training data

## Files description

app

| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

data

|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

models

|- train_classifier.py
|- classifier.pkl  # saved model 

README.md


## Instructions

1. Open a new terminal window and navigate to the app folder 
2. In order to run ETL pipeline type the following to the command line:
`python ../data/process_data.py ../data/disaster_messages.csv ../data/disaster_categories.csv ../data/DisasterResponse.db`
3. In order to run ML pipeline type the following to the command line:
`python ../models/train_classifier.py ../data/DisasterResponse.db ../models/classifier.pkl`
4. Once the model is saved type the following command in the command line in order to run the main app:
`python run.py`
5. Go to http://0.0.0.0:3000/

## Acknowledgements
Acknowledgements to Udacity for prepring the web app layout.
