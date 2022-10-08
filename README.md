# Disaster Response Pipeline Project

The aim of this project is to analyse disaster data from Appen (formally Figure 8) in order to build a model for an API that classifies diseaster messages.

### Instructions:
In order to run ths application go to the 'app' folder in terminal and type run.py
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
