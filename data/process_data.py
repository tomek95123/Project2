import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    ''' 
    Preforms inner join operation on the provided tables 
    
    Input arguments:
    messages_filepah: path to messages csv file
    categories_filepath: path to categories csv file
    
    Returns:
    df: a dataframe, messages joined with categories on 'id' column
    '''
    # read messages data table
    messages = pd.read_csv(messages_filepath)
    # read categories data table
    categories = pd.read_csv(categories_filepath)
    # merge both tables
    df = pd.merge(messages, categories, on=['id', 'id'])
    
    return df

def clean_data(df):

    ''' 
    Performs cleaning operations on the provided table. 
        
    Input arguments:
    df: a dataframe with categories column
    
    Returns:
    df: dataframe after cleaning operations
    '''
    
    # split column with categories information into multiple columns, one column per category 
    categories = df.categories.str.split(';',expand = True)
    # select first row of categories dataframe
    row = categories.iloc[0]
    # extract a list of new column names
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames
    # for each column, convert category values to numbers 0 or 1
    for column in categories:
        
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # first category column contains relevant 1 and 2 values. Lets unify them and set to 1
    categories.loc[categories['related'] == 2, 'related'] = 1
    # delete categories column 
    df.drop('categories', axis = 1, inplace = True)
    # merge dataframe with created categories 
    df = pd.concat([df, categories], axis=1)
    # remove duplicates from dataframe
    df=df[df.duplicated()==False]

    return df

def save_data(df, database_filename):
    
    '''
    Creates an SQLite database object and saves a dataframe into it.

    Input arguments:
    df: a dataframe to save in a database
    
    database_filename: path in which the database will be created

    '''
    # create sql engine
    engine = create_engine(f'sqlite:///{database_filename}')
    # save clean dataframe into SQLite database
    df.to_sql('messages_table', engine, index=False, if_exists='replace')
    
def main():

    '''
    Performs loading data, cleaning and saving into SQLite database
    '''

    if len(sys.argv) == 4:
        
        # get input arguments provided in terminal
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        # load and merged data table
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        # clean dataframe
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        # save dataframe into SQLite database
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
