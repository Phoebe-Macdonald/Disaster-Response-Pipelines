import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - filepath to csv file containing messages and message ids
    categories_filepath - filepath to csv file containing categorisation of messages and message ids
    OUTPUT:
    df - dataframe of messages with binarised categorisation

    Loads and combines messages and message catgorisation datasets
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')

    # expand categorisation so that each category is a named column and each row is a message.
    # Cells contain either 1 or 0 depending on whether that category is applicable
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.slice(stop=-2)
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.strip().str[-1]

    # original dataframe with expanded categorisation
    df.drop('categories', axis=1, inplace=True)


    return df


def clean_data(df):
    '''
    INPUT:
    df - dataframe potentially with duplicates
    OUTPUT:
    df - dataframe with no duplicates

    Removes duplicated rows from a dataframe
    '''
    df = df[~df.duplicated()]
    return df


def save_data(df, database_filename):
    '''
    INPUT:
    df - dataframe to save in sql database
    database_filename = desired name of file within sql database

    Saves a dataframe with given filename within filepath: Disaster-Response-Pipelines/data/DisasterResponseDb.db
    '''
    engine = create_engine('sqlite:///data/DisasterResponseDb.db')
    df.to_sql(database_filename, engine, index=False)


def main():
    '''
    INPUT:
    messages_filepath - filepath to csv file containing messages and message ids
    categories_filepath - filepath to csv file containing categorisation of messages and message ids
    database_filename = desired name of file within sql database

    Extracts, transforms and loads categorised message data to sql database:
    Disaster-Response-Pipelines/data/DisasterResponseDb.db
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
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