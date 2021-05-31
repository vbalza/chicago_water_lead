import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime

##### Read
def read_csv(file, date_cols=[]): 
    '''
    Reads a CSV file, and interprets specified columns as datetime
    
    Inputs: 
        file (string)
        date_cols (list of str)
    
    Outputs: 
        Pandas DataFrame
    '''
    return pd.read_csv(file, parse_dates=date_cols, 
        infer_datetime_format=True)    

##### Explore 
def show(df, head=5):
    '''
    Prints the shape and datatype of a Pandas DataFrame and returns the 
    dataframe head. 
    Inputs: 
        df (Pandas DataFrame)
        head (int) - Size of head to return
    Outputs: 
        dataframe 
    '''
    print()
    print('Shape:') 
    print(df.shape)
    print()
    print('Data Types:') 
    print(df.dtypes)
    
    return df.head(head)

def describe(df): 
    '''
    Runs Pandas.DataFrame.describe() returning summary statistics of all 
    numerical columns. 
    Inputs: 
        df (Pandas DataFrame)
    Outputs: 
        dataframe
    '''
    return df.describe()

def group_count(df, groupby, new_name='Count'): 
    '''
    Produces a Pandas Series showing the observation counts grouped by 'groupby'
    with parameter new_name for the new column. 
    Inputs: 
        df (Pandas DataFrame)
        groupby (str) - name of column to groupby
        new_name (str) - name of Pandas Series
    Outputs: 
        Pandas Series
    '''
    return (df.groupby(groupby, dropna=False)[groupby]
                .count().rename(new_name))

##### Training/Testing
def split(df, test_size=0.2, random_state=14, print_size=True): 
    '''
    Splits training and testing data according to the proportion of test_size
    and the random_state. 
    Inputs: 
        df (Pandas DataFrame)
        test_size (float) - proportion size of test data [0.0, 1.0]
        random_state (int) - for replication purposes
        print_size (Bool) - whether to print training and test shapes
    Outputs: 
        training, testing (Pandas DataFrames)
    '''
    training, testing = train_test_split(df, 
        test_size=test_size, random_state=random_state)
    if print_size: 
        print('Training Size:')
        print(training.shape)
        print()
        print('Testing Size:')
        print(testing.shape)
    
    return training, testing

##### Pre-Processing
def impute_missing(df): 
    '''
    Imputes missing values with the column median for columns that are 
    of numeric dtype (namely int, float, bool). Also drops any columns that are 
    all NaN.
    Inputs: 
        df (Pandas DataFrame)
    Outputs: 
        new df (Pandas DataFrame)
    '''
    numeric_mask = df.dtypes.apply(is_numeric_dtype)
    na_mask = df.isna().any()
    print(f'Contains NA Values:\n{na_mask}')
    col_mask = numeric_mask.combine(other=na_mask, func=min)
    
    medians = df.loc[:,col_mask].median().to_dict()
    
    return df.fillna(value=medians).dropna(axis=1, how='all')

def normalized_values(df, ignore=[], quiet=False): 
    '''
    Normalizes numeric columns using sklearn.preprocessing.StandardScaler()
    Inputs: 
        df (Pandas DataFrame)
        ignore (list of str) - list of numerical columns to ignore when normalizing. 
            Usually will include at least the target column. 
    Outputs: 
        scaler (StandardScaler()) for use in function normalize()
    '''
    df_working = df.drop(columns=ignore)
    numeric_mask = df_working.dtypes.apply(is_numeric_dtype)

    scaler = StandardScaler()
    scaler.fit(df_working.loc[:,numeric_mask])
    if not quiet: 
        print('Normalization Results:')
        print(list(df_working.loc[:,numeric_mask].columns))
        print('Column Means:')
        print(scaler.mean_)
        print('Column Variances:')
        print(scaler.var_)

    return scaler

def normalize(df, scaler, ignore=[], inplace=False):
    '''
    Normalizes all numeric columns not included in ignore using scaler. 
    Inputs: 
        df (Pandas DataFrame)
        scaler (sklearn.preprocessing.StandardScaler())
        ignore (list of str) - columns to ignore. 
        inplace (bool) - Whether to overwrite columns or concatenate new ones
            with normalized values
    Outputs: 
        df (Pandas DataFrame) - Either overwritten or with normalized columns 
            appended. 
    '''
    df_working = df.drop(columns=ignore)
    numeric_mask = df_working.dtypes.apply(is_numeric_dtype)
    
    if not inplace: 
        col_list = list(numeric_mask[numeric_mask].index)
        for index, col in enumerate(col_list):
            col_list[index] = col + '_norm'
        df_norm = pd.DataFrame(
            data=scaler.transform(df_working.loc[:,numeric_mask]), 
            columns=col_list, 
            index=df_working.loc[:,numeric_mask].index)
        return df.join(df_norm)

    df_working.loc[:,numeric_mask] = (scaler.transform(
        df_working.loc[:,numeric_mask]))
    return df.loc[:,ignore].join(df_working)

##### Generate Features
def get_dummies(df, df_original): 
    '''
    Creates dummy variables of all non-numeric features. 
    Inputs: 
        df (Pandas DataFrame) - dataframe with just columns to dummify. 
        df_original() - full dataframe to append dummified columns to. 
    Outputs: 
        df (Pandas DataFrame)
    '''
    return df_original.join(pd.get_dummies(df))

def cut(df, bins, right=True, labels=None): 
    '''
    Bin numerical columns in a dataframe. 
    Inputs: 
        df (Pandas DataFrame)
        bins (int) - Number of bins 
        right (bool) - Include right boundary in bins
        labels (list of str) - label names
    Outputs: 
        df (Pandas DataFrame)
    '''
    pd.cut(df, bins=bins, right=right, labels=labels)

##### Build Classifiers
def learn(target, training, testing, models, grid):
    '''
    Learn a series of models using a grid of parameters, evaluate models, and 
    report the total time elapsed. 
    Inputs: 
        target (str) - Name of target column
        training (Pandas DataFrame) - training dataframe
        testing (Pandas DataFrame) - testing dataframe
        models (dict) - dictionary of sklearn models
        grid (dict of lists) - grid of parameters to use in models. 
    Outputs: 
        results (Pandas DataFrame) - dataframe of models, parameters as str, and 
            various metrics. 
    '''
    # Begin timer 
    start = datetime.datetime.now()

    # Initialize results data frame 
    results = pd.DataFrame(columns=['Model', 'Parameters', 'Accuracy', 
        'Precision', 'Recall', 'Classification Report'])

    # Loop over models 
    for model_key in models.keys(): 
        
        # Loop over parameters 
        for params in grid[model_key]: 
            model_start = datetime.datetime.now()
            print("Training model:", model_key, "|", params)
            
            # Create model 
            model = models[model_key]
            model.set_params(**params)
            
            # Fit model on training set 
            model.fit(X=training.drop(columns=target), y=training.loc[:,target])
            
            # Predict on testing set 
            predictions = model.predict(testing.drop(columns=target))
            
            # Evaluate predictions 
            accuracy = metrics.accuracy_score(testing.loc[:,[target]], predictions)
            recall = metrics.recall_score(testing.loc[:,[target]], predictions)
            precision = metrics.precision_score(testing.loc[:,[target]], predictions)
            classification_report = metrics.classification_report(
                testing.loc[:,[target]], predictions)
            
            # Store results in your results data frame 
            result = pd.DataFrame(
                        data=np.array([[model_key, str(params), accuracy, 
                            recall, precision, classification_report]]), 
                        columns=["Model", "Parameters", "Accuracy", "Recall", 
                            "Precision", "Classification Report"])
            results = results.append(result, ignore_index=True)
            model_end = datetime.datetime.now()
            print('    Model Run Time:', model_end - model_start)
            
    # End timer
    stop = datetime.datetime.now()
    print("Total Time Elapsed:", stop - start)

    return results

def print_coefs(model, df, target, n=10): 
    '''
    Prints target name, intercept and top/bottom n feature names and coefficient 
    results from a model. Assumes that all non-target columns in dataframe are 
    features of model.
    Inputs: 
        model: scikit-learn model instance
        df (Pandas DataFrame) - used only to extract column names
        target (str) - Name of target column
        n (int) - top/bottom n features to print
    Outputs: 
        None, prints to screen. 
    '''
    if n > pd.get_option('display.max_rows'):
        pd.set_option('display.max_rows', n)

    series = (pd.Series(
                    data=model.coef_.reshape(-1,), 
                    index=df.drop(columns=target).columns)
                .sort_values(ascending=False))
    
    print(f'Target:\n{target}\n\nIntercept:\n{model.intercept_}\n')        
    
    if n >= model.coef_.shape[1]: 
        print(f'Features and Coefficients:\n{series}\n')
    else: 
        print(f'Top {n} Features and Coefficients:\n{series.head(n)}\n\n' + 
              f'Bottom {n} Features and Coefficients:\n{series.tail(n)}\n\n')
    pd.reset_option('display.max_rows')