import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df[['study_hours', 'attendance_percentage', 'previous_score']]
    y = df['pass']
    return X, y
