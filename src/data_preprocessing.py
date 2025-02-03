import pandas as pd
import os
import string

def load_data(csv_path=None):
    """
    Load and preprocess the mental health dataset.

    If a valid CSV path is provided and the file exists, the function loads
    the dataset from that CSV. If the CSV contains a column named "Disorder",
    it will create a "symptoms" column by combining symptom columns (where the
    cell contains "yes") and set the "condition" column to the value from "Disorder".
    
    If csv_path is not provided or the file does not exist, a dummy dataset is used.

    Preprocessing steps:
      - Convert symptom descriptions to lowercase.
      - Remove punctuation from the descriptions.
      - Drop any rows with missing 'symptoms' or 'condition' values.

    Args:
      csv_path (str): Path to the CSV file with the dataset.

    Returns:
      pd.DataFrame: A DataFrame with preprocessed data containing columns "symptoms" and "condition".
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Check if the CSV file has the expected columns
        if 'Disorder' in df.columns:
            # Assume all columns except 'Disorder' are symptom indicators
            symptom_columns = [col for col in df.columns if col != 'Disorder']
            
            # Function to combine symptoms where the value is "yes"
            def combine_symptoms(row):
                symptoms_list = [
                    col.replace('.', ' ')  # replace periods for better readability
                    for col in symptom_columns 
                    if str(row[col]).strip().lower() == "yes"
                ]
                # If no symptom is indicated, return a default message
                return " ".join(symptoms_list) if symptoms_list else "no symptoms reported"
            
            df['symptoms'] = df.apply(combine_symptoms, axis=1)
            df['condition'] = df['Disorder']
            df = df[['symptoms', 'condition']]
    else:
        # Dummy dataset as fallback
        data = {
            'symptoms': [
                'I feel sad and hopeless', 
                'Constant worry and restlessness', 
                'Manic episodes and high energy', 
                'Loss of interest in daily activities',
                'Anxiety and panic attacks',
                'Severe mood swings and impulsiveness',
                'Feeling down and worthless',
                'Overwhelming fear and panic',
                'Rapid speech and racing thoughts',
                'Persistent sadness and fatigue'
            ],
            'condition': [
                'Depression', 'Anxiety', 'Bipolar', 'Depression', 
                'Anxiety', 'Bipolar', 'Depression', 'Anxiety', 
                'Bipolar', 'Depression'
            ]
        }
        df = pd.DataFrame(data)
    
    # Basic text preprocessing: convert to lowercase and remove punctuation.
    df['symptoms'] = df['symptoms'].apply(
        lambda x: x.lower().translate(str.maketrans('', '', string.punctuation))
    )
    df = df.dropna(subset=['symptoms', 'condition'])
    
    return df

if __name__ == '__main__':
    # Update the csv_path below with the correct location.
    df = load_data(csv_path="data/mental_health_data.csv")
    print(df.head())