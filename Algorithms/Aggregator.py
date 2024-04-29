import os
import pandas as pd
import re
from .Preprocessor import main as preprocess_main 

def preprocess_files_in_directory(directory_path):
    # List to store dataframes
    dfs = []
    files = os.listdir(directory_path)

    for file_name in files:
        #skip .DS_Store file
        if file_name == '.DS_Store':
            continue
        # Regex to remove number from file name, Netflix1 = Netflix
        website = re.sub(r'\d*$', '', file_name.split(".csv")[0])
        file_path = os.path.join(directory_path, file_name)
        statistics_df = preprocess_main(file_path, website)
        dfs.append(statistics_df)
        
    # Merge all DataFrames into one big DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

# Save DataFrame to CSV
def save_statistics_to_csv(directory_path, output_file):
    statistics_df = preprocess_files_in_directory(directory_path)
    statistics_df.to_csv(output_file, index=False)
    print("Output saved to:", output_file)
    print('\n')

def main(directory_path = "./Data", output_filename = "statistics_output.csv"):
    save_statistics_to_csv(directory_path, output_filename)
