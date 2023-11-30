import pandas as pd


def import_excel():
    # Replace 'your_file.csv' with the path to your CSV file
    file_path = 'Dry_Bean_Dataset.xlsx'



    # Read the CSV file
    data = pd.read_excel(file_path)


    return data
