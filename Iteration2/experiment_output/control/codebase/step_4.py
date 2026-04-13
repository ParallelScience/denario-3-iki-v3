# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd

def main():
    data_dir = 'data/'
    file_path = os.path.join(data_dir, 'processed_data.csv')
    print('Loading processed dataset...')
    df = pd.read_csv(file_path, low_memory=False)
    print('Columns in the dataset:')
    for col in df.columns:
        print(col)

if __name__ == '__main__':
    main()