import numpy as np
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path','-d', type=str)
parser.add_argument('--output_path','-o', type=str)
args = parser.parse_args()

df=pd.read_csv(args.data_path)
train, test = train_test_split(df, test_size=0.2, stratify=df["shot_outcome_grouped"], random_state=22) 

print(len(train),len(test))

train.to_csv(args.output_path+"train.csv",index=False)
test.to_csv(args.output_path+"test.csv",index=False)

# pdb.set_trace()
