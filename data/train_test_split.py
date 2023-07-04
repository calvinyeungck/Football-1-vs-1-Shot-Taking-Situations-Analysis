import numpy as np
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split


df=pd.read_csv("dataset.csv")
train, test = train_test_split(df, test_size=0.2, stratify=df["shot_outcome_grouped"], random_state=22) 

print(len(train),len(test))

train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)

# pdb.set_trace()
