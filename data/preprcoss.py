#did not used 
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pdb

data=pd.read_csv("/home/c_yeung/workspace6/python/project3/data/train.csv")
latent=pd.read_csv("/home/c_yeung/workspace6/python/project3/data/latent_vector.csv")
new_columns = ['latent_' + str(col) for col in latent.columns]
latent.columns = new_columns
data = pd.concat([data.reset_index(drop=True), latent], axis=1)
#features
main_features=["position",    "shot_aerial_won",   "shot_body_part",    "shot_first_time",    "shot_technique",    "under_pressure","shot_one_on_one",
                "shot_open_goal", "shot_follows_dribble",    "location_x","location_y", "Dist2Goal", "Ang2Goal"]
other_features=latent.columns.tolist()

features=main_features+other_features

df=data[features+["shot_outcome_grouped"]]
#target
df['shotoff_y'] = df['shot_outcome_grouped'].apply(lambda x: 1 if x == 'Off T' else 0)
#drop shot_outcome_grouped
df=df.drop(["shot_outcome_grouped"],axis=1)
df['shot_aerial_won'] = df['shot_aerial_won'].apply(lambda x: 1 if x == True else 0)
df['shot_first_time'] = df['shot_first_time'].apply(lambda x: 1 if x == True else 0)
df['shot_follows_dribble'] = df['shot_follows_dribble'].apply(lambda x: 1 if x == True else 0)
df['shot_open_goal'] = df['shot_open_goal'].apply(lambda x: 1 if x == True else 0)
df['under_pressure'] = df['under_pressure'].apply(lambda x: 1 if x == True else 0)
df['shot_one_on_one'] = df['shot_one_on_one'].apply(lambda x: 1 if x == True else 0)
result=["shotoff_y"]

train=df[:]

#main_feature preprocessing
encoder = OneHotEncoder()
scaler = StandardScaler()
#trainset one hot encoding
encoded_column = encoder.fit_transform(df[["position", "shot_body_part","shot_technique"]])
encoded_column = encoder.transform(train[["position", "shot_body_part","shot_technique"]])
    #replace the original column with the encoded version
column_names = ['{}_{}'.format(col, val) for col in ["position", "shot_body_part","shot_technique"] for val in df[col].unique()]
train = train.drop(["position", "shot_body_part","shot_technique"], axis=1)
train = pd.concat([train.reset_index(drop=True), pd.DataFrame(encoded_column.toarray(), columns=column_names)], axis=1)
#trainset standardization
train[["location_x","location_y", "Dist2Goal", "Ang2Goal"]]=scaler.fit_transform(train[["location_x","location_y", "Dist2Goal", "Ang2Goal"]])

#other_feature preprocessing
scaler1 = MinMaxScaler()
#standardization
train[other_features]=scaler1.fit_transform(train[other_features])

train.to_csv("/home/c_yeung/workspace6/python/project3/data/preprocessed_latent.csv",index=False)












