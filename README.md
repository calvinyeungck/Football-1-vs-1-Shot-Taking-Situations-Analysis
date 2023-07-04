# Analyzing-Two-Agent-Interaction-in-Football-Shot-Taking-Situations

## Calculate xSOT and xOSOT 
1. Download data from StatsBomb and preprocess the data
```
python data/data_download_plot.py
```
1.5 Optional: Split the dataset into train and test set
```
python data/train_test_split.py
```
2. Get the xSOT and counterfactual xSOT value
```
python game_theory.py -d <dataset_path> -o <output_path>
```

## xSOT and xOSOT based on the data

  xSOT = real_df.loc[index, "xSOT"] 
  
  xOSOT = real_df.loc[index, "xOSOT"]
## xSOT and xOSOT based on the counterfactual analysis

  xSOT = cf_df.loc[index, "xSOT"]
  
  xOSOT = cf_df.loc[index, "xOSOT"]

## Reference
For technical details and full experimental results, please check the [paper](https://arxiv.org/abs/). Please consider citing our work if you find it helpful to yours:

```
@misc{

}
```
