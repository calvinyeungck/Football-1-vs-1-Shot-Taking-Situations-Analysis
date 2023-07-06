# Analyzing-Two-Agent-Interaction-in-Football-Shot-Taking-Situations

## Calculate xSOT and xOSOT 
0. Install required package
```
pip install -r requirements.txt
```
1. Download data from StatsBomb and preprocess the data
```
python data/data_download_plot.py -o <output_dir>
```
1.5 Optional: Split the dataset into train and test set
```
python data/train_test_split.py -d <dataset_path> -o <output_dir>
```
2. Get the xSOT and counterfactual xSOT value
```
python game_theory.py -d <dataset_path> -o <output_dir>
```
3. Analyze the shot-taking situations with xSOT
```
python xsot_analysis -r <row_to_analyze> -d <real_df_path> -o <output_dir>
```

## xSOT and xOSOT based on the data (step 2)

  xSOT = real_df.loc[index, "xSOT"] 
  
  xOSOT = real_df.loc[index, "xOSOT"]
## xSOT and xOSOT based on the counterfactual analysis (step 2)

  xSOT = cf_df.loc[index, "xSOT"]
  
  xOSOT = cf_df.loc[index, "xOSOT"]

## Analysis of shot-taking situations with xSOT (step 3)
<img src="https://github.com/calvinyeungck/Analyzing-Two-Agent-Interaction-in-Football-Shot-Taking-Situations/blob/main/analysis/testing_plot_21.png" alt="alt text" width="558.8" height="320">

| Jersey Number | xSOT | P(Shot Off) | P(Shot Block) | P(Control) |
|---------------|------|-------------|---------------|------------|
| 9             | 0.27 | 0.32        | 0.22          | 0.59       |
| 20            | 0.23 | 0.60        | 0.03          | 0.63       |
| 14            | 0.17 | 0.67        | 0.16          | 0.99       |
| 12            | 0.15 | 0.63        | 0.20          | 0.89       |
| 6             | 0.05 | 0.53        | 0.18          | 0.17       |
| Shooter       | 0.03 | 0.51        | 0.46          | -          |
| 8             | 0.00 | 0.61        | 0.40          | 0.60       |


## Reference
For technical details and full experimental results, please check the [paper](https://arxiv.org/abs/). Please consider citing our work if you find it helpful to yours:

```
@misc{

}
```
