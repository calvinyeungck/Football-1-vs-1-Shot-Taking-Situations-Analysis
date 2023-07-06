# Analyzing-Two-Agent-Interaction-in-Football-Shot-Taking-Situations

## Calculate xSOT and xOSOT 
0. Install required package
```
pip install -r requirements.txt
```
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
3. Analyze the shot-taking situations with xSOT
```
python xsot_analysis -r <row_to_analyze> -d <real_df_path> -o <output_path>
```

## xSOT and xOSOT based on the data (step 2)

  xSOT = real_df.loc[index, "xSOT"] 
  
  xOSOT = real_df.loc[index, "xOSOT"]
## xSOT and xOSOT based on the counterfactual analysis (step 2)

  xSOT = cf_df.loc[index, "xSOT"]
  
  xOSOT = cf_df.loc[index, "xOSOT"]

## Analysis of shot-taking situations with xSOT
<img src="https://github.com/calvinyeungck/Analyzing-Two-Agent-Interaction-in-Football-Shot-Taking-Situations/blob/main/analysis/testing_plot_21.png" alt="alt text" width="698.5" height="400">


## Reference
For technical details and full experimental results, please check the [paper](https://arxiv.org/abs/). Please consider citing our work if you find it helpful to yours:

```
@misc{

}
```
