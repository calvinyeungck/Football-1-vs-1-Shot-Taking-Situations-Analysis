# Football 1-vs-1 Shot Taking Situations Analysis
Complex interactions between two opposing agents frequently occur in various domains, and analyzing the strategies involved can be both beneficial and challenging. One such critical scenario is shot-taking in football, where decisions such as whether the attacker should shoot or pass the ball and if the defender should attempt to block the shot or not, play a crucial role in the outcome of the game. However, there are currently no effective and data-driven approaches to analyzing such situations. 
<p align="center">
  <img src="https://github.com/calvinyeungck/Analyzing-Two-Agents-Interaction-in-Football-Shot-Taking-Situations/blob/main/xSOT_concept.png" alt="alt text" width="433.9" height="397.5">
</p>
To address this gap, we have proposed a novel framework that integrates machine learning models, rule-based models, and game theory to analyze the optimal strategies in football shot-taking scenarios. Additionally, we have introduced a novel metric called xSOT to evaluate players' actions in these situations. Overall, we expect that this framework will contribute to and inspire the analysis of complex interaction situations, particularly in the context of sports, where understanding optimal strategies can greatly benefit teams and players.

## Calculate xSOT and xOSOT, and shot-taking situation analysis
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
## Theory-based: shot block model parameters
In the code, the parameter scaler_1, scaler_2, scaler_3, sigma, a corresponding to c_1, c_2, c_3, c_4, a in the paper.
```
calculate_shot_block_prob(index,train,scaler_1=1,scaler_2=1,scaler_3=1,mean=0,sigma=0.4,a=-10,b=10)
```

## xSOT and xOSOT when the closest defender blocking (step 2)

  xSOT = `real_df.loc[<index>, "xSOT"] `
  
  xOSOT = `real_df.loc[<index>, "xOSOT"]`
## xSOT and xOSOT when the closest defender not blocking  (step 2)

  xSOT = `cf_df.loc[<index>, "xSOT"]`
  
  xOSOT = `cf_df.loc[<index>, "xOSOT"]`

## Analysis of shot-taking situations with xSOT (step 3)
<p align="center">
<img src="https://github.com/calvinyeungck/Analyzing-Two-Agent-Interaction-in-Football-Shot-Taking-Situations/blob/main/analysis/testing_plot_21.png" alt="alt text" width="558.8" height="320">
</p>
<!--
| Jersey Number | xSOT | P(Shot Off) | P(Shot Block) | P(Control) |
|---------------|------|-------------|---------------|------------|
| 9             | 0.27 | 0.32        | 0.22          | 0.59       |
| 20            | 0.23 | 0.60        | 0.03          | 0.63       |
| 14            | 0.17 | 0.67        | 0.16          | 0.99       |
| 12            | 0.15 | 0.63        | 0.20          | 0.89       |
| 6             | 0.05 | 0.53        | 0.18          | 0.17       |
| Shooter       | 0.03 | 0.51        | 0.46          | -          |
| 8             | 0.00 | 0.61        | 0.40          | 0.60       |
-->
<div align="center">
  <table>
    <tr>
      <th>Jersey Number</th>
      <th>xSOT</th>
      <th>P(Shot Off)</th>
      <th>P(Shot Block)</th>
      <th>P(Control)</th>
    </tr>
    <tr>
      <td>9</td>
      <td>0.27</td>
      <td>0.32</td>
      <td>0.22</td>
      <td>0.59</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.23</td>
      <td>0.60</td>
      <td>0.03</td>
      <td>0.63</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.17</td>
      <td>0.67</td>
      <td>0.16</td>
      <td>0.99</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.15</td>
      <td>0.63</td>
      <td>0.20</td>
      <td>0.89</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.05</td>
      <td>0.53</td>
      <td>0.18</td>
      <td>0.17</td>
    </tr>
    <tr>
      <td>Shooter</td>
      <td>0.03</td>
      <td>0.51</td>
      <td>0.46</td>
      <td>-</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.00</td>
      <td>0.61</td>
      <td>0.40</td>
      <td>0.60</td>
    </tr>
  </table>
</div>

## Reference
For technical details and full experimental results, please check the [paper](https://arxiv.org/abs/). Please consider citing our work if you find it helpful to yours:

```
@misc{

}
```
