import pandas as pd
import numpy as np
from prediction_model.rule_based import calculate_shot_block_prob
from prediction_model.rule_based import is_point_within_area
import argparse
import torch
from torch import nn
import pdb
from prediction_model.shot_block_model_fc import Shot_block
from prediction_model.shot_off_model_fc import Shot_off
import os
from prediction_model.dataloader_fc import CustomDataset
import prediction_model.Metrica_PitchControl as mpc
import math
import matplotlib.pyplot as plt


def plot_row(index,attacker_i,offball_block_prob,row,path):
    offball_row=row.copy()
    for i in range(22):
        x, y = offball_row[f"player{i}_location_x"], offball_row[f"player{i}_location_y"]
        if offball_row[f"player{i}_teammate"]==True:
            plt.scatter(x, y, color='blue')
        elif offball_row[f"player{i}_position_name"]=="Goalkeeper":
            plt.scatter(x, y, color='yellow')
        else:
            plt.scatter(x, y, color='black')
    plt.scatter(offball_row['location_x'],offball_row['location_y'], color='red')
    x = [offball_row['location_x'], 120]
    y = [offball_row['location_y'], 30]
    plt.plot(x, y)
    x = [offball_row['location_x'], 120]
    y = [offball_row['location_y'], 50]
    plt.plot(x, y)
    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Multiple Points')

    # Save the plot as a PNG image
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+f'/plot_{index}_{attacker_i}_{offball_block_prob}.png')
    plt.clf()

def get_xSOT_xOSOT(dataset,verbose=False):
    df=dataset.copy()
    df['shotblock_y'] = df['shot_outcome_grouped'].apply(lambda x: 1 if x == 'Blocked' else 0)
    df['shotoff_y'] = df['shot_outcome_grouped'].apply(lambda x: 1 if x == 'Off T' else 0)
    df['shoton_y']=df['shot_outcome_grouped'].apply(lambda x: 1 if x == 'On T' else 0)

    #get the rule_based prediction features
        #hyperparameter for the rule_based feature
    optimized_params=[36.94630071, 12.3578812 ,  0.4997678,   0.15766148, -2.30980703]
    prediction_list = []
    for index, row in df.iterrows():
        #calculate the probability of the shot block for each row
        prediction = calculate_shot_block_prob(index, df, scaler_1=optimized_params[0], scaler_2=optimized_params[1], scaler_3=optimized_params[2], mean=0, sigma=optimized_params[3], a=optimized_params[4], b=-optimized_params[4])
        prediction_list.append(prediction)

    rule_based_prediction=pd.DataFrame(prediction_list)
    df["rule_based_prediction"]=rule_based_prediction[0]
    
    #features preprocessing
    coding_dict = dict(enumerate(pd.Categorical(df["position"]).categories))
    encoding_dict = {v: k for k, v in coding_dict.items()}
    df["position"] = pd.Categorical(df["position"]).codes
    

    #get shot block and shot off data set
    shot_block_features=["position", "location_x","location_y", "Dist2Goal", "Ang2Goal"]+["rule_based_prediction"]
    shot_block_target=["shotblock_y"]
    shot_block_df=df[shot_block_features+shot_block_target]

    shot_off_features=["position", "location_x","location_y", "Dist2Goal", "Ang2Goal"]
    shot_off_target=["shotoff_y"]
    shot_off_df=df[shot_off_features+shot_off_target]

    #get the dataset for the prediction model
    shot_block_dataset = CustomDataset(shot_block_df.loc[:, ~shot_block_df.columns.isin(shot_block_target)], shot_block_df[shot_block_target])
    shot_off_dataset = CustomDataset(shot_off_df.loc[:, ~shot_off_df.columns.isin(shot_off_target)], shot_off_df[shot_off_target])

    #model hyperparameter
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shot_block_hyperparameters = {
        'num_layers': [2],
        'hidden_dim': [64],
        'dropout_rate': [0.0],
        'activation': [nn.Sigmoid]
        ,'embedding1':[1]
    }
    shot_off_hyperparameters = {
        'num_layers': [1],
        'hidden_dim': [128],
        'dropout_rate': [0.0],
        'activation': [nn.Tanh]
        ,'embedding1':[2]
    }

    shot_block_input_dim=len(shot_block_features)-1+shot_block_hyperparameters["embedding1"][0]
    shot_block_model = Shot_block(num_layers=shot_block_hyperparameters["num_layers"][0],
                                input_dim=shot_block_input_dim, 
                                hidden_dim=shot_block_hyperparameters["hidden_dim"][0], 
                                dropout_rate=shot_block_hyperparameters["dropout_rate"][0], 
                                activation=shot_block_hyperparameters["activation"][0], 
                                embedding1=shot_block_hyperparameters["embedding1"][0],
                                class_weights=0, #not required for prediction
                                device=device).to(device)
    
    shot_off_input_dim=len(shot_off_features)-1+shot_off_hyperparameters["embedding1"][0]
    shot_off_model = Shot_off(num_layers=shot_off_hyperparameters["num_layers"][0],
                            input_dim=shot_off_input_dim, 
                            hidden_dim=shot_off_hyperparameters["hidden_dim"][0], 
                            dropout_rate=shot_off_hyperparameters["dropout_rate"][0], 
                            activation=shot_off_hyperparameters["activation"][0], 
                            embedding1=shot_off_hyperparameters["embedding1"][0],
                            class_weights=0, #not required for prediction
                            device=device).to(device)

    #load model
    #get the path of the current file
    current_path=os.path.dirname(os.path.abspath(__file__))
    shot_block_model_path=os.path.join(current_path, "prediction_model/shot_block.pt")
    shot_off_model_path=os.path.join(current_path, "prediction_model/shot_off.pt")

    shot_block_model_params=torch.load(shot_block_model_path)
    shot_off_model_params=torch.load(shot_off_model_path)

    shot_block_model.load_state_dict(shot_block_model_params)
    shot_off_model.load_state_dict(shot_off_model_params)

    with torch.no_grad():
        shot_block_model.eval()
        shot_block_pred_prob = shot_block_model(shot_block_dataset.data.to(device))
        shot_block_pred_prob = shot_block_pred_prob.cpu().numpy()

        shot_off_model.eval()
        shot_off_pred_prob = shot_off_model(shot_off_dataset.data.to(device))
        shot_off_pred_prob = shot_off_pred_prob.cpu().numpy()
    #shot block and shot off probability 
    df["shot_block_pred_prob"] = shot_block_pred_prob.flatten()
    df["shot_off_pred_prob"] = shot_off_pred_prob.flatten()
    df["xSOT"] = np.maximum(1-df["shot_block_pred_prob"]-df["shot_off_pred_prob"],0)

#calculate the xOSOT 
    params = mpc.default_model_params()
    prob_control_list=[]
    xOSOT_list=[]
    attacker_dict_list=[]
    for index, row in df.iterrows():
        if verbose:
            print(index+1,"/",len(df))
        attackers = []
        attacker_dict={}
        defenders = []
        ball_position=[row["location_x"]*105/120,row["location_y"]*68/80] #same as the starting location of the action
        gk_num=[np.nan,np.nan]
        #define the attacking and defending team players class
        for player_num in range(22):
            '''
            features for each other player (non-shooter and in the 360 freeze frame) in the dataset:
            row[f"player{player_num}_location_x"]
            row[f"player{player_num}_location_y"]
            row[f"player{player_num}_player_id"]
            row[f"player{player_num}_player_name"]
            row[f"player{player_num}_position_id"]
            row[f"player{player_num}_position_name"]
            row[f"player{player_num}_teammate"]
            '''
            if math.isnan(row[f"player{player_num}_location_x"]) and math.isnan(row[f"player{player_num}_location_y"]):
                continue
            pid=player_num
            team={"x":row[f"player{player_num}_location_x"]*105/120,"y":row[f"player{player_num}_location_y"]*68/80}
            if row[f"player{player_num}_teammate"]==True:
                teamname="Home" #attacking, the "Home" and "Away" are requirement from the original code
            else:
                teamname="Away" #defending
            if row[f"player{player_num}_position_name"]=='Goalkeeper':
                GKid=player_num
                if teamname=="Home":
                    gk_num[0]=GKid
                else:
                    gk_num[1]=GKid
            else:
                GKid=None
            team_player = mpc.player(pid,team,teamname,params,GKid)
            if row[f"player{player_num}_teammate"]==True:
                attackers.append(team_player)
            else:
                defenders.append(team_player)
        #remove offside attacking players
        all_attacker=attackers.copy()
        attackers=mpc.check_offsides(attacking_players=attackers, 
                                            defending_players=defenders,
                                            ball_position=ball_position,
                                            GK_numbers=gk_num,
                                            verbose=False, tol=0.2)
        for attacker_i in all_attacker:
            if attacker_i not in attackers:
                attacker_dict[attacker_i.id]=[1,0,np.nan,np.nan,np.nan,np.nan]#[offside,gk,xSOT,shotoff_prob,shot_block_prob,control_prob]
        #calculate pitch control
        for attacker_i in attackers: #loop for each feasible attacker
            if attacker_i.is_gk:
                attacker_dict[attacker_i.id]=[0,1,np.nan,np.nan,np.nan,np.nan]
            else:
                target_position= attacker_i.position# the location of the targeted offball attacker
                # defenders= #for both block and not blocking (remove the closest defender)
                #calculate the probabilty the offball player can control the ball and shoot at the current target position.
                prob_control=mpc.calculate_pitch_control_at_target(target_position=target_position, attacking_players=[attacker_i], defending_players=defenders, ball_start_pos=np.array(ball_position), params=params)
                attacker_dict[attacker_i.id]=[0,0,np.nan,np.nan,np.nan,prob_control]#[offside,xSOT,shotoff_prob,shot_block_prob,control_prob]
        #calculate the xSOT for offball players
        for attacker_i in attacker_dict.keys():
            if attacker_dict[attacker_i][0]==0 and attacker_dict[attacker_i][1]==0:
                offball_position=row[f"player{attacker_i}_position_name"]
                offball_location_x=row[f"player{attacker_i}_location_x"]
                offball_location_y=row[f"player{attacker_i}_location_y"]
                offball_Dist2Goal=(((offball_location_x-120)*105/120)**2+((offball_location_y-40)*68/80)**2)**0.5
                offball_Ang2Goal=np.abs(np.arctan2((40-offball_location_y)*68/80,(120-offball_location_x)*105/120))

                offball_row=row.copy()
                offball_row[f"player{attacker_i}_location_x"]=offball_row["location_x"]
                offball_row[f"player{attacker_i}_location_y"]=offball_row["location_y"]
                offball_row[f"player{attacker_i}_position_name"]=coding_dict.get(offball_row["position"])
                offball_row["location_x"]=offball_location_x
                offball_row["location_y"]=offball_location_y
                offball_row["position"]= encoding_dict[offball_position]
                offball_row["Dist2Goal"]=offball_Dist2Goal
                offball_row["Ang2Goal"]=offball_Ang2Goal

                offball_df = offball_row.to_frame().T
                offball_block_prob=calculate_shot_block_prob(index, offball_df, scaler_1=optimized_params[0], scaler_2=optimized_params[1], scaler_3=optimized_params[2], mean=0, sigma=optimized_params[3], a=optimized_params[4], b=-optimized_params[4])
                # plot_row(index,attacker_i,offball_block_prob,offball_row,path='/home/c_yeung/workspace6/python/project3/script/analysis/testing_image/')
                
                offball_df=pd.DataFrame({"position":offball_row["position"],"location_x":offball_row["location_x"],"location_y":offball_row["location_y"],"Dist2Goal":offball_row["Dist2Goal"],"Ang2Goal":offball_row["Ang2Goal"],"rule_based_prediction":offball_block_prob},index=[0])
                # pdb.set_trace()
                with torch.no_grad():
                    shot_block_model.eval()
                    shot_block_pred_prob = shot_block_model(torch.tensor(offball_df[shot_block_features].values).to(device))
                    shot_block_pred_prob = shot_block_pred_prob.cpu().numpy()
                    shot_block_pred_prob = shot_block_pred_prob.item()

                    shot_off_model.eval()
                    shot_off_pred_prob = shot_off_model(torch.tensor(offball_df[shot_off_features].values).to(device))
                    shot_off_pred_prob = shot_off_pred_prob.cpu().numpy()
                    shot_off_pred_prob = shot_off_pred_prob.item()
                attacker_dict[attacker_i][3]=shot_off_pred_prob#[offside,gk,xSOT,shotoff_prob,shot_block_prob,control_prob]
                attacker_dict[attacker_i][4]=shot_block_pred_prob

                attacker_dict[attacker_i][2]=max(1-shot_off_pred_prob-shot_block_pred_prob,0)*attacker_dict[attacker_i][5]

        #get the minimum xSOT
        xOSOT_values = [lst[2] for lst in attacker_dict.values() if not math.isnan(lst[2])]
        max_xOSOT = max(xOSOT_values) if len(xOSOT_values)!=0 else 0
        xOSOT_list.append(max_xOSOT)
        attacker_dict_list.append(attacker_dict)
    df["xOSOT"]=xOSOT_list
    df["attacker_dict"]=attacker_dict_list
    return df

def remove_closest_defender(df):
    cf_df=df.copy()
    have_defender_list=[]
    for index,row in cf_df.iterrows():
        pose_left_x=120
        pose_left_y=36
        pose_right_x=120
        pose_right_y=44

        #get the location of the shooter
        shoter_x=row["location_x"]
        shoter_y=row["location_y"]
        #get the location of the other players
        other_player={}
        for other_player_num in range(22):
            if not pd.isnull(row["player"+str(other_player_num)+"_location_x"]):
                # print(index,other_player,train["player"+str(other_player)+"_location_x"])
                other_player[other_player_num]={"x":row["player"+str(other_player_num)+"_location_x"],"y":row["player"+str(other_player_num)+"_location_y"],
                                            "teammate":row["player"+str(other_player_num)+"_teammate"],"position_name":row["player"+str(other_player_num)+"_position_name"]}

        #filter non teammate, Goalkeeper, and player not in between the shooter and the goal
        filtered_player = {k: v for k, v in other_player.items() if v.get('position_name') != 'Goalkeeper'}
        filtered_player = {k: v for k, v in filtered_player.items() if v.get('teammate') == False}
        keys_to_remove = []
        for k, v in filtered_player.items():
            filter=is_point_within_area((filtered_player[k]["x"], filtered_player[k]["y"]),[(pose_left_x, 30), (pose_right_x, 50), (shoter_x, shoter_y)])
            if not filter:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del filtered_player[k]

        if len(filtered_player)==0:
            have_defender_list.append(False)
            continue
        else:
            have_defender_list.append(True)
            #for each defender calculate the the distance to the shooter
            closest_distance=999
            for defender_i in filtered_player:
                distance=((filtered_player[defender_i]["x"]-shoter_x)**2+(filtered_player[defender_i]["y"]-shoter_y)**2)**0.5
                if closest_distance>distance:
                    closest_distance=distance
                    closest_defender_id=defender_i

            remove_defender_id=closest_defender_id
            cf_df.loc[index,f"player{remove_defender_id}_location_x"]=np.nan
            cf_df.loc[index,f"player{remove_defender_id}_location_y"]=np.nan
            cf_df.loc[index,f"player{remove_defender_id}_teammate"]=np.nan
            cf_df.loc[index,f"player{remove_defender_id}_position_name"]=np.nan
    cf_df["have_defender"]=have_defender_list
    
    return cf_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unprocessed_dataset_path","-d", help="data file path")
    parser.add_argument("--output_path","-o", help="path to save the df with prediction value")
    parser.add_argument("--verbose","-v", help="print out the progress", default=False)
    args = parser.parse_args()

    #get the xSOT and xOSOT
    data_set=pd.read_csv(args.unprocessed_dataset_path)
    df=data_set[:]
    real_df=get_xSOT_xOSOT(df,verbose=args.verbose)
    
    #get the counterfactual xSOT and xOSOT
    cf_df=df.copy() #conterfactual df
    cf_df=remove_closest_defender(cf_df)
    cf_df=get_xSOT_xOSOT(cf_df,verbose=args.verbose)

    #save the dfs
    real_df.to_csv(args.output_path+"real_df_v2.csv")
    cf_df.to_csv(args.output_path+"cf_df_v2.csv")

    #calculate the average xSOT and xOSOT
    have_defender_index=cf_df[cf_df["have_defender"] == True].index #n=1468
    print("n:",len(have_defender_index))

    xSOT=real_df.loc[have_defender_index, "xSOT"].mean() 
    xOSOT=real_df.loc[have_defender_index, "xOSOT"].mean()

    CxSOT=cf_df.loc[have_defender_index, "xSOT"].mean()
    CxOSOT=cf_df.loc[have_defender_index, "xOSOT"].mean()
    print("xSOT",xSOT,"xOSOT",xOSOT,"CxSOT",CxSOT,"CxOSOT",CxOSOT)

    









