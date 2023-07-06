import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch, FontManager, Sbopen
import argparse

def plot_event(row_of_data,output_path):
    
    sample=row_of_data

    plt.style.use('ggplot')
    
    # get event and lineup dataframes 
    # event data
    parser = Sbopen()
    df_event, df_related, df_freeze, df_tactics = parser.event(sample.match_id)
    
    # lineup data
    df_lineup = parser.lineup(sample.match_id)
    df_lineup = df_lineup[['player_id', 'jersey_number', 'team_name']].copy()
    # pdb.set_trace()
    SHOT_ID = sample.id
    df_freeze_frame = df_freeze[df_freeze.id == SHOT_ID].copy()
    df_shot_event = df_event[df_event.id == SHOT_ID].dropna(axis=1, how='all').copy()
    
    # add the jersey number
    df_freeze_frame = df_freeze_frame.merge(df_lineup, how='left', on='player_id')
    
    # strings for team names
    team1 = df_shot_event.team_name.iloc[0]
    team2 = list(set(df_event.team_name.unique()) - {team1})[0]
    
    # subset the team shooting, and the opposition (goalkeeper/ other)
    df_team1 = df_freeze_frame[df_freeze_frame.team_name == team1]
    df_team2_goal = df_freeze_frame[(df_freeze_frame.team_name == team2) &
                                    (df_freeze_frame.position_name == 'Goalkeeper')]
    df_team2_other = df_freeze_frame[(df_freeze_frame.team_name == team2) &
                                     (df_freeze_frame.position_name != 'Goalkeeper')]
    
    
    # Setup the pitch
    pitch = VerticalPitch(half=True, goal_type='box', pad_bottom=-20)
    
    # We will use mplsoccer's grid function to plot a pitch with a title axis.
    fig, axs = pitch.grid(figheight=8, endnote_height=0,  # no endnote
                          title_height=0.1, title_space=0.02,
                          # Turn off the endnote/title axis. I usually do this after
                          # I am happy with the chart layout and text placement
                          axis=False,
                          grid_height=0.83)
    
    # Plot the players
    sc1 = pitch.scatter(df_team1.x, df_team1.y, s=600, c='#727cce', label='Attacker', ax=axs['pitch'])
    sc2 = pitch.scatter(df_team2_other.x, df_team2_other.y, s=600,
                        c='#5ba965', label='Defender', ax=axs['pitch'])
    sc4 = pitch.scatter(df_team2_goal.x, df_team2_goal.y, s=600,
                        ax=axs['pitch'], c='#c15ca5', label='Goalkeeper')
    
    # plot the shot
    sc3 = pitch.scatter(df_shot_event.x, df_shot_event.y, marker='football',
                        s=600, ax=axs['pitch'], label='Shooter', zorder=1.2)
    line = pitch.lines(df_shot_event.x, df_shot_event.y,
                       df_shot_event.end_x, df_shot_event.end_y, comet=True,
                       label='shot', color='#cb5a4c', ax=axs['pitch'])
    
    # plot the angle to the goal
    pitch.goal_angle(df_shot_event.x, df_shot_event.y, ax=axs['pitch'], alpha=0.2, zorder=1.1,
                     color='#cb5a4c', goal='right')
    
    # fontmanager for google font (robotto)
    robotto_regular = FontManager()
    
    # plot the jersey numbers
    jersey_num_dict={}
    for i, label in enumerate(df_freeze_frame.jersey_number):
        jersey_num_dict[label]=[df_freeze_frame.x[i], df_freeze_frame.y[i]]
        pitch.annotate(label, (df_freeze_frame.x[i], df_freeze_frame.y[i]),
                       va='center', ha='center', color='white',
                       fontproperties=robotto_regular.prop, fontsize=15, ax=axs['pitch'])
    
    # add a legend and title
    legend = axs['pitch'].legend(loc='center left', labelspacing=1.5)
    for text in legend.get_texts():
        text.set_fontproperties(robotto_regular.prop)
        text.set_fontsize(20)
        text.set_va('center')
    
    # title
    if sample.competition_id==55:
        competition="EURO 2020"
    elif sample.competition_id==43:
        competition="World Cup 2022"
    axs['title'].text(0.5, 0.5, f'Shooter: {df_shot_event.player_name.iloc[0]}\nMatch: {team1} vs. {team2} ({competition})',
                      va='center', ha='center', color='black',
                      fontproperties=robotto_regular.prop, fontsize=25)
    
    plt.savefig(output_path)
    return jersey_num_dict,df_lineup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--row_num','-r', type=int)
    parser.add_argument('--data_path','-d', type=str)
    parser.add_argument('--output_path','-o', type=str)
    args = parser.parse_args()

    # read the data
    df = pd.read_csv(args.data_path)
    # select the first row

    row_num=21
    sample = df.iloc[row_num]
    # plot the event
    jersey_num_dict,df_lineup=plot_event(sample,args.output_path+f"testing_plot_{row_num}.png")

    attacker_stats= eval(sample.attacker_dict.replace('nan', '-1'))
    for key,value in attacker_stats.items():
        x=sample[f"player{key}_location_x"]
        y=sample[f"player{key}_location_y"]
        #check with value in jersey_num_dict equals to x,y
        for key2,value2 in jersey_num_dict.items():
            if value2[0]==x and value2[1]==y:
                jersey_num=key2
                attacker_stats[key].append(jersey_num)

    #create a dataframe with the attacker stats #[offside,gk,xSOT,shotoff_prob,shot_block_prob,control_prob]
    df_attacker_stats=pd.DataFrame.from_dict(attacker_stats,orient='index',columns=['offside','gk','xSOT','shotoff_prob','shot_block_prob','control_prob','jersey_number'])
    #add a row
    shooter=[0,0,sample.xSOT,sample.shot_off_pred_prob,sample.shot_block_pred_prob,np.nan,"shooter"]
    #turn shooter in to df with the same column name as df_attacker_stats
    shooter_row=pd.DataFrame([shooter],columns=df_attacker_stats.columns)
    #append the shooter row to df_attacker_stats
    df_attacker_stats = pd.concat([df_attacker_stats, shooter_row], ignore_index=True)
    df_attacker_stats = df_attacker_stats[['xSOT','shotoff_prob','shot_block_prob','control_prob','jersey_number']]
    df_attacker_stats.to_csv(args.output_path+f"attacker_stats_{row_num}.csv")

        # pdb.set_trace()
