#%%
import pandas as pd
from statsbombpy import sb
import argparse
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--output_path','-o', type=str)
args = parser.parse_args()


#competition=sb.competitions() #UEFA Euro 2020 competition id = 55, season=43
#world cup 2022 competition id = 43, season=106
#%% get euro 2020 match id
matches1=sb.matches(competition_id=55, season_id=43) 
matches2=sb.matches(competition_id=43, season_id=106) 
matches_id=matches1.match_id.tolist()+matches2.match_id.tolist()

match1_len=len(matches1)

#%%

#lineups=sb.lineups(match_id=3795220)

#%%
#events = sb.events(match_id=3795220)
#%%
#events = events[events["type"]=="Shot"]
#events = events.dropna(axis=1,how='all')

#%%get shot data
df=pd.DataFrame()
count=1
for i in matches_id:
    if count<=match1_len:
        competition_id=55
        season_id=43
    else:
        competition_id=43
        season_id=106
    # print(i,"/",count)
    events = sb.events(match_id=i)
    events = events[events["type"]=="Shot"]
    events["competition_id"]=competition_id
    events["season_id"]=season_id
    events = events.dropna(axis=1,how='all')
    df=pd.concat([df,events], ignore_index=True)
    count+=1
#%%
df[['location_x','location_y']] = pd.DataFrame(df.location.tolist(), index= df.index) #location from list to location_x and location_y

df=df.drop(['location'], axis=1) #remove location

df = df[df['period'] != 5] #drop all penlaty shootout

df= df.drop(["related_events"], axis=1) #drop related events

df = df[df['shot_type'] == "Open Play"]# drop penlaty and free kick
#%% shot outcome grouping
'''
Wayward (off T) = An unthreatening shot that was way
off target or did not have enough
power to reach the goal line (or a
miskick where the player didn’t make
contact with the ball)

Post (off T) = A shot that hit one of the three posts

Saved Off Target (off T) = A shot that was saved by the
goalkeeper but was not on target.

Saved to Post (removed) = If the keeper saves the shot and it
bounces off the goal frame

Goal (On T) = A shot that was deemed to cross the
goal-line by officials

Saved (On T) = A shot that was saved by the opposing
team’s keeper
'''
df=df[df['shot_outcome'] != "Saved to Post "]
df=df.reset_index(drop=True)
df["shot_outcome_grouped"]=""
for i in range(len(df)):
    temp=df.loc[i,"shot_outcome"]
    if temp in ["Off T","Wayward","Post","Saved Off Target"]:
        out="Off T"
    elif temp in ["Blocked"]:
        out="Blocked"
    elif temp in ["Saved","Goal"]:
        out="On T"        
    df.loc[i,"shot_outcome_grouped"]=out

#%%

df[['shot_end_location_x','shot_end_location_y']] = pd.DataFrame(df.shot_end_location.tolist(), index= df.index).iloc[:,[0,1]]
df=df.drop(['shot_end_location'], axis=1) #remove shot_end_location


#%%
for i in range(22):
    df[f"player{i}_location_x"]=""
    df[f"player{i}_location_y"]=""
    df[f"player{i}_player_id"]=""
    df[f"player{i}_player_name"]=""
    df[f"player{i}_position_id"]=""
    df[f"player{i}_position_name"]=""
    df[f"player{i}_teammate"]=""

#%%
for j in range(len(df)):
    freezeframe_data=df.shot_freeze_frame[j]
    num_player=len(freezeframe_data)
    for i in range(num_player):
        df.loc[j,f"player{i}_location_x"]=freezeframe_data[i]["location"][0]
        df.loc[j,f"player{i}_location_y"]=freezeframe_data[i]["location"][1]
        df.loc[j,f"player{i}_player_id"]=freezeframe_data[i]["player"]["id"]
        df.loc[j,f"player{i}_player_name"]=freezeframe_data[i]["player"]["name"]
        df.loc[j,f"player{i}_position_id"]=freezeframe_data[i]["position"]["id"]
        df.loc[j,f"player{i}_position_name"]=freezeframe_data[i]["position"]["name"]
        df.loc[j,f"player{i}_teammate"]=freezeframe_data[i]["teammate"]

#%%
df=df.drop(['shot_freeze_frame'], axis=1) #drop shot freeze frame

#%%%

df.to_csv(args.output_path+"dataset.csv", index=False, encoding='utf-8')

#%%
df1=pd.read_csv("dataset.csv")
import numpy as np
#%%
#number of match 115, 51 from Euro 2020 and 64 from world cup 2022

#create feature distance to goal and angle to goal
##goal location (120,40)
##football pitch size 105 by 68 metres, x scale 105/120, y scale 68/80



df1["Dist2Goal"]= (((df1.location_x-120)*105/120)**2+((df1.location_y-40)*68/80)**2)**0.5
df1["Ang2Goal"]= np.abs(np.arctan2((40-df1.location_y)*68/80,(120-df1.location_x)*105/120))
df1.to_csv(args.output_path+"dataset.csv", index=False, encoding='utf-8')
#%% plot graph for player position and feasible angle to goal

def plot_event(row_of_data):
    
    sample=row_of_data
    import matplotlib.pyplot as plt
    
    from mplsoccer import Pitch, VerticalPitch, FontManager, Sbopen
    
    
    
    plt.style.use('ggplot')
    
    # get event and lineup dataframes for game 7478
    # event data
    parser = Sbopen()
    df_event, df_related, df_freeze, df_tactics = parser.event(sample.match_id)
    
    # lineup data
    df_lineup = parser.lineup(sample.match_id)
    df_lineup = df_lineup[['player_id', 'jersey_number', 'team_name']].copy()
    
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
    for i, label in enumerate(df_freeze_frame.jersey_number):
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
    
    plt.show()  # If you are using a Jupyter notebook you do not need this line

# plot_event(df1.iloc[0])
# plot_event(df1.iloc[1000])
# plot_event(df1.iloc[2000])
# plot_event(df1.iloc[2500])
