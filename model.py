#-*- coding:utf-8 -*-


from __future__ import print_function, division

import pandas as pd
import numpy as np

import pickle
from glob import glob


# 想加上球员真实姓名吗？
def player_mapping(player_id, team_id, df):
        '''
        player_id: int
        df: pd.DataFrame or [df1, df2, df3], i.e. 'results/player_info.csv'
        '''
        if type(player_id) == str:
            raise TypeError('player_id must be int !!!')
        
        ## find player in one Player_List
        if type(df) == pd.core.frame.DataFrame:
            try:
                transfer = str(df.loc[df.player_id.astype(int) == player_id, 'transfer'].values[0])
                if transfer == 'Yes':
                    team = df.loc[df.player_id.astype(int) == player_id, 'new_team'].values[0]
                    player = df.loc[df.player_id.astype(int) == player_id, 'player_name'].values[0]
                    position = df.loc[df.player_id.astype(int) == player_id, 'position'].values[0]
                else:
                    team = df.loc[df.player_id.astype(int) == player_id, 'team_name'].values[0]
                    player = df.loc[df.player_id.astype(int) == player_id, 'player_name'].values[0]
                    position = df.loc[df.player_id.astype(int) == player_id, 'position'].values[0]
            except:
                print('Warning: Could not find player__', player_id)
                team, player, position = 'Unknown', 'Unknown', 'Unknown'
                #raise ValueError('Error: Could not find player__', player_id)
        
        ## find player in multiple Player_Lists
        elif type(df) == list:
            found = False
            for _df in df:
                try:
                    team = _df.loc[_df.player_id.astype(int) == player_id, 'team_name'].values[0]
                    player = _df.loc[_df.player_id.astype(int) == player_id, 'player_name'].values[0]
                    position = _df.loc[_df.player_id.astype(int) == player_id, 'position'].values[0]
                    found = True
                    break
                except:
                    continue
            if not found:
                print('Warning: Could not find player__', player_id)
                team, player, position = 'Unknown', 'Unknown', 'Unknown'
        return team, player, position

## 主函数
def main():
    # 读取已经保存的回归系数
    with open('params.pkl', 'r') as pfile:
        weights_attack, weights_defence = pickle.load(pfile)
        
    ## 进攻、防守参数
    att_features = ['任意球次数', '长传成功次数', '传中成功次数', '角球成功次数', \
                        '过人成功次数', '关键传球次数', '向前传球成功次数',\
                        '争高空球成功次数', '第二张黄牌次数', '直接红牌次数',\
                            ]

    def_features = ['抢断次数', '拦截次数', '解围次数', '封堵射门次数']


    # 读取球员基本信息，效力球队、国家、位置等
    info = pd.read_csv('EPL_16-17_player_info.csv')


    # 预设数据框
    Rating = pd.DataFrame({'player_id':info.player_id.tolist(), 'team_id':info.team_id.tolist(), \
                           'player_name':info.player_name.tolist(), 'team':info.team_name.tolist(), \
                           'position':info.position.tolist(), 'R1':[0.0]*len(info), \
                           'R2':[0.0]*len(info), 'R3':[0.0]*len(info), 'R4':[0.0]*len(info), 'R5':[0.0]*len(info), \
                           'R6':[0.0]*len(info), 'appearance':[0]*len(info)},
                          columns=['player_id', 'team_id', 'player_name', 'team', 'position', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'appearance'], \
                          index=info.player_id.tolist())



    ################# 一些准备 ##################
    ## For R2 ##
    # 统计每只队的总出场时间、总积分
    _l = []
    _k = []
    for _f in glob('player/*.csv'):
        df = pd.read_csv(_f, index_col=[0])[['球队', '球员', '位置', 'team_id', '出场次数', '出场时间', '积分']]
        _l.append(df)
        _k.append(df.groupby('team_id').mean())
    appear_df = pd.concat(_l)
    appear = appear_df.groupby('team_id').sum()['出场时间'].to_dict()

    points_df = pd.concat(_k)
    points_df['team_id'] = points_df.index
    points = points_df.groupby('team_id').sum()['积分'].to_dict()


    ## For R5 ##
    # 不失球得分的权重（按位置）
    clean_sheet_weights = {'Goalkeeper':0.51, 'Defender':0.33, 'Midfielder':0.05, 'Forward':0.03, \
                           'Striker':0.03, 'Unknown':0.00, '':0.00}

    clean_sheet_dict = {'':0}

    Teams = set()
    for _f in glob('player/*.csv'):
        df = pd.read_csv(_f, index_col=[0])
        score = df.groupby('team_id').sum()['进球数'].to_dict()
        (team1, score1), (team2, score2) = score.items()
        Teams.add(team1)
        Teams.add(team2)
        if int(score1) == 0:
            clean_sheet_dict[team2] = clean_sheet_dict.get(team2, 0) + 1
        if int(score2) == 0:
            clean_sheet_dict[team1] = clean_sheet_dict.get(team1, 0) + 1
    # 由于杯赛比赛少，有些球队没有出现不失球的场次
    for t in Teams:
        if not clean_sheet_dict.has_key(t):
            clean_sheet_dict[t] = 0


    ################ 循环每场比赛，统计球员评分 ################
    for _f in glob('player/*.csv'):
        print('match--------', _f.split('/')[1])
        # each match
        df = pd.read_csv(_f, index_col=[0])
        # each player in this match, calculate his rate
        for player_id in df.index:

            # if this player is not in the player_info file, then it will be missing when initialize the Rating
            if player_id not in Rating.index:
                Rating.loc[player_id, :] = [player_id]+['']*4+[0.0]*7
            
            # player's team_id in this match
            _team_id_thisInMatch = df.loc[player_id, 'team_id']
            
            #R1, the vector of stats
            sample = df.loc[player_id, att_features].values
            #shot effectiveness
            #shot_eff = df.ix[player_id, '进球效率']+1
            shot_eff = 0.07 #杯赛平均射门进球率
            #points
            _points = (df.loc[player_id, '积分'] + 3)**(1/3)
            #
            r1 = sample.dot(weights_attack)*shot_eff*_points
            #
            Rating.loc[player_id, 'appearance'] += df.loc[player_id, '出场时间']
            Rating.loc[player_id, 'R1'] += r1
            
            # R2, points/appearance
            #team = df.loc[player_id, 'team_id']
            team_appear = appear[_team_id_thisInMatch]
            team_points = points[_team_id_thisInMatch]
            Rating.loc[player_id, 'R2'] += team_points*df.loc[player_id, '出场时间']/team_appear
            
            # R3, goals
            _goals = df.loc[player_id, '进球数']
            #_shots = df.loc[player_id, '射门数']
            _shots_on = df.loc[player_id, '射正球门数']
            Rating.loc[player_id, 'R3'] += 1.039*_goals + 0.2*_shots_on #+ 0.2*_shots

            # R4, assists
            _assists = df.loc[player_id, '助攻次数']
            Rating.loc[player_id, 'R4'] += 1.039*_assists
            
            # R6， defensive actions
            _sample = df.loc[player_id, def_features].values
            #shot conceded effectiveness
            #shot_concede_eff = df.ix[player_id, '被进球效率']+1
            shot_concede_eff = 0.07
            #points
            _points = (df.loc[player_id, '积分'] + 3)**(1/3)
            r6 = _sample.dot(weights_defence)*shot_concede_eff*_points
            Rating.loc[player_id, 'R6'] += r6

            # R5, clean-sheets
            #_pos = Rating.loc[player_id, 'position']
            _pos = player_mapping(player_id, _team_id_thisInMatch, info)[2]
            #_team = Rating.loc[player_id, 'team_id']
            _w = clean_sheet_weights[_pos]
            #_cls = clean_sheet_dict[_team]
            #Rating.loc[player_id, 'R5'] = _w*_cls*2.784
            
            # clean sheet or not (in this match)
            _score = df.groupby('team_id').sum()['进球数'].to_dict()
            for _k,_g in _score.items():
                if _k != _team_id_thisInMatch:#opponent's goals
                    if _g == 0:
                        Rating.loc[player_id, 'R5'] += _w*1*2.784

    ## 合并6个单项得分，并调整尺度1~99分，保存结果 ##
    Rating['Total'] = 0.2*Rating['R1'] + \
                        0.4*Rating['R2'] + \
                        0.1*Rating['R3'] + \
                        0.05*Rating['R4'] + \
                        0.1*Rating['R5'] + \
                        0.15*Rating['R6']


    Rating['Total'] = np.sqrt(Rating['Total'].values)
    Rating['Total'] = (Rating['Total'] - Rating['Total'].min()) / (Rating['Total'].max() - Rating['Total'].min())
    Rating['Total'] = [round(k*99, 0) for k in Rating['Total']]

    Rating.to_csv('Rating.csv', index=False)


if __name__ == "__main__":
    main()




