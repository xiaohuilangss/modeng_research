# encoding=utf-8

"""
待删除
"""
import pandas as pd

if __name__ == '__main__':
    rank_result = pd.read_excel('C:/Users/Administrator/Desktop/排名.xlsx', sheet_name=0)
    score = pd.read_excel('C:/Users/Administrator/Desktop/分数.xls', sheet_name=0)

    score_no_nan = score.dropna().set_index(keys='姓名')
    rank_result = rank_result.set_index(keys='姓名')

    for idx in rank_result.index:
        try:
            if isinstance(score_no_nan.loc[idx, :], type(pd.DataFrame())):
                df = score_no_nan.loc[idx, :]

                rank_result.loc[idx, '获得资格时间'] = str(df.loc[idx, '获得资格时间'])

            else:
                rank_result.loc[idx, '获得资格时间'] = score_no_nan.loc[idx, '获得资格时间']
        except Exception as e_:
            print(str(e_))

    end = 0
