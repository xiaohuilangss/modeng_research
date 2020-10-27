# encoding=utf-8
import tushare as ts


def cal_df_col_rank(df, col):
    """
    将df中的一列排名话
    :param df:
    :param col:
    :return:
    """
    l = len(df)
    df_origin = df.deepcopy()
    
    df = df.sort_values(by=col, ascending=True)
    df[col+'_rank_abs'] = list(range(l))
    df[col+'_rank'] = df.apply(lambda x: x[col+'_rank_abs']/l, axis=1)
    df = df.sort_index(ascending=True)
    df_origin[col+'_rank'] = df[col+'_rank']
    
    return df_origin


if __name__ == '__main__':
    
    df = ts.get_k_data('000001')
    r = cal_df_col_rank(df, 'close')