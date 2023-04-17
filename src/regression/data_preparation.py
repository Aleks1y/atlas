import numpy as np
import json
import pandas as pd

from properties import app_properties

corr1_columns = ['values.phi', 'input_Qy', 'values.q']
zond_column = ['values.rk~DF20', 'values.rk~DF16', 'values.rk~DF14',
               'values.rk~DF11', 'values.rk~DF10', 'values.rk~DF08',
               'values.rk~DF07', 'values.rk~DF06', 'values.rk~DF05']
res_columns = ['values.A', 'values.C0', 'values.S0', 'values.phi0', 'values.phi', 'values.q',
               'values.g']  # all values.p == 1.0


def transform(df):
    df['input_Sw0'] = np.log(df['input_Sw0'])
    df['input_Per'] = np.log(df['input_Per']) + 35

    df['input_Per'] = np.log(df['input_Per'])
    df['input_kk_per_0'] = np.log(df['input_kk_per_0'])

    df['wat-oil'] = df['input_n_wat'] / df['input_n_oil']
    df = df.drop(columns=['input_n_wat', 'input_n_oil'])
    return df


def prepare():
    with open(app_properties.zond_file, 'r') as f:
        zond_data = json.load(f)
        zond_df = pd.json_normalize(zond_data)
        zond_df = zond_df[['resId', 'time'] + zond_column]

    with open(app_properties.res_file, 'r') as f:
        res_data = json.load(f)
        res_df = pd.json_normalize(res_data)
        res_df = res_df[res_df["type"] == "ArchiFull"]
        res_df = res_df[['id', 'modelId'] + res_columns]
        res_df = res_df[res_df['id'].isin(zond_df['resId'])]

    with open(app_properties.models_file, 'r') as f:
        models_data = json.load(f)
        models_df = pd.json_normalize(models_data["list"])
        models_df = models_df.drop(columns=['input_kk_per_1', 'input_comp'])
        models_df = models_df[models_df['id'].isin(res_df['modelId'])]
        models_df = models_df[models_df["id"] != 5403]

    # соединяем всё в одну таблицу
    df = models_df.merge(res_df, left_on='id', right_on='modelId', how='inner')
    df = df.merge(zond_df, left_on='id_y', right_on='resId', how='inner')

    # преобразуем входные данные
    df = transform(df)

    df = df[df['time'] != 0]
    df = df.reset_index().drop(columns=['index'])

    # нормируем
    for column in df.columns.drop(['id_x', 'modelId', 'id_y', 'resId', 'input_Por', 'input_Sw1', 'input_Por_cake',
                                   'input_Qx'] + zond_column):
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    # отделяем предикторы
    X = df.drop(columns=['id_x', 'modelId', 'id_y', 'resId'] + zond_column + corr1_columns)

    Y = df[zond_column]
    Y = np.log(Y)

    return X, Y


def prepare_x(X):
    with open(app_properties.zond_file, 'r') as f:
        zond_data = json.load(f)
        zond_df = pd.json_normalize(zond_data)
        zond_df = zond_df[['resId', 'time'] + zond_column]

    with open(app_properties.res_file, 'r') as f:
        res_data = json.load(f)
        res_df = pd.json_normalize(res_data)
        res_df = res_df[res_df["type"] == "ArchiFull"]
        res_df = res_df[['id', 'modelId'] + res_columns]
        res_df = res_df[res_df['id'].isin(zond_df['resId'])]

    with open(app_properties.models_file, 'r') as f:
        models_data = json.load(f)
        models_df = pd.json_normalize(models_data["list"])
        models_df = models_df.drop(columns=['input_kk_per_1', 'input_comp'])
        models_df = models_df[models_df['id'].isin(res_df['modelId'])]
        models_df = models_df[models_df["id"] != 5403]

    # соединяем всё в одну таблицу
    df = models_df.merge(res_df, left_on='id', right_on='modelId', how='inner')
    df = df.merge(zond_df, left_on='id_y', right_on='resId', how='inner')

    # преобразуем входные данные
    X = transform(X)
    X = X.drop(columns=corr1_columns)

    df = transform(df)
    df = df[df['time'] != 0]
    df = df.reset_index().drop(columns=['index'])

    # нормируем
    for column in df.columns.drop(['id_x', 'modelId', 'id_y', 'resId', 'input_Por', 'input_Sw1', 'input_Por_cake', 'input_Qx']
                                  + zond_column + corr1_columns):
        X[column] = (X[column] - df[column].min()) / (df[column].max() - df[column].min())

    return X
