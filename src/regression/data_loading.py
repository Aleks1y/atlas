import json
import requests
import pandas as pd
from properties import app_properties

model_columns = ["id", "input_Depth", "input_Por", "input_Per", "input_Sw0", "input_Den_r", "input_vis_oil",
                 "input_vis_wat", "input_Cw0", "input_Cw1", "input_delta", "input_Den_w", "input_Qx", "input_Qy",
                 "input_Ts", "input_zb", "input_kk_per_0", "input_kk_per_1", "input_n_wat", "input_n_oil",
                 "input_comp", "input_Sw1", "input_Per_cake", "input_Por_cake"]


def _load_models(filename):
    r = requests.post(app_properties.api_url + "/getModelList", json={"fields": model_columns, "filters": {}})
    data = r.json()
    with open(filename, 'w') as f:
        json.dump(data, f)


def _load_resises(res_filename, models_filename):
    with open(models_filename, 'r') as f:
        data = json.load(f)
    resises = []
    df = pd.json_normalize(data['list'])
    for cur_id in df["id"]:
        r = requests.post(app_properties.api_url + "/getResises", json={"modelId": cur_id})
        resises += r.json()["resises"]

    with open(res_filename, 'w') as f:
        json.dump(resises, f)


def _load_zonds(zond_filename, res_filename):
    with open(res_filename, 'r') as f:
        res_data = json.load(f)
    zonds = []
    res_df = pd.json_normalize(res_data)
    res_df = res_df[res_df["type"] == "ArchiFull"]

    for cur_id in res_df["id"]:
        r = requests.post(app_properties.api_url + "/getZondsInfo", json={"resisId": cur_id})
        for zondInfo in r.json()['list']:
            if zondInfo['name'] == 'VIKIZ1D':
                zondInfo['resId'] = cur_id
                zonds.append(zondInfo)

    with open(zond_filename, 'w') as f:
        json.dump(zonds, f)


def load_data_files():
    _load_models(app_properties.models_file)
    _load_resises(app_properties.res_file, app_properties.models_file)
    _load_zonds(app_properties.zond_file, app_properties.res_file)
