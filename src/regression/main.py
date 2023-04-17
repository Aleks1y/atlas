import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

import data_loading
import data_preparation
import polynomial_regression
import mini_batch_regressor
from properties import app_properties
from properties import rmq_properties

url = "http://emf.ru:8234"
corr1_columns = ['values.phi', 'input_Qy', 'values.q']
zond_column = ['values.rk~DF20', 'values.rk~DF16', 'values.rk~DF14',
               'values.rk~DF11', 'values.rk~DF10', 'values.rk~DF08',
               'values.rk~DF07', 'values.rk~DF06', 'values.rk~DF05']
res_columns = ['values.A', 'values.C0', 'values.S0', 'values.phi0', 'values.phi', 'values.q',
               'values.g']  # all values.p == 1.0
model_columns = ["id", "input_Depth", "input_Por", "input_Per", "input_Sw0", "input_Den_r", "input_vis_oil",
                 "input_vis_wat", "input_Cw0", "input_Cw1", "input_delta", "input_Den_w", "input_Qx", "input_Qy",
                 "input_Ts", "input_zb", "input_kk_per_0", "input_kk_per_1", "input_n_wat", "input_n_oil",
                 "input_comp", "input_Sw1", "input_Per_cake", "input_Por_cake"]


def main():
    channel = rmq_properties.rmq_channel
    channel.queue_declare(queue=rmq_properties.INPUT_QUEUE)
    channel.queue_declare(queue=rmq_properties.OUTPUT_QUEUE, durable=True)

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)
        body = json.loads(body)
        if body['action'] == 'load_data':
            print(' [x] Loading data files')
            data_loading.load_data_files()
        elif body['action'] == 'train_polynom_model':
            print(' [x] Training polynomial regression model')
            X, Y = data_preparation.prepare()

            # получаем полином
            polynomial_features = PolynomialFeatures(degree=3)
            X1 = pd.DataFrame(data=polynomial_features.fit_transform(X),
                              columns=polynomial_features.get_feature_names_out(X.columns))

            if 'params' in body and 'alpha' in body['params']:
                params = body['params']
                model = polynomial_regression.PolynomialRegression(alpha=params['alpha'])
            else:
                model = polynomial_regression.PolynomialRegression()
            # тренируем модель
            model.fit(X1, Y)

            coefs = pd.DataFrame(columns=zond_column, data=model.w)
            coefs.to_csv(app_properties.polynom_coefs_file)

            answer = {"id": body["id"], "params": X1.columns.to_list(), "coefs": model.w.tolist()}
            channel.basic_publish(exchange='', routing_key=rmq_properties.OUTPUT_QUEUE, body=json.dumps(answer))
        elif body['action'] == 'train_gradient_model':
            print(' [x] Training gradient regression model')
            params = body['params']
            X, Y = data_preparation.prepare()

            # получаем полином
            polynomial_features = PolynomialFeatures(degree=4)
            X1 = pd.DataFrame(data=polynomial_features.fit_transform(X),
                              columns=polynomial_features.get_feature_names_out(X.columns))

            model = mini_batch_regressor.MBGDRegressor(alpha=params['alpha'], eta0=params['eta0'])
            if 'init_w' in params:
                init_w = params['init_w']
            else:
                init_w = None
            # тренируем модель
            model.fit(X1, Y, init_w=init_w, batch_size=params['batch_size'], epoch=params['epoch'])
            answer = {"id": body["id"], "params": X1.columns.to_list(), "coefs": model.w.tolist()}
            channel.basic_publish(exchange='', routing_key=rmq_properties.OUTPUT_QUEUE, body=json.dumps(answer))
        elif body['action'] == 'predict':
            params = body['params']
            X = pd.DataFrame(params['predictors'])
            X.columns = params['columns']
            X = data_preparation.prepare_x(X)
            if params['type'] == 'polynom':
                # получаем полином
                polynomial_features = PolynomialFeatures(degree=3)
                X = pd.DataFrame(data=polynomial_features.fit_transform(X),
                                  columns=polynomial_features.get_feature_names_out(X.columns))
                coefs = pd.read_csv(app_properties.polynom_coefs_file).drop(columns='Unnamed: 0')
                values = pd.DataFrame(np.dot(X, coefs), columns=zond_column)
                answer = {"id": body["id"], "columns": values.columns.to_list(),
                          "data": values.values.tolist()}
                channel.basic_publish(exchange='', routing_key=rmq_properties.OUTPUT_QUEUE, body=json.dumps(answer))

            elif params['type'] == 'gradient':
                # получаем полином
                polynomial_features = PolynomialFeatures(degree=4)
                X = pd.DataFrame(data=polynomial_features.fit_transform(X),
                                 columns=polynomial_features.get_feature_names_out(X.columns))
                coefs = pd.read_csv(app_properties.grad_coefs_file).drop(columns='Unnamed: 0')
                values = pd.DataFrame(np.dot(X, coefs), columns=zond_column)
                answer = {"id": body["id"], "columns": values.columns.to_list(),
                          "data": values.to_json(orient="values")}
                channel.basic_publish(exchange='', routing_key=rmq_properties.OUTPUT_QUEUE, body=json.dumps(answer))

    channel.basic_consume(queue=rmq_properties.INPUT_QUEUE, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        exit(0)
