#!/usr/bin/env python
# coding: utf-8

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta, datetime

import mxnet
import pandas as pd
import requests
from gluonts.dataset import common
from gluonts.model import deepar
from gluonts.trainer import Trainer

def download_train_predict(*args):
    def download_code_data(code, start_date, end_date, data_path, code_data_length_limit=20000):
        code_download_path_urls = (
            "http://quotes.money.163.com/service/chddata.html?code=1{code}&start={start}&end={end}&fields={fields}",
            "http://quotes.money.163.com/service/chddata.html?code=0{code}&start={start}&end={end}&fields={fields}"
        )
        fields = 'TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
        for url_template in code_download_path_urls:
            u = url_template.format(code=code, start=start_date, end=end_date, fields=fields)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
                'Accept-Language': 'zh-CN,zh;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'Accept-Encoding': 'gzip, deflate'
            }
            text = requests.get(u, headers=headers).text
            if len(text) >= code_data_length_limit:
                with open(os.path.join(data_path, '{code}.csv'.format(code=code)), 'w', encoding='utf-8') as f:
                    f.write(text)
                return True
        return False

    def train_and_predict(code, start_date, end_date, data_path, predict_path):
        predict_days = 2
        csv = os.path.join(data_path, '{code}.csv'.format(code=code))
        df = pd.read_csv(csv)

        # skip training data lenght < 360
        if len(df) < 360:
            return False

        # set DT as index, TCLOSE as label and order by DT desc
        df.set_axis(
            ['DT', 'CODE', 'NAME', 'TCLOSE', 'HIGH', 'LOW', 'TOPEN', 'LCLOSE', 'CHG', 'PCHG', 'TURNOVER',
             'VOTURNOVER',
             'VATURNOVER', 'TCAP', 'MCAP'],
            axis='columns',
            inplace=True)
        df.drop(
            ['CODE', 'NAME', 'HIGH', 'LOW', 'TOPEN', 'LCLOSE', 'CHG', 'PCHG', 'TURNOVER', 'VOTURNOVER',
             'VATURNOVER',
             'TCAP', 'MCAP'], axis=1, inplace=True)
        df.set_index(['DT'], inplace=True)
        df = df.iloc[df.index.argsort()]

        # fill the lost DT and label (TCLOSE) with last available exchange day's value
        all_dt = [(datetime.strptime(df.index[0], "%Y-%m-%d") + timedelta(days=i)).__format__('%Y-%m-%d') for i in
                  range(1, (datetime.strptime(end_date, "%Y%m%d") - datetime.strptime(df.index[0], "%Y-%m-%d")).days)]
        miss_data = []
        value = df.TCLOSE[df.index[0]]
        for dt in all_dt:
            if dt in df.index:
                value = df.TCLOSE[dt]
            else:
                miss_data.append([dt, value])
        miss_df = pd.DataFrame(miss_data, columns=['DT', 'TCLOSE'])
        miss_df.set_index(['DT'], inplace=True)
        miss_df = miss_df.iloc[miss_df.index.argsort()]

        new_df = pd.concat([df, miss_df], axis=0)
        new_df = new_df.iloc[new_df.index.argsort()]
        new_df['timestamp'] = pd.to_datetime(new_df.index)
        new_df.set_index(['timestamp'], inplace=True)
        new_df = new_df.iloc[new_df.index.argsort()]
        train_data = new_df

        # build the training dataset for deepar
        data = common.ListDataset(
            [
                {'start': train_data.index[0], 'target': train_data.TCLOSE[:]}
            ], freq='1d')

        # now training the model
        if len(mxnet.test_utils.list_gpus()):
            estimator = deepar.DeepAREstimator(freq='1d', prediction_length=predict_days, trainer=Trainer(ctx='gpu', epochs=100))
        else:
            estimator = deepar.DeepAREstimator(freq='1d', prediction_length=predict_days, trainer=Trainer(epochs=100))
        predictor = estimator.train(training_data=data)

        # predict the future data
        predict = predictor.predict(data, 1)
        predict_list = list(predict)
        max, min, max_id, min_id = predict_list[0].samples.max(), predict_list[0].samples.min(), predict_list[0].samples.argmax(), predict_list[0].samples.argmin()
        predict_x = [(predict_list[0].start_date + timedelta(days=i)).__format__('%Y-%m-%d') for i in range(0, predict_days + 1)]
        predict_y = predict_list[0].samples[0]
        predict_df = pd.DataFrame(zip(pd.to_datetime(predict_x), predict_y), columns=['DT', 'TCLOSE'])
        predict_df['timestamp'] = predict_df['DT']
        predict_df.set_index('timestamp', inplace=True)
        train_df = train_data.loc[train_data.index[-5:]]
        train_df['DT'] = pd.to_datetime(train_df.index)
        output_df = pd.concat([train_df, predict_df], axis=0)
        if min_id < max_id and (max - min) / min >= 0.099:
            output_df.to_csv(os.path.join(predict_path, 'red_{code}.csv'.format(code=code)))
        else:
            output_df.to_csv(os.path.join(predict_path, 'green_{code}.csv'.format(code=code)))
        return True

    # function body
    code, start_date, end_date, data_path, predict_path = args[0]
    return download_code_data(code, start_date, end_date, data_path) and train_and_predict(code, start_date, end_date, data_path, predict_path)


if __name__ == "__main__":
    # pass the code from cmdline or get from web
    if len(sys.argv) > 1:
        all_code = sys.argv[1:]
        print(all_code)
    else:
        all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12&_=1579615221139"
        r = requests.get(all_code_url, timeout=5).json()
        all_code = [data['f12'] for data in r['data']['diff']]
        print(all_code)
    start_date = (date.today() - timedelta(days=720)).strftime("%Y%m%d")
    end_date = date.today().strftime("%Y%m%d")
    data_path = os.path.join(end_date, "data")
    os.makedirs(data_path, exist_ok=True)
    predict_path = os.path.join(end_date, "predict")
    os.makedirs(predict_path, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=20) as tpe:
        tpe.map(download_train_predict, [(code, start_date, end_date, data_path, predict_path) for code in all_code])
