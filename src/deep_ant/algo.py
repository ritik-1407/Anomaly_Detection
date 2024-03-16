#!/usr/bin/env python
%pip install torch
import json, sys
import numpy as np
import torch
import numpy as np
import pandas as pd
from deepant.detector import Detector
from deepant.predictor import Predictor
from deepant.dataset import TimeSeries
from pathlib import Path
from sklearn.utils import shuffle
from helper import retrieve_save_path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


EPOCHS = 500
WINDOW = 45
PRED_WINDOW =  1
LR = 1e-5
WEIGHT_DECAY = 1e-6
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.25
BATCH_SIZE = 64
EARLY_STOPPING_DELTA = 0.05
EARLY_STOPPING_PATIENCE = 10
RANDOM_STATE = 42



class Config:
    dataInput: Path
    dataOutput: Path
    modelInput: Path
    modelOutput: Path
    executionType: str
    epochs: int
    window: int
    pred_window: int
    lr: float
    batch_size: int
    split: float
    early_stopping_delta: float
    early_stopping_patience: int
    random_state: int

    def __init__(self, params):
        self.dataInput = Path(params.get("dataInput", "data/dataset.csv"))
        # self.dataOutput = Path(params.get("dataOutput", "/Workspace/Shared/deep_ant/anomalies.csv"))
        self.dataOutput =Path(params.get("dataOutput", f"/Workspace/Shared/deep_ant/Models/anomalies_{params.get('dataset')}.csv"))
        # self.modelInput = Path(params.get("modelInput", "/Workspace/Shared/deep_ant/Models/model_{params.get("dataset")}.pt"))
        self.modelInput = Path(params.get("modelInput", f"/Workspace/Shared/deep_ant/Models/model_{params.get('dataset')}.pt"))
        # self.modelOutput = Path(params.get("modelOutput", "/Workspace/Shared/deep_ant/Models/model.pt"))
        self.modelOutput = Path(params.get("modelInput", f"/Workspace/Shared/deep_ant/Models/model_{params.get('dataset')}.pt"))
        self.executionType = params.get("executionType")
        self.dataset = params.get("dataset")
        try:
            customParameters = params["customParameters"]
        except KeyError:
            customParameters = {}
        self.epochs = customParameters.get("epochs", EPOCHS)
        self.window = customParameters.get("window_size", WINDOW)
        self.pred_window = customParameters.get("prediction_window_size", PRED_WINDOW)
        self.lr = customParameters.get("learning_rate", LR)
        self.batch_size = customParameters.get("batch_size", BATCH_SIZE)
        self.split = customParameters.get("split", TRAIN_SPLIT)
        self.early_stopping_delta = customParameters.get("early_stopping_delta", EARLY_STOPPING_DELTA)
        self.early_stopping_patience = customParameters.get("early_stopping_patience", EARLY_STOPPING_PATIENCE)
        self.random_state = customParameters.get("random_state", RANDOM_STATE)
    
    def __str__(self):
        if config.executionType == "train":
            outputString = f"Config("\
                f"dataInput={self.dataInput}, modelOutput={self.modelOutput}, executionType={self.executionType}," \
                f"epochs={self.epochs}, window={self.window}, lr={self.lr}," \
                f"pred_window={self.pred_window}, batch_size={self.batch_size})"
        elif config.executionType == "execute":
            outputString = f"Config("\
                f"dataInput={self.dataInput}, dataOutput={self.dataOutput}, modelInput={self.modelInput}," \
                f"executionType={self.executionType}, window={self.window}, pred_window={self.pred_window})"
        return outputString


def get_subsequences(data, window, pred_window, channels):
    X = []
    Y = []

    for i in range(len(data) - window - pred_window):
        X.append(data[i : i + window])
        Y.append(data[i + window : i + window + pred_window])

    X = np.array(X)
    Y = np.array(Y)
    X = np.moveaxis(X, source=2, destination=1)
    Y = np.reshape(Y, (Y.shape[0], channels*pred_window))
    return X, Y


def preprocess_data(config):
    """
    Requirements for dataset:
    - CSV dataset
    - 1. column is index (e.g. timestamp)
    - all other columns are values (float)
    - there must not be a specific label
    """

    if config.dataset == "synthetic-cbf":
        ts_data = spark.sql("select * from synthetic_datasets.cylinder_bell_funnel")
        ts_data = ts_data.toPandas()
        ts_data = ts_data.drop(columns='timestamp')
        
    elif config.dataset == "synthetic-sine":
        ts_data = spark.sql("select * from synthetic_datasets.sine_platform")
        ts_data = ts_data.toPandas()
        ts_data = ts_data.drop(columns='timestamp')
    elif config.dataset == "synthetic-ecg":
        ts_data = spark.sql("select * from synthetic_datasets.ecg_noise_10per1")
        ts_data = ts_data.toPandas()
        ts_data = ts_data.drop(columns='timestamp')

    elif config.dataset == "synthetic-sine-mean":
        ts_data = spark.sql("select * from synthetic_datasets.sine_mean")
        ts_data = ts_data.toPandas()
        ts_data = ts_data.drop(columns='timestamp')
    elif config.dataset == "swat"  and  config.executionType == "train":
        ts_data = spark.sql("select * from default.copy_of_swat_dataset_normal_v0_csv")
        ts_data = ts_data.toPandas()
        ts_data = ts_data.drop(columns='timestamp')
        ts_data.rename(columns={'Normal/Attack': 'is_anomaly'}, inplace=True)

# Replace values in the 'is_anomaly' column
        ts_data['is_anomaly'].replace({'Normal': 0, 'Attack': 1}, inplace=True)
        ts_data = ts_data.astype(float)
    elif config.dataset == "swat"  and  config.executionType == "execute":
        ts_data = spark.sql("select * from swat_preprocessed.swat_2015")
        ts_data = ts_data.toPandas()
        ts_data = ts_data.drop(columns='Timestamp')
        ts_data = ts_data.astype(float)

    print(f"Dataset {config.dataInput};")

    c_values = ts_data.columns
    channels = len(c_values)

    if config.executionType == "train":
        # define train and validation datasets
        train_samples = int(config.split * len(ts_data))
        valid_samples = int((1 - config.split) * len(ts_data))
        print(f"Training data: {train_samples} ({config.split*100:.0f}%)")
        print(f"Validation data: {valid_samples} ({(1 - config.split)*100:.0f}%)")
        print(ts_data.iloc[:train_samples].values)
        train_dataset = TimeSeries(ts_data.iloc[:train_samples].values, window_length=config.window, prediction_length=config.pred_window)
        valid_dataset = TimeSeries(ts_data.iloc[train_samples:].values, window_length=config.window, prediction_length=config.pred_window)

        return {
            "train": train_dataset,
            "val": valid_dataset,
            "n_channels": channels
        }
    elif config.executionType == "execute":
        test_data = ts_data.iloc[:]
        print(f"Creating subsequences with window length {config.window + config.pred_window}")
        test_dataset = TimeSeries(test_data.values, config.window, config.pred_window)

        return {
            "test": test_dataset,
            "n_channels": channels
        }

    return {}


def train(config):
    print("\nPREPROCESSING ====")
    data = preprocess_data(config)

    # create components
    predictor = Predictor(config.window, config.pred_window, config.lr, config.batch_size, in_channels=data["n_channels"])

    # train
    print("\nTRAINING =========")
    train_dataset = data["train"]
    valid_dataset = data["val"]
    predictor.train(train_dataset, valid_dataset, n_epochs=config.epochs, save_path=config.modelOutput,
                    early_stopping_delta=config.early_stopping_delta, early_stopping_patience=config.early_stopping_patience)


def execute(config):
    data = preprocess_data(config)

    print("\nPREDICTION =======")
    predictor = Predictor(window=config.window, pred_window=config.pred_window, in_channels=data["n_channels"])
    predictor.load(config.modelInput)
    print(predictor.model)

    detector = Detector()
    
    test_dataset = data["test"]
    predictedY = predictor.predict(test_dataset)
    print(predictedY)
    _, test_y = next(iter(DataLoader(test_dataset, batch_size=predictedY.shape[0])))
    anomaly_score = torch.sqrt(F.mse_loss(predictedY.detach(), test_y.detach(), reduction="none").sum(dim=[1, 2]))
    print(anomaly_score)

    anomalies = detector.detect(predictedY, test_dataset)
    print(config.dataOutput)

    result_save_path = retrieve_save_path(config.dataOutput, "anomalies.csv")
    print(result_save_path)
    anomalies.tofile(result_save_path, sep="\n")


    

def parse_args():
    print(len(sys.argv))
    if len(sys.argv) < 4:
        # print("No arguments supplied, please specify the execution type at least!", file=sys.stderr)

        json_string = '{"dataset":"swat","executionType": "execute", "dataInput": "", "customParameters": {"EPOCHS": 500, "WINDOW": 45,"PRED_WINDOW": 1,"LR" :1e-5 ,"WEIGHT_DECAY" :1e-6,"TRAIN_SPLIT" :0.8, "VAL_SPLIT" : 0.25,"BATCH_SIZE":64,"EARLY_STOPPING_DELTA":0.05, "EARLY_STOPPING_PATIENCE" : 10 ,"RANDOM_STATE":42}}'
        params =json.loads(json_string)
        return Config(params)


def set_random_state(config: Config) -> None:
    seed = config.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    config = parse_args()
    print(config)
    set_random_state(config)
    print(config)
    if config.executionType == "train":
        train(config)
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(f"No executionType '{config.executionType}' available! Choose either 'train' or 'execute'.")
