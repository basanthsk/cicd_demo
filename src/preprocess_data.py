''' read the data apply scaler
    save it in the processed folder for futher process
'''

from get_data import read_params, get_data, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib


def preprocess(config_path):
    print(config_path)
    config = read_params(config_path)
    scalar_path = config["webapp_scalar_dir"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    x_train_path = config["pre_process"]["x_train_path"]
    y_train_path = config["pre_process"]["y_train_path"]
    x_test_path = config["pre_process"]["x_test_path"]
    y_test_path = config["pre_process"]["y_test_path"]
    split_ratio = config["pre_process"]["test_size"]
    random_state = config["base"]["random_state"]
    label = config["base"]["label"]

    data = pd.read_csv(raw_data_path)

    scalar = StandardScaler()

    # drop the columns specified and separate the feature columns
    X = data.drop(columns=label)
    X = replace_zeros_mean(X)  # replace zeros with mean values
    Y = data[label]
    X_scaled = scalar.fit_transform(X)

    joblib.dump(scalar, scalar_path)

    x_train, x_test, y_train, y_test = split_data(
        X_scaled, Y, split_ratio, random_state)

    x_train.to_csv(x_train_path, sep=",", index=False, encoding="utf-8")
    y_train.to_csv(y_train_path, sep=",", index=False, encoding="utf-8")
    x_test.to_csv(x_test_path, sep=",", index=False, encoding="utf-8")
    y_test.to_csv(y_test_path, sep=",", index=False, encoding="utf-8")


def split_data(X, Y, split_ratio, random_state):
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=split_ratio, random_state=random_state)
    return pd.DataFrame(x_train), pd.DataFrame(x_test), pd.DataFrame(y_train), pd.DataFrame(y_test)


def replace_zeros_mean(x_data):
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    # zero_cols = [k for k, v in dict((x_data == 0).sum()).items() if v > 0]
    for col in zero_cols:
        x_data[col] = x_data[col].replace(0, x_data[col].mean())
    return x_data


if __name__ == "__main__":

    preprocess(config_path=get_config())
