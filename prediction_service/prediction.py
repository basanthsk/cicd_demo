import yaml
import os
import json
import numpy as np
import joblib
import pandas as pd

params_path = "params.yaml"

schema_path = os.path.join("prediction_service", "scheama_in.json")


class NotInRange(Exception):
    def __init__(self, message="Values entered not in range"):
        self.message = message
        super().__init__(self.message)


class NotInCols(Exception):
    def __init__(self, message="Not in columns"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    scalar_path = config["webapp_scalar_dir"]

    scalar = joblib.load(scalar_path)
    model = joblib.load(model_dir_path)

    data_df = pd.DataFrame(data)
    scaled_data = scalar.transform(data_df)

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        # Returning the message for use on the same index.html page
        return 'You have chance of having diabetes.'
    else:
        return 'You are safe.'


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema


def form_response(dict_request):
    data = dict_request.values()
    data = [list(map(float, data))]
    response = predict(data)
    print(response)
    return response


def api_response(request):
    try:
        data = np.array([list(request.json.values())])
        response = predict(data)
        response = {"response": response}
        return response
    except Exception as e:
        print(e)
        error = {"error": "Something went wrong !! Try again"}
        return error
