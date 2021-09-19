from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import pickle
import numpy as np

params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.load(yaml_file)
    return config


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = pickle.load(open(model_dir_path, 'rb'))
    prediction = model.predict_proba(data)
    result = '{0:.{1}f}'.format(prediction[0][1], 2)
    return result
    # if result > str(0.5):
    #     # Returning the message for use on the same index.html page
    #     return f'You have chance of having diabetes.\nProbability of having Diabetes is {result}'
    # else:
    #     return 'You are safe.\n Probability of having diabetes is {result}'


def api_response(request):
    try:
        data=np.array([list(request.json.values())])
        response = predict(data)
        response = {"response": response}
        return response
    except Exception as e:
        print(e)
        error = {"error": "Something went wrong !! Try again"}
        return error 


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                data = dict(request.form).values()
                data = [list(map(float, data))]
                response = predict(data)
                print(response)
                return render_template("index.html", response=response)
            elif request.json:
                response = api_response(request)
                return jsonify(response)
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong !! Try again"}
            return render_template("404.html", error=error)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
