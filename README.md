# OpML19-MLflow
In this tutorial we'll walk through an example utilizing the three MLflow components (tracking, projects, models). The problem
statement is to predict the quality of wine given some features like "fixed acidity", "citric acid", etc...

## Setup
Install the required python modules using 
```
pip install -r requirements.txt
```
If you wish, you can use [virtualenv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/) to sandbox these requirements.

## MLflow Tracking
The first model we'll use is the [elastic net model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) from sklearn which is linear regression with some regularization included.

### Running the model
1. Let's look at `train.py`

```python
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    train_y = train[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
```

We use the `mlflow.*` APIs to log metadata about this training run to our tracking server. We also log the serialized model to our tracking server as an **artifact**. Artifacts are simply larger data outputs
from MLflow runs such as data, serialized models, images, etc and are usually stored in cloud storage. In our example, we've used the default configuration of the tracking server which logs the artifacts
to our local filesystem.

2. Run the model with various hyperparameters: `bash grid-search.sh`

### Exploring the UI
1. Launch the MLflow UI: `mlflow ui` and view it at http://localhost:5000
![](https://github.com/andrewmchen/OpML19-MLflow/raw/master/images/1.png)
2. Select all of the runs and click **Compare**
![](https://github.com/andrewmchen/OpML19-MLflow/raw/master/images/2.png)
3. We can filter only runs which have a non-zero `alpha` by putting `params.alpha != "0.0"` in the search box.
![](https://github.com/andrewmchen/OpML19-MLflow/raw/master/images/3.png)

## MLflow Projects
MLflow projects enable you to package the training code in a reusable format so that other data scientists (even outside your org) can easily reuse the model, or so that you can run the training remotely
for example on Databricks or kubernetes.

This is especially powerful with MLflow multistep workflows which are in currently in design/development. With MLflow multistep workflows, you can define a DAG of MLprojects which can re-use cached **artifact** results/be retriggered
in case of partial failure of the DAG. Two example use cases of this could be coordinating hyperparameter search in a distributed kubernetes cluster or defining a custom featurization step before using an off the shelf open source model.
![](https://github.com/andrewmchen/OpML19-MLflow/raw/master/images/5.png)


In this example we will use an implementation of gradient boosted tree regression implemented in the [mlflow-app](https://github.com/mlflow/mlflow-apps/tree/master) repository.

1. Look at the [MLproject definition](https://github.com/mlflow/mlflow-apps/tree/master/apps/gbt-regression)

```yaml
##################################
# MLproject
##################################
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      train: path
      test: path
      n-trees: {type: int, default: 100}
      m-depth: {type: int, default: 3}
      learning-rate: {type: float, default: .1}
      loss: {type: str, default: "rmse"}
      label-col: str
    command: "python main_gbt.py {train} {test} {n-trees} {m-depth} {learning-rate} {loss} {label-col}"

##################################
# conda.yaml
##################################
name: main
channels:
  - defaults
dependencies:
  - numpy
  - pandas=0.22.0
  - python==3.6
  - scikit-learn==0.19.1
  - pip:
    - mlflow
    - pyarrow
    - xgboost==0.71
```

Notice that the MLProject file simply defines the dependencies of project through conda 
and the entrypoint of the project. It also defines the parameters to the project
which must be configured.

In our example, we'll avoid downloading the conda dependencies (`--no-conda` flag) when running the project since we've
already downloaded them in `requirements.txt`.

2. Run the project with `mlflow run "https://github.com/mlflow/mlflow-apps/#apps/gbt-regression" --no-conda -Ptrain=wine-quality.parquet -Plabel-col=quality -Ptest=wine-quality.parquet`

## MLflow Models
The motivation of the Models component of MLflow is to simplify the N-N complexity between the number of ML algorithms and the number of possible deployment options. 
In our examples, we have already created boosted tree/elasticnet models which conform to the [pyfunc](https://mlflow.org/docs/latest/models.html#python-function-python-function) model interface. Simply speaking,
this interface is a serialized python function which has the following signature and a set of 
dependencies required to run the python function.
```
predict(model_input: pandas.DataFrame) -> [numpy.ndarray | pandas.Series | pandas.DataFrame]
```

We can see the model's filesystem layout by looking at the artifacts of any one of our MLflow runs in the UI.
```yaml
##################################
# MLmodel
##################################
artifact_path: model
flavors:
  python_function:
    data: model.pkl
    env: conda.yaml
    loader_module: mlflow.sklearn
    python_version: 2.7.16
  sklearn:
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 0.20.3
run_id: 451060d166ea47a99cf699a83a77cc57
utc_time_created: '2019-05-19 18:43:53.714752'

##################################
# conda.yaml
##################################
channels:
- defaults
dependencies:
- python=2.7.16
- scikit-learn=0.20.3
- pip:
  - cloudpickle==0.6.1
  name: mlflow-env

##################################
# model.pkl
##################################
```

1. Load the model in the Python interpreter
```python
from mlflow.pyfunc import load_pyfunc
model = load_pyfunc('model', run_id='451060d166ea47a99cf699a83a77cc57')
print(type(model))
# <class 'sklearn.linear_model.coordinate_descent.ElasticNet'>

print(model.predict.__doc__)
#  Predict using the linear model
# 
#         Parameters
#         ----------
#         X : array_like or sparse matrix, shape (n_samples, n_features)
#             Samples.
# 
#         Returns
#         -------
#         C : array, shape (n_samples,)
#             Returns predicted values.
```

2. We can also deploy the model as a HTTP server `mlflow pyfunc serve -m "model" --no-conda -p 6000 --run-id RUN_ID`
Usually this would be served within a conda environment containing the proper dependencies but since we already have them we use the `--no-conda` option.

3. Make an HTTP request to the server by doing `curl localhost:6000/invocations -H "Content-Type: application/json" -d "$(cat data.json)"`
