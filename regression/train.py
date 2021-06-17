import joblib
import logging
from typing import Tuple

from catboost import CatBoostRegressor
from environs import Env
import numpy as np
import pandas as pd
from sklearn import metrics

from bedrock_client.bedrock.analyzer import ModelTask, ModelTypes
from bedrock_client.bedrock.analyzer.model_analyzer import ModelAnalyzer
from bedrock_client.bedrock.api import BedrockApi
from bedrock_client.bedrock.metrics.collector import (
    BaselineMetricCollector,
    FeatureHistogramCollector,
    InferenceHistogramCollector
)
from bedrock_client.bedrock.metrics.encoder import MetricEncoder

env = Env()
OUTPUT_MODEL_PATH = env("OUTPUT_MODEL_PATH")
TRAIN_DATA_PATH = env("TRAIN_DATA_PATH")
TEST_DATA_PATH = env("TEST_DATA_PATH")

CONFIG_FAI = {
    "large_rings": {
        "privileged_attribute_values": [1],
        # privileged group name corresponding to values=[1]
        "privileged_group_name": "Large",
        "unprivileged_attribute_values": [0],
        # unprivileged group name corresponding to values=[0]
        "unprivileged_group_name": "Small",
    }
}


def load_dataset(filepath: str,
                 target: str) -> Tuple[pd.core.frame.DataFrame,
                                       np.ndarray]:
    """
    Loads the dataset and returns the features as a pandas dataframe and
    the target variable as a numpy array.

    :param filepath: Path to load the data
    :type filepath: str
    :param target: Target variable
    :type target: str
    :return: The features pandas dataframe and the target numpy array
    :rtype: tuple[pandas.core.frame.DataFrame, numpy.ndarray]
    """
    df = pd.read_csv(filepath).drop('Type', axis=1)  # Removes 'Type' column
    df['large_rings'] = (df['Rings'] > 10).astype(int)

    # Ensure nothing missing
    original_len = len(df)
    df.dropna(how="any", axis=0, inplace=True)
    num_rows_dropped = original_len - len(df)
    if num_rows_dropped > 0:
        print(f"Warning - dropped {num_rows_dropped} rows with NA data.")

    y = df[target].values
    df.drop(target, axis=1, inplace=True)

    return df, y


def train_catboost_model(X: pd.core.frame.DataFrame,
                         y: np.ndarray,
                         verbose: bool = False) -> CatBoostRegressor:
    """
    Scales the features and trains a logistic regression model.

    :param X: Features for training
    :type X: pandas.core.frame.DataFrame
    :param y: Target variable
    :type y: numpy.ndarray
    :param verbose: Whether to print additional info
    :type verbose: bool
    :return: Trained CatBoostRegressor model
    :rtype: catboost.CatBoostRegressor
    """
    verbose and print('\nTRAIN\nScaling...')
    model = CatBoostRegressor(iterations=100,
                              learning_rate=0.1,
                              depth=5)

    verbose and print('Fitting...')
    model.fit(X, y)

    verbose and print('Done training.')

    return model


def compute_log_metrics(model: CatBoostRegressor,
                        x_test: pd.core.frame.DataFrame,
                        y_test: np.ndarray):
    """
    Computes, prints and log metrics.

    :param model: Trained CatBoostRegressor model
    :type model: catboost.CatBoostRegressor
    :param x_test: Features for testing
    :type x_test: pandas.core.frame.DataFrame
    :param y_test: Target variable data for testing
    :type y_test: numpy.ndarray
    :return: Test predicted probability and predictions
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    y_pred = model.predict(x_test)

    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    r2_score = metrics.r2_score(y_test, y_pred)

    print("\nEVALUATION\n"
          f"\tMean absolute error          = {mae:.4f}\n"
          f"\tMean squared error           = {mse:.4f}\n"
          f"\tR2 regression score function = {r2_score:.4f}\n")

    # Bedrock Logger: captures model metrics
    bedrock = BedrockApi(logging.getLogger(__name__))

    bedrock.log_metric("MAE", mae)
    bedrock.log_metric("MSE", mse)
    bedrock.log_metric("R2", r2_score)

    return y_pred


def main():
    x_train, y_train = load_dataset(
        filepath=TRAIN_DATA_PATH,
        target='ShellWeight'
    )
    x_test, y_test = load_dataset(
        filepath=TEST_DATA_PATH,
        target='ShellWeight'
    )
    print('X (train)')
    print(x_train)

    model = train_catboost_model(x_train,
                                 y_train,
                                 verbose=True)
    # Save trained model
    feature_names = x_train.columns.tolist()
    print("\nSAMPLE FEATURES")
    print({
        feature_name: str(x_train[feature_name][0])
        for feature_name in feature_names
    })
    joblib.dump([feature_names, model], OUTPUT_MODEL_PATH)
    print('\nSaved feature names and catboost regression model.')

    y_pred = compute_log_metrics(model,
                                 x_test,
                                 y_test)

    # Save feature and inferance distribution
    train_predicted = model.predict(x_train).flatten().tolist()
    collectors = [
        FeatureHistogramCollector(
            data=x_train.iteritems(),
            discrete={6, 7},  # Specify which column indices are discrete
        ),
        InferenceHistogramCollector(data=train_predicted,
                                    is_discrete=False)
        # Specify inference as discrete
    ]
    encoder = MetricEncoder(collectors=collectors)
    with open(BaselineMetricCollector.DEFAULT_HISTOGRAM_PATH, "wb") as f:
        f.write(encoder.as_text())
    print('Saved feature and inference distribution.')

    # Train Shap model and calculate xafai metrics
    analyzer = (
        ModelAnalyzer(model,
                      model_name='catboost_model',
                      model_type=ModelTypes.TREE,
                      model_task=ModelTask.REGRESSION)
        .test_features(x_test)
    )

    (
        analyzer
        .fairness_config(CONFIG_FAI)
        .test_labels(y_test)
        .test_inference(y_pred)
    )

    analyzer.analyze()
    print('Saved Shap model and XAI for regression.')


if __name__ == '__main__':
    main()
