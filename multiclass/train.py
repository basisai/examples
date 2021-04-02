import joblib
import logging
from typing import Tuple

from environs import Env
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bedrock_client.bedrock.analyzer import ModelTypes
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
C = env.float("C")

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
    df = pd.read_csv(filepath)
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


def train_log_reg_model(X: pd.core.frame.DataFrame,
                        y: np.ndarray,
                        seed: float = 0,
                        C: float = 1,
                        verbose: bool = False) -> Pipeline:
    """
    Scales the features and trains a logistic regression model.

    :param X: Features for training
    :type X: pandas.core.frame.DataFrame
    :param y: Target variable
    :type y: numpy.ndarray
    :param seed: `random_state` for logistic regression model
    :type seed: float
    :param C: Inverse of regularization strength
    :type C: float
    :param verbose: Whether to print additional info
    :type verbose: bool
    :return: Pipeline of transforms with a trained final estimator
    :rtype: sklearn.pipeline.Pipeline
    """
    verbose and print('\nTRAIN\nScaling...')
    scaling = StandardScaler()
    X = scaling.fit_transform(X)

    verbose and print('Fitting...')
    verbose and print('C:', C)
    model = LogisticRegression(random_state=seed, C=C, max_iter=4000)
    model.fit(X, y)

    verbose and print('Chaining pipeline...')
    pipe = Pipeline([('scaling', scaling), ('model', model)])

    verbose and print('Done training.')

    return pipe


def compute_log_metrics(pipe: Pipeline,
                        x_test: pd.core.frame.DataFrame,
                        y_test: np.ndarray,
                        y_test_onehot: np.ndarray):
    """
    Computes, prints and log metrics.

    :param pipe: Pipeline of transforms with a trained final estimator
    :type pipe: sklearn.pipeline.Pipeline
    :param x_test: Features for testing
    :type x_test: pandas.core.frame.DataFrame
    :param y_test: Target variable data for testing
    :type y_test: numpy.ndarray
    :param y_test_onehot: One hot encoded target variable data
    :type y_test_onehot: numpy.ndarray
    :return: Test predicted probability and predictions
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    test_prob = pipe.predict_proba(x_test)
    test_pred = pipe.predict(x_test)

    acc = metrics.accuracy_score(y_test, test_pred)
    precision = metrics.precision_score(y_test, test_pred, average='macro')
    recall = metrics.recall_score(y_test, test_pred, average='macro')
    f1_score = metrics.f1_score(y_test, test_pred, average='macro')
    roc_auc = metrics.roc_auc_score(y_test_onehot,
                                    test_prob,
                                    average='macro',
                                    multi_class='ovr')
    avg_prc = metrics.average_precision_score(y_test_onehot,
                                              test_prob,
                                              average='macro')
    print("\nEVALUATION\n"
          f"\tAccuracy                  = {acc:.4f}\n"
          f"\tPrecision (macro)         = {precision:.4f}\n"
          f"\tRecall (macro)            = {recall:.4f}\n"
          f"\tF1 score (macro)          = {f1_score:.4f}\n"
          f"\tROC AUC (macro)           = {roc_auc:.4f}\n"
          f"\tAverage precision (macro) = {avg_prc:.4f}")

    # Bedrock Logger: captures model metrics
    bedrock = BedrockApi(logging.getLogger(__name__))

    # `log_chart_data` assumes binary classification
    # For multiclass labels, we can use a "micro-average" by
    # quantifying score on all classes jointly
    # See https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html  # noqa: E501
    # This will allow us to use the same `log_chart_data` method
    bedrock.log_chart_data(
        y_test_onehot.ravel().astype(int).tolist(),  # list of int
        test_prob.ravel().astype(float).tolist()  # list of float
    )

    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("Precision (macro)", precision)
    bedrock.log_metric("Recall (macro)", recall)
    bedrock.log_metric("F1 Score (macro)", f1_score)
    bedrock.log_metric("ROC AUC (macro)", roc_auc)
    bedrock.log_metric("Avg precision (macro)", avg_prc)

    return test_prob, test_pred


def main():
    x_train, y_train = load_dataset(
        filepath=TRAIN_DATA_PATH,
        target='Type'
    )
    x_test, y_test = load_dataset(
        filepath=TEST_DATA_PATH,
        target='Type'
    )
    print('X (train)')
    print(x_train)

    # sklearn `roc_auc_score` and `average_precision_score` expects
    # binary label indicators with shape (n_samples, n_classes)
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    y_train_onehot = enc.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = enc.fit_transform(y_test.reshape(-1, 1))
    print('\nCATEGORIES')
    for value, category in enumerate(enc.categories_[0]):
        print(f'{category} : {value}')

    # Convert target variable to numeric values
    # ModelMonitoringService.export_text expect both features
    # and inference to be numeric values
    y_train = np.argmax(y_train_onehot, axis=1)
    y_test = np.argmax(y_test_onehot, axis=1)

    pipe = train_log_reg_model(x_train,
                               y_train,
                               seed=0,
                               C=C,
                               verbose=True)
    # Save trained model
    feature_names = x_train.columns.tolist()
    print("\nSAMPLE FEATURES")
    print({
        feature_name: str(x_train[feature_name][0])
        for feature_name in feature_names
    })
    joblib.dump([feature_names, enc, pipe], OUTPUT_MODEL_PATH)
    print('\nSaved trained one hot encoder and logistic regression model.')

    test_prob, test_pred = compute_log_metrics(pipe,
                                               x_test,
                                               y_test,
                                               y_test_onehot)

    # Save feature and inferance distribution
    train_predicted = pipe.predict(x_train).flatten().tolist()
    collectors = [
        FeatureHistogramCollector(
            data=x_train.iteritems(),
            discrete={7, 8},  # Specify which column indices are discrete
        ),
        InferenceHistogramCollector(data=train_predicted,
                                    is_discrete=True)
        # Specify inference as discrete
    ]
    encoder = MetricEncoder(collectors=collectors)
    with open(BaselineMetricCollector.DEFAULT_HISTOGRAM_PATH, "wb") as f:
        f.write(encoder.as_text())
    print('Saved feature and inference distribution.')

    # Train Shap model and calculate xafai metrics
    analyzer = (
        ModelAnalyzer(pipe[1],
                      model_name='logistic',
                      model_type=ModelTypes.LINEAR)
        .train_features(x_train)
        .test_features(x_test)
        .fairness_config(CONFIG_FAI)
        .test_labels(y_test)
        .test_inference(test_pred)
    )
    analyzer.analyze()
    print('Saved Shap model and fairness results.')


if __name__ == '__main__':
    main()
