"""
Script for serving.
"""
import joblib
import json
from typing import AnyStr, BinaryIO, Dict, List, Optional

from environs import Env
from bdrk.model import BaseModel


env = Env()
OUTPUT_MODEL_NAME = env("OUTPUT_MODEL_PATH")


class Model(BaseModel):
    def __init__(self):
        self.feature_names, self.enc, self.pipe = (
            joblib.load(OUTPUT_MODEL_NAME)
        )

    def pre_process(self,
                    http_body: AnyStr,
                    files: Optional[Dict[str, BinaryIO]] = None
                    ) -> List[List[float]]:
        """
        Converts http_body to a list of one feature vector which will be
        passed to predict.

        :param http_body: The serialized http request body
        :type http_body: AnyStr
        :param files: A dictionary of field names to file handles. Ignored.
        :type files: dict[str, BinaryIO] | None
        :return: List of one feature vector
        :rtype: list[list[float]]
        """
        features = json.loads(http_body)
        return [[float(features[name]) for name in self.feature_names]]

    def predict(self, features: List[List[float]]) -> List[Dict[int, float]]:
        """
        Makes an inference.

        :param features: The list of one feature vector
        :type features: list[list[float]]
        :return: Return a list of one inference result
        :rtype: list[dict[int, float]]
        """
        return [{
            index: prob
            for index, prob in
            enumerate(self.pipe.predict_proba(features)[0, :])
        }]

    def post_process(self,
                     score: List[Dict[int, float]],
                     prediction_id: str) -> Dict[str, float]:
        """
        Returns post processed result (category to predicted
        probability) to client

        :param score: The inference result returned from predict
        :type score: list[dict[int, float]]
        :param prediction_id: Unique id for prediction
        :type prediction_id: str
        :return: Returns post processed result
        :rtype: dict[str, float]
        """
        return {
            self.enc.categories_[0][index]: prob
            for index, prob in score[0].items()
        }
