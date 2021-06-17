"""
Script for serving.
"""
import joblib
import json
from typing import AnyStr, BinaryIO, Dict, List, Optional

from environs import Env
from bedrock_client.bedrock.model import BaseModel


env = Env()
OUTPUT_MODEL_NAME = env("OUTPUT_MODEL_PATH")


class Model(BaseModel):
    def __init__(self):
        self.feature_names, self.model = (
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

    def predict(self, features: List[List[float]]) -> List[float]:
        """
        Makes an inference.

        :param features: The list of one feature vector
        :type features: list[list[float]]
        :return: Return a float
        :rtype: float
        """
        return [self.model.predict(features)[0]]
