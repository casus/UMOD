import json
from pathlib import Path
from typing import Union

import numpy as np


def read_json_config(config_file_path: Union[str, Path]) -> dict[str, any]:
    """ Reads a json config file and evaluates strings if possible.

    Args:
        config_file_path (Union[str, Path]): A path to the json configuration file.

    Returns:
        dict[str, any]: A dictionary with configuration.
    """
    def eval_config(config: Union[dict, list, str, int, float, bool, None]):
        """ Recursively iterates throught the config to find values to evaluate.
        """
        if isinstance(config, dict):
            return {key: eval_config(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [eval_config(value) for value in config]
        elif isinstance(config, str):
            if config[0] == '$':
                return eval(config[1:])
        return config

    with open(config_file_path, 'r') as config_file:
      config = json.load(config_file)
    return eval_config(config)
