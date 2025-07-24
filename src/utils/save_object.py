"""Utility module for saving Python objects using dill."""

import os
import sys
import dill

from src.utils.exception import CustomException


def save_object(file_path, obj):
    """
    Save a Python object to a specified file path using dill.

    Args:
        file_path (str): Path where the object should be saved.
        obj (Any): The Python object to serialize and save.

    Raises:
        CustomException: If any exception occurs during saving.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e
