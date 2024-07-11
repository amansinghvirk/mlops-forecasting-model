"""
module  : deploy_experiment.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - module contains the logic to deploy the selected model experiment

"""

import os
import json
import shutil
import argparse

from utils.utils import setup, get_proj_dir


class NotValidParamsFileException(Exception):
    "Raised when params in experiment parameters file is invalid"
    pass


def copy_files(src_dir: str, dest_dir: str) -> None:
    # fetch all files

    if os.path.exists(dest_dir) == False:
        os.mkdir(dest_dir)
    for file_name in os.listdir(src_dir):
        # construct full file path
        source = os.path.join(src_dir, file_name)
        destination = os.path.join(dest_dir, file_name)
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
        else:
            copy_files(source, destination)


def validate_deploy(src_dir: str, dest_dir: str) -> bool:
    """
    Validates if the files in the source directory are successfully deployed to the destination directory.

    Args:
        src_dir (str): The source directory containing the files to be deployed.
        dest_dir (str): The destination directory where the files should be deployed.

    Returns:
        bool: True if all files are successfully deployed, False otherwise.
    """
    if os.path.exists(dest_dir) == False:
        return False
    for file_name in os.listdir(src_dir):
        source = os.path.join(src_dir, file_name)
        destination = os.path.join(dest_dir, file_name)
        if os.path.isfile(source):
            if not os.path.exists(destination):
                return False
        else:
            if not validate_deploy(source, destination):
                return False
    return True
