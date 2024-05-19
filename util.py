import logging
import os
import json


def setup_logger(name, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(console_handler)
    return logger


def build_dirs(path):
    if not os.path.exists(path):
        print('not exist')
        os.makedirs(path)
    print('exist')
    return


def save_json(payload, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(payload, outfile)
