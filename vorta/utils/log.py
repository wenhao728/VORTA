import argparse
import json
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(
    output_file: Optional[str] = None,
    log_level: str = 'INFO',
    log_format: str = "%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    date_format: str = '%Y-%m-%d %H:%M:%S',
) -> None:
    """setup logging"""
    handlers = [logging.StreamHandler(stream=sys.stdout)]

    if output_file is not None:
        handlers.append(logging.FileHandler(output_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )


class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, Enum):
            return o.value
        return super().default(o)


def arg_to_json(arg: argparse.Namespace) -> str:
    return json.dumps(vars(arg), cls=JsonEncoder, indent=4, sort_keys=True)


def arg_to_yaml(arg: argparse.Namespace) -> str:
    return yaml.dump(vars(arg), indent=4, sort_keys=True)