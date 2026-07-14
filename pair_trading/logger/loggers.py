import logging
import yaml
from pathlib import Path


ROOT_FILENAME = Path(__file__).parent / "configs"

TYPE_TO_FILENAME = {
    'basic': 'basic.yaml',
    'detailed': 'detailed.yaml',
}


def setup_logging(type_, /):
    with open(ROOT_FILENAME / TYPE_TO_FILENAME.get(type_, type_)) as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
