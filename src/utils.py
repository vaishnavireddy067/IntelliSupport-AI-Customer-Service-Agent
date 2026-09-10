"""Utility functions and common helpers for Hiver AI Support Agent."""

import os
import json
import logging
import yaml
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Safely load a YAML file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Save data structure to JSON with formatting."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    logger.info("Saved JSON output to %s", file_path)


def load_json(file_path: str) -> Any:
    """Safely load JSON from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logger(name: str = "hiver_agent") -> logging.Logger:
    """Configure standardized logger."""
    return logging.getLogger(name)
