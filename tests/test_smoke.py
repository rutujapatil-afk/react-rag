from pathlib import Path

import react_rag

from react_rag.utils.config import load_config


def test_package_import():
    assert react_rag is not None


def test_config_loading():
    config_path = Path("configs/default.yaml")
    config = load_config(config_path)

    assert config["project"]["name"] == "REACT-RAG"
    assert config["retrieval"]["top_k"] == 10