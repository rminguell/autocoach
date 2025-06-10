import os
import yaml
from pathlib import Path
from typing import Union, Optional
from paths import DATA_DIR, ENV_FPATH


def load_publication(publication_external_id="yzN0OCQT7hUS"):
    publication_fpath = Path(os.path.join(DATA_DIR, f"{publication_external_id}.md"))
    if not publication_fpath.exists():
        raise FileNotFoundError(f"Publication file not found: {publication_fpath}")
    try:
        with open(publication_fpath, "r", encoding="utf-8") as file:
            return file.read()
    except IOError as e:
        raise IOError(f"Error reading publication file: {e}") from e


def load_all_publications(publication_dir: str = DATA_DIR) -> list[str]:
    publications = []
    for pub_id in os.listdir(publication_dir):
        if pub_id.endswith(".md"):
            publications.append(load_publication(pub_id.replace(".md", "")))
    return publications


def load_yaml_config(file_path: Union[str, Path]) -> dict:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e


def load_env(api_key_type="GROQ_API_KEY") -> None:
    api_key = os.getenv(api_key_type)
    assert api_key, f"Environment variable '{api_key_type}' has not been loaded or is not set in the .env file."


def save_text_to_file(text: str, filepath: Union[str, Path], header: Optional[str] = None) -> None:
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            if header:
                f.write(f"# {header}\n")
                f.write("# " + "=" * 60 + "\n\n")
            f.write(text)
    except IOError as e:
        raise IOError(f"Error writing to file {filepath}: {e}") from e
