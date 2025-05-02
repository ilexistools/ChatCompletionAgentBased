import yaml
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import logging
from gpts.agents import GPTAgent

logger = logging.getLogger(__name__)
# By default we look for a "config" directory in the current working directory
DEFAULT_CONFIG_DIR = Path.cwd() / "config"

class GPTFactory:
    """Factory for creating GPTAgent instances from YAML specs."""

    def __init__(self, config_dir: Optional[Path | str] = None) -> None:
        """
        :param config_dir: directory containing your YAML files.
                           Defaults to "./config" in your current working directory.
        """
        # allow either a Path or a string
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR

        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"Config directory {self.config_dir!r} not found. "
                "Make sure it exists or pass the correct path to GPTFactory."
            )
        if not self.config_dir.is_dir():
            raise NotADirectoryError(f"{self.config_dir!r} exists but is not a directory.")

        self.config: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Read every .yaml file in `self.config_dir` into `self.config`."""
        for path in sorted(self.config_dir.glob("*.yaml")):
            try:
                key, spec = self._load_from_yaml(path)
            except Exception as e:
                logger.warning(f"Skipping {path.name}: {e}")
                continue

            if key in self.config:
                logger.warning(f"Overwriting existing spec for {key!r}")
            self.config[key] = spec

    def _load_from_yaml(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Load one YAML file and return (top_key, spec_dict).
        Expects exactly one top-level key whose value is a dict.
        """
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or len(data) != 1:
            raise ValueError(f"{path.name} must contain exactly one top-level key.")
        key, spec = next(iter(data.items()))
        if not isinstance(spec, dict):
            raise ValueError(f"Spec under {key!r} in {path.name} must be a mapping.")
        return key, spec

    def build(self, name: str, **overrides: Any) -> GPTAgent:
        """
        Instantiate a GPTAgent by merging the YAML spec with any overrides.
        
        :param name: the top-level key in one of the YAML files
        :param overrides: any GPTAgent __init__ args (role, goal, model, max_tokens, etc.)
        :return: configured GPTAgent
        """
        base = self.config.get(name)
        if base is None:
            raise KeyError(f"No GPT config named {name!r} in {self.config_dir}")

        # Merge YAML defaults with any user overrides
        params: Dict[str, Any] = {**base, **overrides}
        return GPTAgent(**params)
