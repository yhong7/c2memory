from pathlib import Path

from dotenv import load_dotenv
import yaml


CONFIG_PATH = Path(__file__).with_name("configs.yaml")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

_ENV_LOADED = False


def load_project_env(*, override: bool = True) -> None:
    """Load project-local environment variables exactly once.

    The project `.env` is the local source of truth for API keys. We load it
    from the config layer instead of `src.__init__` so every configuration
    consumer gets the same deterministic environment.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(ENV_PATH, override=override)
    _ENV_LOADED = True


def load_config(path: str | None = None) -> dict:
    load_project_env()
    target = Path(path) if path else CONFIG_PATH
    with open(target, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
