from .fever import load_fever_tasks
from .hotpotqa import load_hotpotqa_tasks
from .wikipedia import WikipediaHit, WikipediaServiceClient, WikipediaServiceSettings
from .registry import load_dataset_tasks

__all__ = [
    "load_fever_tasks",
    "load_hotpotqa_tasks",
    "load_dataset_tasks",
    "WikipediaHit",
    "WikipediaServiceClient",
    "WikipediaServiceSettings",
]
