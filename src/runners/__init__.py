REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_all import EpisodeRunner as EpisodeRunner_all
REGISTRY["episode_all"] = EpisodeRunner_all

from .episode_runner_bootstrap import EpisodeRunner as EpisodeRunner_bootstrap
REGISTRY["episode_bootstrap"] = EpisodeRunner_bootstrap
