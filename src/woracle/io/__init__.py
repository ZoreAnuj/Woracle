from woracle.io.episodes import (
    list_rollouts,
    load_frames,
    load_rollout,
    save_episode,
)


def rollout_from_video(*args, **kwargs):
    """Lazy proxy — real implementation needs the [video] extra only at call time."""
    from woracle.io.video import rollout_from_video as _impl

    return _impl(*args, **kwargs)


__all__ = ["list_rollouts", "load_frames", "load_rollout", "rollout_from_video", "save_episode"]
