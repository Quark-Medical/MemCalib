"""Recipe-local checkpoint workers for eventually consistent shared mounts."""

from __future__ import annotations

import os
import time

import torch

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import set_expandable_segments
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker


def _ensure_directory(
    path: str,
    timeout: float = 60.0,
    poll_interval: float = 0.2,
) -> None:
    """Create ``path`` and wait until this process can observe it.

    OSSFS can transiently return ``ENOENT`` while a recursively created parent
    is propagating. Retrying only that transient case keeps permission and
    configuration errors visible.
    """

    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while True:
        if os.path.isdir(path):
            return
        try:
            os.makedirs(path, exist_ok=True)
        except (FileExistsError, FileNotFoundError) as exc:
            last_error = exc
        if os.path.isdir(path):
            return
        if time.monotonic() >= deadline:
            message = f"Directory is not visible after mkdir retries: {path}"
            if last_error is not None:
                raise FileNotFoundError(message) from last_error
            raise FileNotFoundError(message)
        time.sleep(poll_interval)


def _prepare_role_checkpoint(path: str) -> None:
    _ensure_directory(path)
    _ensure_directory(os.path.join(path, "dist_ckpt"))
    _ensure_directory(os.path.join(path, "huggingface"))


def _prepare_distributed_checkpoint(path: str) -> None:
    """Serialize directory creation, then verify visibility on every rank."""

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    rank_zero_error = None
    if rank == 0:
        try:
            _prepare_role_checkpoint(path)
        except Exception as exc:  # propagated to every rank below
            rank_zero_error = f"{type(exc).__name__}: {exc}"

    rank_zero_errors = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(rank_zero_errors, rank_zero_error)
    rank_zero_errors = [error for error in rank_zero_errors if error]
    if rank_zero_errors:
        raise RuntimeError(
            f"Failed to pre-create checkpoint directories: {rank_zero_errors}"
        )
    torch.distributed.barrier()

    visibility_error = None
    try:
        _prepare_role_checkpoint(path)
    except Exception as exc:  # propagated to every rank below
        visibility_error = f"rank {rank}: {type(exc).__name__}: {exc}"

    visibility_errors = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(visibility_errors, visibility_error)
    visibility_errors = [error for error in visibility_errors if error]
    if visibility_errors:
        raise RuntimeError(
            "Checkpoint directories are not visible on every rank: "
            f"{visibility_errors}"
        )
    torch.distributed.barrier()


class SharedFilesystemTrainingWorker(TrainingWorker):
    """Pre-create Megatron critic checkpoint paths on shared storage."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        global_step=0,
        max_ckpt_to_keep=None,
    ):
        _prepare_distributed_checkpoint(local_path)
        return super().save_checkpoint(
            local_path,
            hdfs_path,
            global_step,
            max_ckpt_to_keep,
        )


class SharedFilesystemActorRolloutRefWorker(ActorRolloutRefWorker):
    """Apply MemCalib's Megatron checkpoint and rollout-memory safeguards."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps=None, mode="auto"):
        effective_mode = (
            mode
            if mode != "auto"
            else self.config.rollout.checkpoint_engine.backend
        )
        if effective_mode == "naive":
            set_expandable_segments(False)
            aggressive_empty_cache(force_sync=True)
        return await super().update_weights(global_steps=global_steps, mode=mode)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        global_step=0,
        max_ckpt_to_keep=None,
    ):
        _prepare_distributed_checkpoint(local_path)
        return super().save_checkpoint(
            local_path,
            hdfs_path,
            global_step,
            max_ckpt_to_keep,
        )
