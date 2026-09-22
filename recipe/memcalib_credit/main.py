"""MemCalib training entry point on verl 0.8 main_ppo_sync."""

from __future__ import annotations

import os
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

try:
    import transfer_queue as tq
except ImportError:
    from verl.utils.transferqueue_utils import tq

from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.main_ppo_sync import PPOTrainer, run_ppo
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

from recipe.memcalib_credit.compat.checkpointing import (
    SharedFilesystemActorRolloutRefWorker,
    SharedFilesystemTrainingWorker,
)


class MemCalibSyncTaskRunner:
    """Recipe-local TaskRunner that injects only the custom trainer class."""

    def __init__(self) -> None:
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config) -> None:
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = (
            lora_rank > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )
        role = (
            Role.ActorRolloutRef
            if need_reference_policy(config) and not ref_in_actor
            else Role.ActorRollout
        )
        worker_cls = ActorRolloutRefWorker
        if config.actor_rollout_ref.actor.strategy == "megatron":
            worker_cls = SharedFilesystemActorRolloutRefWorker
        self.role_worker_mapping[role] = ray.remote(worker_cls)
        self.mapping[role] = "global_pool"

    def add_critic_worker(self, config) -> None:
        if need_critic(config):
            worker_cls = TrainingWorker
            if config.critic.strategy == "megatron":
                worker_cls = SharedFilesystemTrainingWorker
            self.role_worker_mapping[Role.Critic] = ray.remote(worker_cls)
            self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config) -> None:
        from verl.single_controller.ray import ResourcePoolManager

        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node]
            * config.trainer.nnodes,
        }
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError(
                    "config.reward.reward_model.n_gpus_per_node must be positive"
                )
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError(
                    "config.reward.reward_model.nnodes must be positive"
                )
            resource_pool_spec["reward_pool"] = [
                config.reward.reward_model.n_gpus_per_node
            ] * config.reward.reward_model.nnodes
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = (
                config.trainer.n_gpus_per_node
            )
            self.mapping[Role.RewardModel] = "global_pool"

        distillation = config.get("distillation")
        if is_distillation_enabled(distillation):
            if distillation.n_gpus_per_node <= 0 or distillation.nnodes <= 0:
                raise ValueError("distillation resource counts must be positive")
            resource_pool_spec["teacher_pool"] = [
                distillation.n_gpus_per_node
            ] * distillation.nnodes
            self.mapping[Role.TeacherModel] = "teacher_pool"

        self.resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )

    def run(self, config) -> None:
        from recipe.memcalib_credit.trainer import MemCalibSyncTrainer

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer: PPOTrainer | None = None
        try:
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.init_resource_pool_mgr(config)
            trainer = MemCalibSyncTrainer(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager,
                credit_config=OmegaConf.to_container(
                    config.get("credit", {}), resolve=True
                ),
            )
            trainer.init_workers()
            trainer.fit()
        finally:
            if trainer is not None:
                trainer.replay_buffer.close()
            tq.close()


@hydra.main(config_path="config", config_name="trainer", version_base=None)
def main(config) -> None:
    auto_set_device(config)
    config.transfer_queue.enable = True
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    api_key_env = str(config.credit.judge.api_key_env)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing Judge credential environment variable: {api_key_env}"
        )
    runner_class = ray.remote(
        num_cpus=1,
        runtime_env={"env_vars": {api_key_env: api_key}},
    )(MemCalibSyncTaskRunner)
    run_ppo(config, task_runner_class=runner_class)


if __name__ == "__main__":
    main()
