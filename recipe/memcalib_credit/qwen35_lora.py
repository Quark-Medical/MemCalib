"""Preflight checks for Qwen3.5-35B-A3B Megatron LoRA with SGLang rollout."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any

from recipe.memcalib_credit.compat.qwen35_moe_bridge import (
    detect_checkpoint_schema,
    register_bridge_extension,
)


EXPECTED_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen3_5_moe"
EXPECTED_TEXT_MODEL_TYPE = "qwen3_5_moe_text"
EXPECTED_TRANSFER_QUEUE_VERSION = "0.1.7"
EXPECTED_SPACY_VERSION = "3.8.14"
EXPECTED_MARKDOWN_IT_VERSION = "4.0.0"
EXPECTED_SPACY_MODEL_VERSION = "3.8.0"


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def validate_model_checkpoint(model_path: Path) -> dict[str, Any]:
    """Validate the local Qwen3.5 config, tokenizer, and expert tensor layout."""

    from transformers import AutoConfig, AutoTokenizer

    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = raw.get("architectures") or []
    if EXPECTED_ARCHITECTURE not in architectures:
        raise RuntimeError(
            f"expected architecture {EXPECTED_ARCHITECTURE!r}, got {architectures!r}"
        )
    if raw.get("model_type") != EXPECTED_MODEL_TYPE:
        raise RuntimeError(
            f"expected model_type {EXPECTED_MODEL_TYPE!r}, got {raw.get('model_type')!r}"
        )

    text_config = raw.get("text_config")
    if not isinstance(text_config, dict):
        raise RuntimeError("Qwen3.5 config must contain text_config")
    if text_config.get("model_type") != EXPECTED_TEXT_MODEL_TYPE:
        raise RuntimeError(
            f"expected text model_type {EXPECTED_TEXT_MODEL_TYPE!r}, "
            f"got {text_config.get('model_type')!r}"
        )

    parsed = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if parsed.model_type != EXPECTED_MODEL_TYPE:
        raise RuntimeError(
            f"Transformers parsed model_type {parsed.model_type!r}, expected {EXPECTED_MODEL_TYPE!r}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not prompt:
        raise RuntimeError("Qwen3.5 chat template produced an empty prompt")

    return {
        "architecture": EXPECTED_ARCHITECTURE,
        "model_type": parsed.model_type,
        "tokenizer": type(tokenizer).__name__,
        "num_experts": int(text_config.get("num_experts", 0)),
        "expert_schema": detect_checkpoint_schema(model_path),
    }


def validate_parallel_layout(
    *,
    nnodes: int,
    gpus_per_node: int,
    actor_tp: int,
    actor_pp: int,
    actor_cp: int,
    actor_ep: int,
    actor_etp: int,
    rollout_tp: int,
    num_experts: int,
    ppo_mini_batch_size: int,
    ppo_micro_batch_size: int,
) -> dict[str, Any]:
    values = {
        "nnodes": nnodes,
        "gpus_per_node": gpus_per_node,
        "actor_tp": actor_tp,
        "actor_pp": actor_pp,
        "actor_cp": actor_cp,
        "actor_ep": actor_ep,
        "actor_etp": actor_etp,
        "rollout_tp": rollout_tp,
        "num_experts": num_experts,
        "ppo_mini_batch_size": ppo_mini_batch_size,
        "ppo_micro_batch_size": ppo_micro_batch_size,
    }
    for label, value in values.items():
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")

    world_size = nnodes * gpus_per_node
    dense_model_parallel = actor_tp * actor_pp * actor_cp
    if world_size % dense_model_parallel != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by TP*PP*CP={dense_model_parallel}"
        )
    expert_partition = actor_etp * actor_ep * actor_pp
    if world_size % expert_partition != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by ETP*EP*PP={expert_partition}"
        )
    if actor_tp % actor_etp != 0:
        raise ValueError(f"actor_tp={actor_tp} must be divisible by actor_etp={actor_etp}")
    if num_experts % actor_ep != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by actor_ep={actor_ep}")
    if world_size % rollout_tp != 0:
        raise ValueError(f"world_size={world_size} must be divisible by rollout_tp={rollout_tp}")

    data_parallel_size = world_size // dense_model_parallel
    if ppo_mini_batch_size % data_parallel_size != 0:
        raise ValueError(
            f"ppo_mini_batch_size={ppo_mini_batch_size} must be divisible by "
            f"attention data parallel size={data_parallel_size}"
        )
    local_ppo_mini_batch_size = ppo_mini_batch_size // data_parallel_size
    if local_ppo_mini_batch_size % ppo_micro_batch_size != 0:
        raise ValueError(
            f"local PPO mini-batch={local_ppo_mini_batch_size} must be divisible by "
            f"ppo_micro_batch_size={ppo_micro_batch_size}"
        )

    rollout_replicas = world_size // rollout_tp
    node_local = {
        "tensor_parallel": actor_tp <= gpus_per_node and gpus_per_node % actor_tp == 0,
        "expert_parallel": (
            actor_etp * actor_ep <= gpus_per_node
            and gpus_per_node % (actor_etp * actor_ep) == 0
        ),
        # With MCore's default tp-cp-ep-dp-pp order and contiguous ranks,
        # PP>1 spans nodes in this two-node layout.
        "pipeline_parallel": actor_pp == 1 or nnodes == 1,
        "rollout_tensor_parallel": (
            rollout_tp <= gpus_per_node and gpus_per_node % rollout_tp == 0
        ),
    }
    return {
        **values,
        "world_size": world_size,
        "data_parallel_size": data_parallel_size,
        "attention_data_parallel_size": data_parallel_size,
        "expert_data_parallel_size": world_size // expert_partition,
        "local_ppo_mini_batch_size": local_ppo_mini_batch_size,
        "ppo_gradient_accumulation_steps": (
            local_ppo_mini_batch_size // ppo_micro_batch_size
        ),
        "rollout_replicas": rollout_replicas,
        "rollout_replicas_per_node": (
            gpus_per_node // rollout_tp if node_local["rollout_tensor_parallel"] else 0
        ),
        "node_local_groups": node_local,
    }


def _require_source_markers(label: str, source: str, markers: tuple[str, ...]) -> None:
    missing = [marker for marker in markers if marker not in source]
    if missing:
        raise RuntimeError(f"{label} lacks required runtime markers: {missing}")


def validate_runtime(*, gpus_per_node: int, require_cuda: bool = True) -> dict[str, Any]:
    """Validate the known Megatron/SGLang merged-LoRA runtime contract."""

    import torch
    import megatron.bridge
    import megatron.core
    import sglang

    try:
        import transfer_queue
    except ImportError as exc:
        raise RuntimeError(
            "TransferQueue==0.1.7 is required by the synchronous MemCalib trainer"
        ) from exc
    try:
        import spacy
        from markdown_it import MarkdownIt
    except ImportError as exc:
        raise RuntimeError(
            "The Qwen3.5 job requires recipe/memcalib_credit/requirements.txt"
        ) from exc

    from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
    from verl.workers.engine.megatron.transformer_impl import MegatronEngine
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer

    if not register_bridge_extension():
        raise RuntimeError("Megatron Bridge does not expose Qwen35VLMoEBridge")
    importlib.import_module("sglang.srt.models.qwen3_5")

    transfer_queue_version = _distribution_version("TransferQueue")
    if transfer_queue_version != EXPECTED_TRANSFER_QUEUE_VERSION:
        raise RuntimeError(
            "TransferQueue runtime mismatch: expected "
            f"{EXPECTED_TRANSFER_QUEUE_VERSION}, got {transfer_queue_version}"
        )
    required_transfer_queue_api = (
        "KVBatchMeta",
        "close",
        "init",
        "kv_batch_get",
        "kv_batch_put",
    )
    missing_transfer_queue_api = [
        name for name in required_transfer_queue_api if not hasattr(transfer_queue, name)
    ]
    if missing_transfer_queue_api:
        raise RuntimeError(
            f"TransferQueue lacks required runtime APIs: {missing_transfer_queue_api}"
        )

    recipe_versions = {
        "spacy": _distribution_version("spacy"),
        "markdown_it_py": _distribution_version("markdown-it-py"),
        "en_core_web_sm": _distribution_version("en-core-web-sm"),
    }
    expected_recipe_versions = {
        "spacy": EXPECTED_SPACY_VERSION,
        "markdown_it_py": EXPECTED_MARKDOWN_IT_VERSION,
        "en_core_web_sm": EXPECTED_SPACY_MODEL_VERSION,
    }
    mismatched_recipe_versions = {
        name: {"expected": expected_recipe_versions[name], "actual": actual}
        for name, actual in recipe_versions.items()
        if actual != expected_recipe_versions[name]
    }
    if mismatched_recipe_versions:
        raise RuntimeError(
            "MemCalib segmentation runtime mismatch: "
            f"{mismatched_recipe_versions}; install "
            "recipe/memcalib_credit/requirements.txt"
        )
    try:
        spacy.load("en_core_web_sm", enable=["tok2vec", "parser"])
        MarkdownIt("commonmark").enable("table")
    except (OSError, ValueError) as exc:
        raise RuntimeError("MemCalib segmentation runtime failed to initialize") from exc

    engine_source = inspect.getsource(MegatronEngine.get_per_tensor_param)
    _require_source_markers(
        "Megatron merged-LoRA export",
        engine_source,
        ("self.model_config.lora.get", "export_hf_weights(self.module)"),
    )
    server_source = inspect.getsource(SGLangHttpServer.launch_server)
    _require_source_markers(
        "SGLang adapter gate",
        server_source,
        ("self.model_config.lora_rank > 0", "enable_lora"),
    )
    save_source = inspect.getsource(MegatronCheckpointManager.save_checkpoint)
    load_source = inspect.getsource(MegatronCheckpointManager.load_checkpoint)
    _require_source_markers(
        "Megatron LoRA checkpoint save",
        save_source,
        ("_maybe_filter_peft_state_dict", "save_hf_adapter"),
    )
    _require_source_markers(
        "Megatron LoRA checkpoint load",
        load_source,
        ("_maybe_filter_peft_state_dict", "strict=self.peft_cls is None"),
    )

    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if require_cuda and visible_gpus < gpus_per_node:
        raise RuntimeError(
            f"expected at least {gpus_per_node} visible GPUs, found {visible_gpus}"
        )

    gpu_properties = []
    for index in range(visible_gpus):
        properties = torch.cuda.get_device_properties(index)
        gpu_properties.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_gib": round(properties.total_memory / (1024**3), 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )

    return {
        "versions": {
            "torch": torch.__version__,
            "transformers": _distribution_version("transformers"),
            "sglang": _distribution_version("sglang"),
            "megatron_core": _distribution_version("megatron-core", "megatron_core"),
            "megatron_bridge": _distribution_version("megatron-bridge", "mbridge"),
            "flash_linear_attention": _distribution_version("flash-linear-attention"),
            "transfer_queue": transfer_queue_version,
            **recipe_versions,
        },
        "visible_gpus": visible_gpus,
        "gpus": gpu_properties,
        "rollout": "sglang",
        "training": "megatron",
        "lora_sync": "merged_full_weights",
        "checkpoint": "distributed adapter + optimizer + extra",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--actor-tp", type=int, required=True)
    parser.add_argument("--actor-pp", type=int, required=True)
    parser.add_argument("--actor-cp", type=int, required=True)
    parser.add_argument("--actor-ep", type=int, required=True)
    parser.add_argument("--actor-etp", type=int, required=True)
    parser.add_argument("--rollout-tp", type=int, required=True)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--ppo-mini-batch-size", type=int, required=True)
    parser.add_argument("--ppo-micro-batch-size", type=int, required=True)
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument("--skip-cuda-check", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    layout = validate_parallel_layout(
        nnodes=args.nnodes,
        gpus_per_node=args.gpus_per_node,
        actor_tp=args.actor_tp,
        actor_pp=args.actor_pp,
        actor_cp=args.actor_cp,
        actor_ep=args.actor_ep,
        actor_etp=args.actor_etp,
        rollout_tp=args.rollout_tp,
        num_experts=args.num_experts,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        ppo_micro_batch_size=args.ppo_micro_batch_size,
    )
    if args.layout_only:
        print(json.dumps({"parallel_layout": layout}, ensure_ascii=False, indent=2))
        return

    if args.model_path is None:
        parser.error("--model-path is required unless --layout-only is used")
    model_path = args.model_path.resolve()
    checkpoint = validate_model_checkpoint(model_path)
    if checkpoint["num_experts"] != args.num_experts:
        raise RuntimeError(
            f"checkpoint has {checkpoint['num_experts']} experts, expected {args.num_experts}"
        )
    runtime = validate_runtime(
        gpus_per_node=args.gpus_per_node,
        require_cuda=not args.skip_cuda_check,
    )
    print(
        json.dumps(
            {
                "status": "Qwen3.5 Megatron+SGLang LoRA preflight passed",
                "model_path": str(model_path),
                "checkpoint": checkpoint,
                "parallel_layout": layout,
                "runtime": runtime,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
