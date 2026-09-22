"""Qwen3.5-MoE checkpoint-schema compatibility for Megatron Bridge.

Transformers checkpoints exist with either grouped expert tensors
(``experts.gate_up_proj``/``experts.down_proj``) or one module per expert
(``experts.<id>.gate_proj``/``up_proj``/``down_proj``). Megatron Bridge's
Qwen3.5 mapping expects the grouped variant by default. This recipe-local
extension detects the checkpoint schema and changes only the two expert
mappings when a split checkpoint is loaded.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_SCHEMA_ENV = "QWEN35_MOE_EXPERT_SCHEMA"
_SUPPORTED_SCHEMAS = frozenset({"auto", "grouped", "split"})
_SPLIT_EXPERT_KEY = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
_GROUPED_EXPERT_KEY = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\."
    r"(?:gate_up_proj|down_proj)(?:\.weight)?$"
)
_INSTANCE_SCHEMA = "_memcalib_qwen35_moe_schema"
_REGISTERED = False
MemCalibQwen35VLMoEBridge: type | None = None


def detect_expert_schema(weight_keys: Iterable[str]) -> str:
    """Classify Qwen3.5-MoE expert keys, rejecting missing or mixed schemas."""

    has_split = False
    has_grouped = False
    for key in weight_keys:
        has_split = has_split or _SPLIT_EXPERT_KEY.match(key) is not None
        has_grouped = has_grouped or _GROUPED_EXPERT_KEY.match(key) is not None
        if has_split and has_grouped:
            raise ValueError("Qwen3.5-MoE checkpoint mixes grouped and split expert tensors")

    if has_split:
        return "split"
    if has_grouped:
        return "grouped"
    raise ValueError("Qwen3.5-MoE checkpoint contains no recognizable expert tensors")


def _configured_schema() -> str:
    schema = os.getenv(_SCHEMA_ENV, "auto").strip().lower()
    if schema not in _SUPPORTED_SCHEMAS:
        supported = ", ".join(sorted(_SUPPORTED_SCHEMAS))
        raise ValueError(f"Unsupported {_SCHEMA_ENV}={schema!r}; expected one of: {supported}")
    return schema


def checkpoint_weight_keys(checkpoint: Path) -> set[str]:
    """Read all tensor keys from a local safetensors checkpoint index."""

    indexes = sorted(checkpoint.glob("*.safetensors.index.json"))
    if not indexes:
        raise ValueError(f"No safetensors index found under Qwen3.5 checkpoint {checkpoint}")
    payload = json.loads(indexes[0].read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid weight_map in {indexes[0]}")
    return set(weight_map)


def detect_checkpoint_schema(checkpoint: Path) -> str:
    """Detect the grouped or split expert layout of a local checkpoint."""

    return detect_expert_schema(checkpoint_weight_keys(checkpoint))


def _pretrained_path(hf_pretrained: Any) -> Path | None:
    for owner in (hf_pretrained, getattr(hf_pretrained, "config", None)):
        if owner is None:
            continue
        for attr in ("model_name_or_path", "_model_name_or_path", "name_or_path"):
            value = getattr(owner, attr, None)
            if value:
                return Path(str(value))
    return None


def _weight_keys(hf_pretrained: Any) -> set[str]:
    state = getattr(hf_pretrained, "state", None)
    source = getattr(state, "source", None)
    get_all_keys = getattr(source, "get_all_keys", None)
    if callable(get_all_keys):
        return set(get_all_keys())

    checkpoint = _pretrained_path(hf_pretrained)
    if checkpoint is None or not checkpoint.is_dir():
        raise ValueError("Megatron Bridge did not expose a local Qwen3.5 checkpoint path")
    return checkpoint_weight_keys(checkpoint)


def _resolve_schema(hf_pretrained: Any) -> str:
    detected = detect_expert_schema(_weight_keys(hf_pretrained))
    configured = _configured_schema()
    if configured != "auto" and configured != detected:
        raise ValueError(
            f"{_SCHEMA_ENV}={configured!r} conflicts with detected checkpoint schema {detected!r}"
        )
    return detected


def _split_registry(registry: Any, registry_cls: type, gated_mapping_cls: type, auto_mapping_cls: type) -> Any:
    patched = []
    replaced: set[str] = set()
    for mapping in registry.mappings:
        megatron_param = getattr(mapping, "megatron_param", "")
        if megatron_param == "language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*":
            patched.append(
                gated_mapping_cls(
                    megatron_param=megatron_param,
                    gate="model.language_model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.experts.*.up_proj.weight",
                )
            )
            replaced.add("linear_fc1")
        elif megatron_param == "language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*":
            patched.append(
                auto_mapping_cls(
                    megatron_param=megatron_param,
                    hf_param="model.language_model.layers.*.mlp.experts.*.down_proj.weight",
                )
            )
            replaced.add("linear_fc2")
        else:
            patched.append(mapping)

    if replaced != {"linear_fc1", "linear_fc2"}:
        raise RuntimeError(
            "Megatron Bridge Qwen3.5 mappings changed upstream; expected both expert mappings, "
            f"replaced {sorted(replaced)}"
        )
    return registry_cls(*patched)


def _registry_handles_split(registry: Any) -> bool:
    for mapping in registry.mappings:
        hf_values = [
            getattr(mapping, name, "")
            for name in ("hf_param", "gate", "up")
        ]
        if any("experts.*.gate_proj.weight" in str(value) for value in hf_values):
            return True
    return False


def register_bridge_extension() -> bool:
    """Register a recipe-local subclass without modifying Bridge or verl code."""

    global _REGISTERED, MemCalibQwen35VLMoEBridge

    try:
        from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
        from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
        from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
        from megatron.bridge.models.qwen_vl import qwen35_vl_bridge as bridge_module
    except ImportError:
        logger.info("Megatron Bridge Qwen3.5 support is unavailable in this process")
        return False

    if _REGISTERED:
        return True

    base_bridge = bridge_module.Qwen35VLMoEBridge
    target = bridge_module.Qwen3VLModel
    provider = getattr(bridge_module, "Qwen35VLMoEModelProvider", None)

    class _MemCalibQwen35VLMoEBridge(base_bridge):
        """Qwen3.5 bridge subclass that accepts grouped and split experts."""

        def provider_bridge(self, hf_pretrained):
            schema = _resolve_schema(hf_pretrained)
            setattr(self, _INSTANCE_SCHEMA, schema)
            logger.info("Detected Qwen3.5-MoE checkpoint expert schema: %s", schema)
            return super().provider_bridge(hf_pretrained)

        def mapping_registry(self):
            registry = super().mapping_registry()
            schema = getattr(self, _INSTANCE_SCHEMA, None)
            if schema is None:
                hf_pretrained = getattr(self, "hf_pretrained", None)
                if hf_pretrained is not None:
                    schema = _resolve_schema(hf_pretrained)
                else:
                    configured = _configured_schema()
                    if configured == "auto":
                        raise RuntimeError(
                            "Qwen3.5-MoE expert schema was not detected before "
                            "mapping construction"
                        )
                    schema = configured
            if schema == "grouped" or _registry_handles_split(registry):
                return registry
            return _split_registry(
                registry,
                MegatronMappingRegistry,
                GatedMLPMapping,
                AutoMapping,
            )

    _MemCalibQwen35VLMoEBridge.__name__ = "MemCalibQwen35VLMoEBridge"
    registration = {
        "source": "Qwen3_5MoeForConditionalGeneration",
        "target": target,
        "model_type": "qwen3_5_moe",
    }
    if provider is not None:
        registration["provider"] = provider
    MemCalibQwen35VLMoEBridge = MegatronModelBridge.register_bridge(
        **registration
    )(_MemCalibQwen35VLMoEBridge)
    _REGISTERED = True
    logger.info("Registered recipe-local Qwen3.5-MoE Bridge subclass")
    return True


register_bridge_extension()
