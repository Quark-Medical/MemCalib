"""Recipe-local compatibility helpers for text-only Ministral 3 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoConfig, PreTrainedTokenizerFast

from verl.utils.model import update_model_config
from verl.utils.tokenizer import set_pad_token_id
from verl.workers.config.model import HFModelConfig


MINISTRAL_MODEL_CONFIG_TARGET = (
    "recipe.memcalib_credit.model_compat.MinistralTextHFModelConfig"
)


def _require_ministral3_config(name_or_path: str, *, trust_remote_code: bool) -> None:
    config = AutoConfig.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
    )
    text_config = getattr(config, "text_config", None)
    model_type = getattr(config, "model_type", None)
    text_model_type = getattr(text_config, "model_type", None)
    if model_type != "mistral3" or text_model_type != "ministral3":
        raise ValueError(
            "MinistralTextHFModelConfig only supports a Mistral 3 checkpoint "
            f"with a Ministral 3 text backbone; got model_type={model_type!r}, "
            f"text_model_type={text_model_type!r} from {name_or_path!r}"
        )


def load_ministral_text_tokenizer(
    name_or_path: str,
    correct_pad_token: bool = True,
    **kwargs: Any,
):
    """Load the tokenizer.json backend whose decode matches Mistral/vLLM.

    Transformers 5 can auto-select ``LlamaTokenizer`` for this checkpoint. Its
    token IDs are correct after the regex fix, but its decoder exposes raw
    tokenizer markers such as ``Ġ`` and ``Ċ``. Loading the fast tokenizer
    base class uses tokenizer.json for both encoding and decoding.
    """

    tokenizer_kwargs = dict(kwargs)
    tokenizer_kwargs.pop("fix_mistral_regex", None)
    trust_remote_code = bool(tokenizer_kwargs.get("trust_remote_code", False))
    _require_ministral3_config(
        name_or_path,
        trust_remote_code=trust_remote_code,
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        name_or_path,
        **tokenizer_kwargs,
    )
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def uses_ministral_text_config(model_config: Any) -> bool:
    """Return whether a Hydra model config selected the recipe-local adapter."""

    if hasattr(model_config, "get"):
        target = model_config.get("_target_", "")
    else:
        target = getattr(model_config, "_target_", "")
    return str(target) == MINISTRAL_MODEL_CONFIG_TARGET


@dataclass
class MinistralTextHFModelConfig(HFModelConfig):
    """HFModelConfig variant that replaces only Ministral's tokenizer.

    The parent performs the normal verl model/config initialization. Afterward,
    this text-only adapter replaces the incorrectly auto-selected slow tokenizer
    with the tokenizer.json backend and removes the unused vision processor.
    """

    def __post_init__(self):
        super().__post_init__()

        self.tokenizer = load_ministral_text_tokenizer(
            self.local_tokenizer_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.processor = None

        if self.custom_chat_template is not None:
            self.tokenizer.chat_template = self.custom_chat_template

        update_model_config(
            self.hf_config,
            override_config_kwargs={
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            },
        )
