import json
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Set, Sequence, Iterable, Optional
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer
from tqdm import tqdm
from PIL import Image


from torch.multiprocessing import set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
torch.cuda.empty_cache()

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_VIEW_KEYS = ("front_left", "front", "front_right")
DEFAULT_FUTURE_INDICES = (3, 7, 11, 15, 19)
VLLM_ALLOWED_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction (built on-the-fly from meta jsonl fields)
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_num(x: Any) -> str:
    value = float(x)
    if round(value, 2) == 0:
        value = 0.0
    return f"{value:.2f}"


def _fmt_xy(pt: Sequence[Any]) -> str:
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        raise ValueError(f"invalid point (need at least 2 dims): {pt!r}")
    return f"[{_fmt_num(pt[0])}, {_fmt_num(pt[1])}]"


def _fmt_traj(points: Iterable[Sequence[Any]]) -> str:
    return ", ".join(_fmt_xy(p) for p in points)


def _build_first_turn_prompt(
    intent: str,
    past_traj_text: str,
    past_accel_text: str,
    past_vel_text: str,
    current_vel_text: str,
    current_accel_text: str,
) -> str:
    """Aligned with convert_traj_jsonl_to_mllm_vision_mutil_turn.py (turn 1 user: <decision> only)."""
    return (
        "You are an expert driver.\n"
        "Input:\n"
        "- 1 frame of multi-view images collected from the ego-vehicle at the present timestep: "
        "front_left_view: <image>; front_view:<image>; front_right_view:<image>\n"
        f"- Current high-level intent:{intent}\n"
        f"- Current acceleration is {current_accel_text}\n"
        f"- Current velocity is {current_vel_text}\n"
        f"- 4-second past trajectory (16 steps at 4 Hz):{past_traj_text}\n"
        f"- 4-second past acceleration (16 steps at 4 Hz):{past_accel_text}\n"
        f"- 4-second past velocity (16 steps at 4 Hz):{past_vel_text}\n"
        "Coordinate System Definition: X-axis: positive forward, negative backward; Y-axis: positive left, negative right.\n"
        "Task: Inspect the input and make the decision.\n"
        "Output format:\n"
        "longitudinal action: xx, lateral action: xx"
    )


def _build_second_turn_user_prompt() -> str:
    """Aligned with convert_traj_jsonl_to_mllm_vision_mutil_turn.py (turn 2 user: <answer> only)."""
    return (
        "Task: Given the above information, predict the optimal 5-second future trajectory (5 steps at 1 Hz) of the ego vehicle.\n"
        "Output format:\n"
        "[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5]"
    )


def _first_turn_prompt_text_from_sample(sample: Dict[str, Any]) -> str:
    intent = str(sample.get("intent", ""))
    past_traj = sample.get("past_traj", [])
    past_accel = sample.get("past_accel", [])
    past_vel = sample.get("past_vel", [])
    current_vel = sample.get("current_vel", [0, 0])
    current_accel = sample.get("current_accel", [0, 0])
    return _build_first_turn_prompt(
        intent=intent,
        past_traj_text=_fmt_traj(past_traj),
        past_accel_text=_fmt_traj(past_accel),
        past_vel_text=_fmt_traj(past_vel),
        current_vel_text=_fmt_xy(current_vel),
        current_accel_text=_fmt_xy(current_accel),
    )


def _build_answer_only_ref(
    future_traj: Sequence[Sequence[Any]],
    indices: Sequence[int] = DEFAULT_FUTURE_INDICES,
) -> str:
    """Reference for turn 2: <answer> only (aligned with training label)."""
    pts = []
    for idx in indices:
        if idx < 0 or idx >= len(future_traj):
            raise IndexError(f"future_traj length {len(future_traj)} is too short for index {idx}")
        pts.append(_fmt_xy(future_traj[idx]))
    return ", ".join(pts)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_sample_ids_from_jsonl(jsonl_path: str) -> Set[str]:
    """Load sample_id set from a JSONL file (each line is JSON with a sample_id field)."""
    ids: Set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("sample_id", "")
            if sid is not None and str(sid).strip():
                ids.add(str(sid).strip())
    return ids


def load_scene_tokens(txt_path: str) -> Set[str]:
    """Load sceneToken set from a text file, one token per line."""
    tokens: Set[str] = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.add(token)
    return tokens


def scene_token_from_sample_id(sample_id: Any) -> str:
    """Extract sceneToken from sample_id formatted as sceneToken-frameID."""
    sid = str(sample_id).strip()
    if not sid:
        return ""
    return sid.rsplit("-", 1)[0]


def filter_samples_by_include_ids(
    samples: List[Dict[str, Any]], include_ids: Set[str]
) -> List[Dict[str, Any]]:
    """Keep only samples whose sample_id (as string) is in include_ids."""
    out: List[Dict[str, Any]] = []
    for s in samples:
        sid = s.get("sample_id", "")
        if str(sid).strip() in include_ids:
            out.append(s)
    return out


def filter_samples_by_scene_tokens(
    samples: List[Dict[str, Any]], scene_tokens: Set[str]
) -> List[Dict[str, Any]]:
    """Keep only samples whose sample_id sceneToken is in scene_tokens."""
    out: List[Dict[str, Any]] = []
    for s in samples:
        if scene_token_from_sample_id(s.get("sample_id", "")) in scene_tokens:
            out.append(s)
    return out


def filter_samples_by_exclude_ids(
    samples: List[Dict[str, Any]], exclude_ids: Set[str]
) -> List[Dict[str, Any]]:
    """Drop samples whose sample_id (as string) is in exclude_ids."""
    out: List[Dict[str, Any]] = []
    for s in samples:
        sid = str(s.get("sample_id", "")).strip()
        if sid not in exclude_ids:
            out.append(s)
    return out


def load_front_images(
    sample: Dict[str, Any],
    view_keys: Sequence[str] = DEFAULT_VIEW_KEYS,
) -> List[Any]:
    """Load images from meta sample's images dict in view order."""
    if "images" not in sample:
        raise ValueError(f"Sample missing 'images' field, available keys: {list(sample.keys())}")

    images_dict = sample["images"]
    if not isinstance(images_dict, dict):
        raise TypeError(f"images field must be a dict, got: {type(images_dict)}")

    images = []
    for key in view_keys:
        if key not in images_dict:
            raise ValueError(f"images dict missing key '{key}', available keys: {list(images_dict.keys())}")
        images.append(Image.open(images_dict[key]).convert("RGB"))
    return images


# ──────────────────────────────────────────────────────────────────────────────
# Conversation construction
# ──────────────────────────────────────────────────────────────────────────────

def build_user_content_from_text(prompt_text: str) -> List[Dict[str, Any]]:
    """Split a prompt string containing <image> markers into a vllm multimodal content list."""
    parts = prompt_text.split("<image>")
    content = []
    for i, text in enumerate(parts):
        if text:
            content.append({"type": "text", "text": text})
        if i < len(parts) - 1:
            content.append({"type": "image"})
    return content


def build_conversation_turn1_from_meta_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn 1: user only (multi-view <image> + decision task)."""
    prompt_text = _first_turn_prompt_text_from_sample(sample)
    user_content = build_user_content_from_text(prompt_text)
    return [{"role": "user", "content": user_content}]


def build_conversation_turn2_from_meta_sample(
    sample: Dict[str, Any], first_assistant_text: str
) -> List[Dict[str, Any]]:
    """Turn 2: user → assistant (turn1 <decision> gen) → user (trajectory <answer> task)."""
    first_prompt = _first_turn_prompt_text_from_sample(sample)
    user1_content = build_user_content_from_text(first_prompt)
    second_text = _build_second_turn_user_prompt()
    return [
        {"role": "user", "content": user1_content},
        {"role": "assistant", "content": first_assistant_text},
        {"role": "user", "content": [{"type": "text", "text": second_text}]},
    ]


def get_reference_turn2_answer(
    sample: Dict[str, Any],
    future_indices: Sequence[int] = DEFAULT_FUTURE_INDICES,
) -> str:
    """Reference for turn 2: GT <answer> from future_traj only."""
    future_traj = sample.get("future_traj", None)
    if not future_traj or not isinstance(future_traj, list) or len(future_traj) == 0:
        return ""
    try:
        return _build_answer_only_ref(future_traj=future_traj, indices=future_indices)
    except Exception:
        return ""


def get_reference_answer(
    sample: Dict[str, Any],
    future_indices: Sequence[int] = DEFAULT_FUTURE_INDICES,
) -> str:
    """Backward-compatible alias: same as get_reference_turn2_answer."""
    return get_reference_turn2_answer(sample, future_indices=future_indices)


# ──────────────────────────────────────────────────────────────────────────────
# VLM initialization
# ──────────────────────────────────────────────────────────────────────────────

def get_chat_template_tokenizer(processor: Any, model_path: str) -> Any:
    """Retrieve a usable tokenizer and ensure chat template is available."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        template_path = os.path.join(model_path, "chat_template.jinja")
        if os.path.isfile(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = f.read()
            template = tokenizer.chat_template

    if not template:
        raise ValueError(
            f"Model does not provide a chat template: {model_path}. "
            f"Please check tokenizer_config.json or chat_template.jinja."
        )
    return tokenizer


def apply_chat_template_with_tokenizer(
    conversation: List[Dict[str, Any]], tokenizer: Any
) -> str:
    """Render the chat template with the tokenizer and return the prompt string."""
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def setup_vllm(model_path: str, tensor_parallel_size: int = 1,
               gpu_memory_utilization: float = 0.9,
               max_model_len: int = 8192,
               max_pixels: int = 262144,
               min_pixels: int = 3136,
               lora_path: Optional[str] = None,
               max_lora_rank: int = 64,
               enable_tower_connector_lora: bool = False) -> LLM:
    """Initialize the vLLM model."""
    print(f"[INFO] Initializing vLLM, model: {model_path}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    if lora_path:
        print(f"[INFO] vLLM LoRA enabled, adapter: {lora_path}, max_lora_rank={max_lora_rank}")

    llm_kwargs = dict(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        # Disable prefix caching for multimodal inference: cached image KVs grow unboundedly and cause deadlock
        enable_prefix_caching=True,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel_size,
        limit_mm_per_prompt={"image": 4},
        max_model_len=max_model_len,
        mm_processor_kwargs={
            "max_pixels": max_pixels,
            "min_pixels": min_pixels,
        },
        trust_remote_code=True,
    )
    if lora_path:
        llm_kwargs.update(
            enable_lora=True,
            max_loras=1,
            max_lora_rank=max_lora_rank,
        )
        if tensor_parallel_size > 1:
            llm_kwargs["fully_sharded_loras"] = True
        if enable_tower_connector_lora:
            llm_kwargs["enable_tower_connector_lora"] = True

    llm = LLM(**llm_kwargs)
    return llm


def _load_adapter_config(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load PEFT adapter_config.json if path points to a LoRA checkpoint."""
    if not path:
        return None
    config_path = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_lora_checkpoint(path: Optional[str]) -> bool:
    """Return True when path looks like a PEFT LoRA adapter directory."""
    return bool(
        path
        and os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "adapter_config.json"))
        and (
            os.path.isfile(os.path.join(path, "adapter_model.safetensors"))
            or os.path.isfile(os.path.join(path, "adapter_model.bin"))
        )
    )


def _ceil_supported_lora_rank(rank: int) -> int:
    """vLLM only accepts fixed rank capacities; choose the smallest one >= rank."""
    for allowed_rank in VLLM_ALLOWED_LORA_RANKS:
        if rank <= allowed_rank:
            return allowed_rank
    raise ValueError(
        f"LoRA rank {rank} is larger than vLLM supported ranks: {VLLM_ALLOWED_LORA_RANKS}"
    )


def resolve_model_and_lora_paths(
    model_path: str,
    max_lora_rank: Optional[int],
    enable_tower_connector_lora: bool,
) -> tuple:
    """Resolve base model, adapter path and LoRA options from PEFT metadata.

    If --model_path points to a PEFT LoRA checkpoint, this function treats it as
    the adapter path and reads the base model path from adapter_config.json.
    """
    inferred_lora_path = None
    if _is_lora_checkpoint(model_path):
        inferred_lora_path = model_path
        print(
            "[INFO] --model_path points to a LoRA checkpoint; "
            "reading base model from adapter_config.json and loading the adapter online"
        )

    adapter_config = _load_adapter_config(inferred_lora_path)
    resolved_model_path = model_path

    if adapter_config:
        base_model = adapter_config.get("base_model_name_or_path")
        if inferred_lora_path == model_path and base_model:
            resolved_model_path = str(base_model)

        adapter_rank = adapter_config.get("r")
        if adapter_rank is not None:
            inferred_rank = _ceil_supported_lora_rank(int(adapter_rank))
            if max_lora_rank is None:
                max_lora_rank = inferred_rank
            elif int(max_lora_rank) < int(adapter_rank):
                print(
                    f"[WARN] configured max_lora_rank={max_lora_rank} is smaller than adapter rank r={adapter_rank}; "
                    f"using {inferred_rank} instead"
                )
                max_lora_rank = inferred_rank

        target_modules = str(adapter_config.get("target_modules", ""))
        if not enable_tower_connector_lora and "model.visual" in target_modules:
            enable_tower_connector_lora = True
            print(
                "[INFO] adapter_config target_modules contains model.visual; "
                "auto-enabling tower/connector LoRA support"
            )

    if inferred_lora_path and max_lora_rank is None:
        max_lora_rank = 64

    return (
        resolved_model_path,
        inferred_lora_path,
        max_lora_rank,
        enable_tower_connector_lora,
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON output parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_json_output(generation_text: str) -> Dict[str, Any]:
    """Parse JSON output from generated text."""
    text = generation_text.strip()

    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parsing failed: {e}")
        print(f"Raw text: {text[:200]}...")
        return {
            "error": "JSON parsing failed",
            "raw_text": text,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Resume support
# ──────────────────────────────────────────────────────────────────────────────

def processed_resume_key(sample_id: str, repeat_idx: int, repeat_num: int) -> str:
    """Stable key for resume dedup. When repeat_num<=1, only sample_id (backward compatible)."""
    if repeat_num <= 1:
        return sample_id
    return f"{sample_id}\t{repeat_idx}"


def load_processed_keys(output_jsonl: str, repeat_num: int = 1) -> Set[str]:
    """Load processed keys from output jsonl for resume (supports multi-repeat per sample)."""
    keys: Set[str] = set()
    if os.path.exists(output_jsonl):
        print(f"[INFO] Existing output file detected, loading processed keys: {output_jsonl}")
        try:
            with open(output_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        sample_id = data.get("sample_id", "")
                        if not sample_id:
                            continue
                        ridx = int(data.get("repeat_idx", 0))
                        keys.add(processed_resume_key(sample_id, ridx, repeat_num))
            print(f"[INFO] Loaded {len(keys)} processed keys")
        except Exception as e:
            print(f"[WARN] Failed to read output file, restarting from scratch: {e}")
            keys = set()
    return keys


def sample_fully_processed(
    sample_id: str, processed_keys: Set[str], repeat_num: int
) -> bool:
    if not sample_id:
        return False
    if repeat_num <= 1:
        return sample_id in processed_keys
    return all(
        processed_resume_key(sample_id, r, repeat_num) in processed_keys
        for r in range(repeat_num)
    )


def load_processed_sample_ids(output_jsonl: str) -> set:
    """Backward-compatible alias: same as load_processed_keys(..., repeat_num=1)."""
    return load_processed_keys(output_jsonl, repeat_num=1)


def load_all_processed_keys(output_jsonl: str, repeat_num: int = 1) -> Set[str]:
    """Load processed keys from the output jsonl (for resume bookkeeping)."""
    return load_processed_keys(output_jsonl, repeat_num=repeat_num)


def load_all_processed_sample_ids(output_jsonl: str) -> set:
    """Backward-compatible alias for repeat_num=1."""
    return load_all_processed_keys(output_jsonl, repeat_num=1)


# ──────────────────────────────────────────────────────────────────────────────
# Prefetch + inference core
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_turn1_sample_inputs(args):
    """Load images and build vLLM inputs for turn 1 (<decision>)."""
    sample, chat_tokenizer = args
    images = load_front_images(sample)
    conversation = build_conversation_turn1_from_meta_sample(sample)
    prompt = apply_chat_template_with_tokenizer(
        conversation=conversation, tokenizer=chat_tokenizer
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": images},
    }


def _prepare_turn2_sample_inputs(args):
    """Reload images and build vLLM inputs for turn 2 (<answer> trajectory), given turn-1 assistant text."""
    sample, first_generation_text, chat_tokenizer = args
    images = load_front_images(sample)
    conversation = build_conversation_turn2_from_meta_sample(sample, first_generation_text)
    prompt = apply_chat_template_with_tokenizer(
        conversation=conversation, tokenizer=chat_tokenizer
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": images},
    }


def _is_valid_meta_sample(s: Dict[str, Any]) -> bool:
    """Check whether a meta sample contains all required fields for inference."""
    images = s.get("images", None)
    if not isinstance(images, dict):
        return False
    if not all(k in images for k in DEFAULT_VIEW_KEYS):
        return False
    return True


def expand_vllm_batch_repeats(
    vllm_inputs: List[Dict[str, Any]],
    valid_samples: List[Dict[str, Any]],
    repeat_num: int,
    processed_keys: Set[str],
) -> tuple:
    """Each logical sample -> repeat_num vLLM rows (distinct seeds applied by caller), skip resumed repeats.

    Repeat rows share the same ``multi_modal_data["image"]`` list/refs (no PIL copy).
    Returns (expanded_inputs, meta) where meta is List[Tuple[sample_dict, repeat_idx]].
    """
    if repeat_num <= 1:
        return vllm_inputs, [(s, 0) for s in valid_samples]

    expanded: List[Dict[str, Any]] = []
    meta: List[tuple] = []
    for inp, s in zip(vllm_inputs, valid_samples):
        sid = s.get("sample_id", "")
        imgs = inp.get("multi_modal_data", {}).get("image", [])
        for r in range(repeat_num):
            if processed_resume_key(sid, r, repeat_num) in processed_keys:
                continue
            expanded.append(
                {
                    "prompt": inp["prompt"],
                    "multi_modal_data": {"image": imgs},
                }
            )
            meta.append((s, r))
    return expanded, meta


def _close_multimodal_images_once(vllm_inputs: List[Dict[str, Any]]) -> None:
    """Close each PIL image at most once (repeat rows may share the same image refs)."""
    seen: Set[int] = set()
    for inp in vllm_inputs:
        for img in inp.get("multi_modal_data", {}).get("image", []):
            iid = id(img)
            if iid in seen:
                continue
            seen.add(iid)
            img.close()


def _start_prefetch_loader(samples, batch_size, processed_keys, repeat_num,
                            chat_tokenizer, io_workers=16):
    """Start a background prefetch thread that loads the next batch while GPU runs inference.

    A persistent ThreadPoolExecutor (not recreated per batch) is held by the background
    thread, which puts prepared batches into a Queue(maxsize=2). The main thread calls
    q.get() to retrieve data; the next batch is usually already ready, minimizing GPU wait.

    Returns (queue, thread). Call thread.join() after the main thread finishes.
    """
    import threading
    from queue import Queue

    q = Queue(maxsize=2)

    def _worker():
        with ThreadPoolExecutor(max_workers=io_workers) as exe:
            for batch_idx in range(0, len(samples), batch_size):
                batch = samples[batch_idx:batch_idx + batch_size]
                pending, n_skipped = [], 0
                for s in batch:
                    sid = s.get("sample_id", "")
                    if sample_fully_processed(sid, processed_keys, repeat_num):
                        n_skipped += 1
                        continue
                    if not _is_valid_meta_sample(s):
                        continue
                    pending.append(s)

                if not pending:
                    q.put(([], [], n_skipped, batch_idx))
                    continue

                task_args = [(s, chat_tokenizer) for s in pending]
                futs = [exe.submit(_prepare_turn1_sample_inputs, a) for a in task_args]
                vllm_ins, valid_s = [], []
                for s, f in zip(pending, futs):
                    try:
                        vllm_ins.append(f.result())
                        valid_s.append(s)
                    except Exception as e:
                        print(f"[WARN] Sample preparation failed, skipping: {e}")

                q.put((vllm_ins, valid_s, n_skipped, batch_idx))
        q.put(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return q, t


def run_inference(
    input_jsonl: str,
    output_jsonl: str,
    model_path: str = "",
    batch_size: int = 8,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_samples: int = None,
    gpu_start_id: int = 0,
    repeat_num: int = 1,
    repeat_seed_base: int = 42,
    include_ids: Optional[Set[str]] = None,
    exclude_ids: Optional[Set[str]] = None,
    scene_tokens: Optional[Set[str]] = None,
) -> None:
    """Run two-turn inference (<decision> → <answer>), aligned with convert_traj_jsonl_to_mllm_vision_mutil_turn."""
    model_path, lora_path, max_lora_rank, enable_tower_connector_lora = resolve_model_and_lora_paths(
        model_path=model_path,
        max_lora_rank=None,
        enable_tower_connector_lora=False,
    )
    print(f"[INFO] Starting inference on GPU {gpu_start_id}-{gpu_start_id + tensor_parallel_size - 1}")
    if lora_path:
        print(
            "[INFO] Resolved LoRA inference paths: "
            f"base_model={model_path}, adapter={lora_path}, max_lora_rank={max_lora_rank}"
        )

    print(f"[INFO] Loading input data: {input_jsonl}")
    samples = load_jsonl(input_jsonl)

    if include_ids:
        n0 = len(samples)
        samples = filter_samples_by_include_ids(samples, include_ids)
        print(f"[INFO] include_jsonl filter: kept {len(samples)} / {n0} samples")

    if exclude_ids:
        n1 = len(samples)
        samples = filter_samples_by_exclude_ids(samples, exclude_ids)
        print(f"[INFO] exclude_jsonl filter: kept {len(samples)} / {n1} samples (removed {n1 - len(samples)})")

    if scene_tokens:
        n2 = len(samples)
        samples = filter_samples_by_scene_tokens(samples, scene_tokens)
        print(f"[INFO] sceneTokens filter: kept {len(samples)} / {n2} samples")

    if max_samples and max_samples > 0:
        samples = samples[:max_samples]
        print(f"[INFO] Limiting to {max_samples} samples")

    print(f"[INFO] Loaded {len(samples)} samples")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    chat_tokenizer = get_chat_template_tokenizer(processor, model_path)
    llm = setup_vllm(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        lora_path=lora_path,
        max_lora_rank=max_lora_rank,
        enable_tower_connector_lora=enable_tower_connector_lora,
    )
    lora_request = None
    if lora_path:
        lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_path=lora_path,
        )

    out_dir = os.path.dirname(output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    processed_keys = load_processed_keys(output_jsonl, repeat_num=repeat_num)
    if processed_keys:
        print(f"[INFO] Resume mode: will skip {len(processed_keys)} already-processed keys")

    file_mode = "a" if processed_keys and os.path.exists(output_jsonl) else "w"

    total_processed = 0
    total_skipped = 0
    total_batches = (len(samples) + batch_size - 1) // batch_size

    import time as _time
    prefetch_q, loader_thread = _start_prefetch_loader(
        samples, batch_size, processed_keys, repeat_num, chat_tokenizer,
    )

    with open(output_jsonl, file_mode, encoding="utf-8") as outfile:
        with tqdm(total=total_batches, desc="Inference progress") as pbar:
            while True:
                item = prefetch_q.get()
                if item is None:
                    break
                raw_inputs, valid_samples, n_skipped, batch_idx = item
                total_skipped += n_skipped
                pbar.update(1)

                if not raw_inputs:
                    print(f"[WARN] Batch {batch_idx // batch_size + 1} has no valid samples")
                    continue

                if repeat_num <= 1:
                    vllm_inputs = raw_inputs
                    valid_meta = [(s, 0) for s in valid_samples]
                else:
                    vllm_inputs, valid_meta = expand_vllm_batch_repeats(
                        raw_inputs, valid_samples, repeat_num, processed_keys
                    )

                if not vllm_inputs:
                    if repeat_num > 1:
                        _close_multimodal_images_once(raw_inputs)
                    print(f"[WARN] Batch {batch_idx // batch_size + 1} all repeats already done")
                    continue

                if repeat_num <= 1:
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    sampling_params = [
                        SamplingParams(
                            temperature=temperature,
                            max_tokens=max_tokens,
                            seed=repeat_seed_base + r,
                        )
                        for (_, r) in valid_meta
                    ]

                _t_infer_start = _time.time()
                outputs_turn1 = llm.generate(
                    vllm_inputs,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                    lora_request=lora_request,
                )

                turn2_task_args = [
                    (sample, outputs_turn1[i].outputs[0].text.strip(), chat_tokenizer)
                    for i, (sample, _) in enumerate(valid_meta)
                ]
                with ThreadPoolExecutor(max_workers=16) as _turn2_exe:
                    turn2_futs = [
                        _turn2_exe.submit(_prepare_turn2_sample_inputs, a)
                        for a in turn2_task_args
                    ]
                    vllm_inputs_turn2 = [f.result() for f in turn2_futs]

                outputs_turn2 = llm.generate(
                    vllm_inputs_turn2,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                    lora_request=lora_request,
                )
                _t_infer_end = _time.time()

                for (sample, repeat_idx), o1, o2 in zip(
                    valid_meta, outputs_turn1, outputs_turn2
                ):
                    generation_turn1 = o1.outputs[0].text.strip()
                    generation_turn2 = o2.outputs[0].text.strip()
                    combined = f"{generation_turn1}\n\n{generation_turn2}"
                    sample_id = sample.get("sample_id", "")
                    result = {
                        "index": total_processed,
                        "sample_id": sample_id,
                        "generation_turn1": generation_turn1,
                        "generation_turn2": generation_turn2,
                        "generation": combined,
                        "reference_turn1": "",
                        "reference_turn2": get_reference_turn2_answer(sample),
                        "reference": get_reference_turn2_answer(sample),
                    }
                    if repeat_num > 1:
                        result["repeat_idx"] = repeat_idx
                        result["seed"] = repeat_seed_base + repeat_idx
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                    outfile.flush()
                    if sample_id:
                        processed_keys.add(
                            processed_resume_key(sample_id, repeat_idx, repeat_num)
                        )
                    total_processed += 1

                print(f"[INFO] Batch {batch_idx // batch_size + 1}/{total_batches} done, "
                      f"processed {len(valid_meta)}, skipped {n_skipped} | "
                      f"inference_turn1+2={_t_infer_end - _t_infer_start:.1f}s")

                _close_multimodal_images_once(vllm_inputs)
                _close_multimodal_images_once(vllm_inputs_turn2)
                del outputs_turn1
                del outputs_turn2
                del vllm_inputs
                del vllm_inputs_turn2
                del valid_meta

    loader_thread.join()
    print(f"\n[INFO] Inference complete! Results saved to: {output_jsonl}")
    print(f"[INFO] Total processed: {total_processed} samples")
    if total_skipped > 0:
        print(f"[INFO] Total skipped (already processed): {total_skipped} samples")


def main():
    parser = argparse.ArgumentParser(
        description="VLM two-turn inference: turn1 <decision>, turn2 <answer> (aligned with convert_traj_jsonl_to_mllm_vision_mutil_turn)."
    )

    parser.add_argument(
        "--input_jsonl",
        type=str,
        # default='/nfs_zhaohang/zhengwc/waymo-e2e-dataset/val_samples_filtered_test_matched_5000_joint_random.jsonl',
        # default="/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/train_data/meta_data/val_samples_479.jsonl",
        # default='/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/train_data/meta_data/test_samples_last_frame.jsonl',
        # default="/nfs_zhaohang/zhengwc/waymo-e2e-dataset/test_samples.jsonl",
        default="/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/train_data/meta_data/val_samples_479_minus_434.jsonl",
        help="Input meta JSONL file path (each line contains images dict, intent, past_traj and other raw fields)",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default='/cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-2B/Mutil-Turn-w-mask-subsample/Second-Turn/v0-20260413-205852/checkpoint-950/val_samples_479_minus_434.jsonl',
        help="Output JSONL: generation_turn1/2, reference_turn1 (empty)/reference_turn2 (<answer> GT), reference=turn2",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='/cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-2B/Mutil-Turn-w-mask-subsample/Second-Turn/v0-20260413-205852/checkpoint-950',
        help="Base VLM model path or a PEFT LoRA checkpoint path. LoRA checkpoints are detected automatically.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (for testing; -1 means no limit)",
    )
    parser.add_argument(
        "--repeat_num",
        type=int,
        default=1,
        help="Per-sample repeat count: duplicate each sample as separate vLLM rows with seeds "
        "repeat_seed_base, repeat_seed_base+1, ... (no SamplingParams.n); model loads once",
    )
    parser.add_argument(
        "--repeat_seed_base",
        type=int,
        default=42,
        help="Base seed for repeat k is repeat_seed_base + k (only used when repeat_num > 1)",
    )
    parser.add_argument(
        "--include_jsonl",
        type=str,
        # default='/cephfs/zhengwc/FluidDrive/ms-swift-3.5/tools/infer_scripts/tmp.jsonl',
        default=None,
        help="Optional JSONL: each line is JSON with sample_id; only those samples are inferred",
    )
    parser.add_argument(
        "--exclude_jsonl",
        type=str,
        default=None,
        help="Optional JSONL: each line is JSON with sample_id; those samples are skipped",
    )
    parser.add_argument(
        "--sceneTokens_txt",
        type=str,
        # default='/cephfs/zhengwc/FluidDrive/ms-swift-3.5/infer_outputs/Qwen3.5-2B/Mutil-Turn-w-mask-subsample/RL-rfs-20ep/620-best-now/sceneTokens_for_vis.txt',
        default=None,
        help="Optional TXT: one sceneToken per line; only samples whose sample_id is sceneToken-frameID are inferred",
    )

    args = parser.parse_args()

    if args.repeat_num < 1:
        raise SystemExit("--repeat_num must be >= 1")

    include_ids: Optional[Set[str]] = None
    if args.include_jsonl:
        if not os.path.isfile(args.include_jsonl):
            raise SystemExit(f"--include_jsonl not found: {args.include_jsonl}")
        include_ids = load_sample_ids_from_jsonl(args.include_jsonl)
        print(f"[INFO] include_jsonl: loaded {len(include_ids)} sample_id(s) from {args.include_jsonl}")

    exclude_ids: Optional[Set[str]] = None
    if args.exclude_jsonl:
        if not os.path.isfile(args.exclude_jsonl):
            raise SystemExit(f"--exclude_jsonl not found: {args.exclude_jsonl}")
        exclude_ids = load_sample_ids_from_jsonl(args.exclude_jsonl)
        print(f"[INFO] exclude_jsonl: loaded {len(exclude_ids)} sample_id(s) from {args.exclude_jsonl}")

    scene_tokens: Optional[Set[str]] = None
    if args.sceneTokens_txt:
        if not os.path.isfile(args.sceneTokens_txt):
            raise SystemExit(f"--sceneTokens_txt not found: {args.sceneTokens_txt}")
        scene_tokens = load_scene_tokens(args.sceneTokens_txt)
        print(f"[INFO] sceneTokens_txt: loaded {len(scene_tokens)} sceneToken(s) from {args.sceneTokens_txt}")

    if args.repeat_num > 1:
        print(
            f"[INFO] repeat_num={args.repeat_num}: each sample runs as {args.repeat_num} "
            f"independent requests with seeds {args.repeat_seed_base}..{args.repeat_seed_base + args.repeat_num - 1}"
        )
    run_inference(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        model_path=args.model_path,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_samples=args.max_samples,
        gpu_start_id=0,
        repeat_num=args.repeat_num,
        repeat_seed_base=args.repeat_seed_base,
        include_ids=include_ids,
        exclude_ids=exclude_ids,
        scene_tokens=scene_tokens,
    )


if __name__ == "__main__":
    main()
