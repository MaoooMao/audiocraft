# === Generate with Multi-Scale Adapters ===
from pathlib import Path
import re
import torch
from torch import nn

from audiocraft.models import MusicGen
from audiocraft.utils.checkpoint import load_checkpoint
from audiocraft.data.audio import audio_write

# Import the adapter class
from audiocraft.modules.transformer import ContentAwareMultiScaleAdapter
from audiocraft.models.lm import LMModel

# === Paths ===
ckpt_path = Path("/content/drive/MyDrive/audiocraft_checkpoints/text2music_3ad_epoch_75.th")
out_dir = Path("/content/drive/MyDrive/audiocraft_output")
out_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# === Temporarily allow non-strict LMModel loading ===
if not getattr(LMModel, "_patched_non_strict_load", False):
    _orig_lm_load = LMModel.load_state_dict

    def _lm_load_non_strict(self, state_dict, strict=True, assign=False):
        return _orig_lm_load(self, state_dict, strict=False, assign=assign)

    LMModel.load_state_dict = _lm_load_non_strict
    LMModel._patched_non_strict_load = True

# === Load the pretrained base model (MusicGen) ===
print("Loading base MusicGen model...")
model = MusicGen.get_pretrained("facebook/musicgen-medium")

# === Restore LMModel loading behavior ===
if getattr(LMModel, "_patched_non_strict_load", False):
    LMModel.load_state_dict = _orig_lm_load
    delattr(LMModel, "_patched_non_strict_load")

# === Set submodules to evaluation mode ===
model.lm.eval()
model.compression_model.eval()
lm = model.lm

# === Load your finetuned checkpoint ===
print(f"Loading checkpoint from {ckpt_path}")
state = load_checkpoint(str(ckpt_path))
sd = (
    state.get("model_best_state")
    or state.get("fsdp_best_state")
    or state.get("best_state")
    or state
)
if "model" in sd:
    sd = sd["model"]

# === AUTO-DETECT adapter configuration from checkpoint ===
# This will detect both old and new adapter configurations

# Adapter configuration mapping
# Format: {prompt_len: (stride, gate_init)}
ADAPTER_CONFIGS = {
    # NEW configuration (current code)
    256: (2, -2.2),    # short-range
    64: (8, -1.5),     # mid-range
    16: (32, -3.0),    # long-range

    # OLD configuration (if checkpoint was trained with old code)
    # Uncomment if your checkpoint uses the old configuration:
    # 512: (8, -2.2),
    # 64: (96, -1.5),
    # 16: (256, -3.0),
}

def get_stride_and_gate(prompt_len):
    """Get stride and gate_init based on prompt_len"""
    if prompt_len in ADAPTER_CONFIGS:
        return ADAPTER_CONFIGS[prompt_len]

    # Fallback: try to infer from prompt_len
    print(f"WARNING: Unknown prompt_len={prompt_len}, using fallback mapping")
    if prompt_len >= 256:
        return (2, -2.2)  # Assume new config
    elif prompt_len >= 64:
        return (8, -1.5)
    else:
        return (32, -3.0)

def build_adapters_for_attn(attn, adapter_specs):
    """
    Build adapters for an attention module

    Args:
        attn: The attention module
        adapter_specs: List of (prompt_len, stride, gate_init) tuples
    """
    if not adapter_specs:
        return

    attn.use_adapter = True
    attn.adapters = nn.ModuleList([
        ContentAwareMultiScaleAdapter(
            embed_dim=attn.embed_dim,
            num_heads=attn.num_heads,
            stride=stride,
            prompt_len=prompt_len,
            gate_init=gate_init,
            device="cpu",
            dtype=torch.float32
        )
        for prompt_len, stride, gate_init in adapter_specs
    ])

    # Initialize adapter_weights
    # Default values from training: [-0.2, 0.6, -0.4]
    if not hasattr(attn, "adapter_weights"):
        if len(adapter_specs) == 3:
            init_weights = torch.tensor([-0.2, 0.6, -0.4])
        else:
            init_weights = torch.zeros(len(adapter_specs))
        attn.adapter_weights = nn.Parameter(init_weights)

# === Detect adapters from checkpoint and attach to model ===
pat_tpl = r"^transformer\.layers\.(\d+)\.(self_attn|cross_attention)\.adapters\.(\d+)\.prompt$"
pat = re.compile(pat_tpl)

# Collect adapter info: {layer_idx: {attn_type: {adapter_idx: prompt_len}}}
adapter_info = {}

for k, v in sd.items():
    m = pat.match(k)
    if m:
        layer_idx = int(m.group(1))
        attn_type = m.group(2)
        adapter_idx = int(m.group(3))
        prompt_len = v.shape[0] if hasattr(v, "shape") else v.size(0)

        if layer_idx not in adapter_info:
            adapter_info[layer_idx] = {}
        if attn_type not in adapter_info[layer_idx]:
            adapter_info[layer_idx][attn_type] = {}

        adapter_info[layer_idx][attn_type][adapter_idx] = prompt_len

# Build adapters for each layer
print("\nBuilding adapters from checkpoint...")
for layer_idx, attn_dict in sorted(adapter_info.items()):
    layer = lm.transformer.layers[layer_idx]

    for attn_type, idx_to_len in attn_dict.items():
        attn = getattr(layer, attn_type, None)
        if attn is None:
            continue

        # Sort by adapter index and get specs
        adapter_specs = []
        for idx in sorted(idx_to_len.keys()):
            prompt_len = idx_to_len[idx]
            stride, gate_init = get_stride_and_gate(prompt_len)
            adapter_specs.append((prompt_len, stride, gate_init))
            print(f"  Layer {layer_idx}.{attn_type}.adapter[{idx}]: "
                  f"prompt_len={prompt_len}, stride={stride}, gate_init={gate_init}")

        build_adapters_for_attn(attn, adapter_specs)

# === Load the fine-tuned weights ===
print("\nLoading checkpoint weights...")
missing, unexpected = lm.load_state_dict(sd, strict=False)
print(f"  Missing keys: {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")

if missing:
    # Check if missing keys are only non-critical
    critical_missing = [k for k in missing if not any(
        skip in k for skip in ['adapter_weights', 'ema', 'fsdp']
    )]
    if critical_missing:
        print(f"  WARNING: Critical missing keys: {critical_missing[:5]}")

# === Move model to device and eval mode ===
print(f"\nMoving model to {device}...")
model.lm = model.lm.to(device).eval()
model.compression_model = model.compression_model.to(device).eval()

# === Set generation parameters ===
model.set_generation_params(
    duration=30,
    use_sampling=True,
    top_k=250,
    temperature=1.0,
    cfg_coef=3.0
)

# === Text prompts ===
descriptions = [
    "Relaxing classical piece with soft flute and harp",
    "Peaceful classical background with simple harmony",
    "Romantic classical tune with gentle pacing"
]

# === Generate and write outputs ===
print("\n" + "="*60)
print("Starting generation...")
print("="*60)

with torch.inference_mode():
    for desc in descriptions:
        print(f"\n{'='*60}")
        print(f"Prompt: {desc!r}")
        print(f"{'='*60}")

        style_name = desc.split(" ")[0].lower()

        for variation in range(1, 4):
            print(f"\n  Generating variation {variation}/3...")
            wav = model.generate([desc], progress=True)

            filename = out_dir / f"{style_name}_v{variation}"
            audio_write(
                str(filename),
                wav[0].cpu(),
                model.sample_rate,
                strategy="loudness",
                loudness_compressor=True
            )
            print(f"  ✓ Saved {filename}.wav")

print("\n" + "="*60)
print("✓ Generation complete!")
print(f"✓ Output directory: {out_dir}")
print("="*60)
