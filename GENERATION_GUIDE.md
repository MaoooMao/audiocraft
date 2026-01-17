# Multi-Scale Adapter Generation Guide

## 关键修正说明

### ❌ 原代码的问题

1. **`choose_stride` 映射错误**
   ```python
   # 原代码（错误）
   if L >= 256:
       return 256  # ❌ 新配置中 prompt_len=256 应该对应 stride=2
   ```

2. **缺少 `gate_init` 参数**
   - 每个adapter有不同的gate初始化值
   - 原代码使用默认值 `-4.0`，不是最优的

3. **`adapter_weights` 初始化错误**
   - 原代码初始化为 `torch.zeros()`
   - 应该是 `[-0.2, 0.6, -0.4]`

### ✅ 新代码的改进

1. **自动检测配置**
   - 从checkpoint的prompt_len自动推断正确的stride和gate_init
   - 支持新旧两种配置

2. **正确的adapter配置映射**
   ```python
   # 新配置 (2025-01-17最新代码)
   prompt_len=256 → stride=2,  gate_init=-2.2  (short-range)
   prompt_len=64  → stride=8,  gate_init=-1.5  (mid-range)
   prompt_len=16  → stride=32, gate_init=-3.0  (long-range)

   # 旧配置 (如果checkpoint是用旧代码训练的)
   prompt_len=512 → stride=8,   gate_init=-2.2
   prompt_len=64  → stride=96,  gate_init=-1.5
   prompt_len=16  → stride=256, gate_init=-3.0
   ```

3. **正确的adapter_weights初始化**
   ```python
   attn.adapter_weights = nn.Parameter(torch.tensor([-0.2, 0.6, -0.4]))
   ```

## 使用方法

### 方法1: 直接运行新脚本

```bash
python generate_with_adapters.py
```

新脚本会：
1. 自动检测checkpoint中的adapter配置
2. 打印每个adapter的详细信息
3. 正确加载并生成音乐

### 方法2: 修改旧代码

如果你想修改原来的代码，需要改3个地方：

#### 修改1: `choose_stride` 函数

```python
# 旧版本
def choose_stride(L):
    if L >= 256:
        return 256
    if L >= 128:
        return 96
    if L >= 64:
        return 8
    return 1

# ↓↓↓ 改为 ↓↓↓

# 新版本 - 支持新配置
def choose_stride(L):
    if L >= 256:
        return 2   # ✅ 修正：256 → 2
    elif L >= 64:
        return 8
    elif L >= 16:
        return 32  # ✅ 修正：添加这个分支
    return 1
```

#### 修改2: 添加 `gate_init` 参数

```python
# 旧版本
ContentAwareMultiScaleAdapter(
    embed_dim=attn.embed_dim,
    num_heads=attn.num_heads,
    stride=choose_stride(int(L)),
    prompt_len=int(L),
    device="cpu",
    dtype=torch.float32
)

# ↓↓↓ 改为 ↓↓↓

# 新版本
def choose_gate_init(L):
    if L >= 256:
        return -2.2
    elif L >= 64:
        return -1.5
    else:
        return -3.0

ContentAwareMultiScaleAdapter(
    embed_dim=attn.embed_dim,
    num_heads=attn.num_heads,
    stride=choose_stride(int(L)),
    prompt_len=int(L),
    gate_init=choose_gate_init(int(L)),  # ✅ 添加这一行
    device="cpu",
    dtype=torch.float32
)
```

#### 修改3: 修正 `adapter_weights` 初始化

```python
# 旧版本
if not hasattr(attn, "adapter_weights"):
    attn.adapter_weights = nn.Parameter(torch.zeros(len(attn.adapters)))

# ↓↓↓ 改为 ↓↓↓

# 新版本
if not hasattr(attn, "adapter_weights"):
    if len(attn.adapters) == 3:
        attn.adapter_weights = nn.Parameter(torch.tensor([-0.2, 0.6, -0.4]))
    else:
        attn.adapter_weights = nn.Parameter(torch.zeros(len(attn.adapters)))
```

## 检查你的checkpoint版本

运行新脚本后，看输出：

```
Building adapters from checkpoint...
  Layer 0.self_attn.adapter[0]: prompt_len=256, stride=2, gate_init=-2.2
  Layer 0.self_attn.adapter[1]: prompt_len=64, stride=8, gate_init=-1.5
  Layer 0.self_attn.adapter[2]: prompt_len=16, stride=32, gate_init=-3.0
```

- **如果看到 `stride=2, 8, 32`** → 你的checkpoint是用新代码训练的 ✅
- **如果看到 `stride=8, 96, 256`** → 你的checkpoint是用旧代码训练的 ⚠️

## 旧checkpoint的处理

如果你的 `text2music_3ad_epoch_75.th` 是用旧配置训练的：

1. **选项A（推荐）**: 用新配置重新训练
   - 新配置更稳定（stride更小，不会过度下采样）
   - 已经禁用了高频增强分支（避免噪音）

2. **选项B**: 修改新脚本支持旧配置
   - 在 `ADAPTER_CONFIGS` 中取消注释旧配置：
   ```python
   ADAPTER_CONFIGS = {
       # NEW configuration
       256: (2, -2.2),
       64: (8, -1.5),
       16: (32, -3.0),

       # OLD configuration - 取消注释这3行
       512: (8, -2.2),    # ← 如果你的checkpoint有prompt_len=512
       # 64: (96, -1.5),  # ← 注意：64会和新配置冲突，需要手动处理
       # 16: (256, -3.0), # ← 注意：16会和新配置冲突，需要手动处理
   }
   ```

## 预期输出

生成时应该看到：

```
Loading base MusicGen model...
Loading checkpoint from /content/drive/MyDrive/...
Building adapters from checkpoint...
  Layer 0.self_attn.adapter[0]: prompt_len=256, stride=2, gate_init=-2.2
  Layer 0.self_attn.adapter[1]: prompt_len=64, stride=8, gate_init=-1.5
  Layer 0.self_attn.adapter[2]: prompt_len=16, stride=32, gate_init=-3.0
  ...

Loading checkpoint weights...
  Missing keys: 0
  Unexpected keys: 0

Moving model to cuda...

============================================================
Starting generation...
============================================================

============================================================
Prompt: 'Relaxing classical piece with soft flute and harp'
============================================================

  Generating variation 1/3...
  ✓ Saved relaxing_v1.wav
  ...
```

## 故障排查

### 问题1: "Missing keys" 很多

**原因**: checkpoint和模型架构不匹配

**解决**:
- 检查checkpoint是用哪个版本的代码训练的
- 确保 `ADAPTER_CONFIGS` 包含正确的映射

### 问题2: 生成的音乐质量差/有噪音

**原因**: 可能是旧配置的问题（stride太大或高频分支导致）

**解决**:
- 用新配置重新训练（stride=2,8,32）
- 或者等待进一步的诊断结果

### 问题3: CUDA out of memory

**解决**:
```python
# 减少生成时长
model.set_generation_params(
    duration=15,  # 从30秒减少到15秒
    ...
)

# 或者一次只生成一个
for desc in descriptions:
    for variation in range(1):  # 从3改为1
        ...
```

## 参考

- 新的adapter实现: `audiocraft/modules/transformer.py:112-269` (ContentAwareMultiScaleAdapter)
- Adapter集成: `audiocraft/modules/transformer.py:388-409` (StreamingMultiheadAttention.__init__)
- 生成脚本: `generate_with_adapters.py`
