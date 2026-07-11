# Evaluation Model Snapshot — lora-step1797

This directory contains the **modeling source code** (`.py` files) used for
evaluation of the LoRA checkpoint at step 1797.

## Restoring config / weight files

The tokenizer/config/weight files are **not tracked in git** (they are large or
checkpoint-specific). To restore them, copy or symlink from a checkpoint:

```bash
CKPT="../../outputs/lora-r32-seq3072/checkpoint-XXXX"
ln -sf "$CKPT"/config.json .
ln -sf "$CKPT"/generation_config.json .
ln -sf "$CKPT"/preprocessor_config.json .
ln -sf "$CKPT"/processor_config.json .
ln -sf "$CKPT"/tokenizer_config.json .
ln -sf "$CKPT"/special_tokens_map.json .
ln -sf "$CKPT"/added_tokens.json .
ln -sf "$CKPT"/merges.txt .
ln -sf "$CKPT"/vocab.json .
ln -sf "$CKPT"/chat_template.jinja .
ln -sf "$CKPT"/model.safetensors.index.json .
ln -sf "$CKPT"/model-0000*-of-*.safetensors .
```

> **Note:** The original `checkpoint-1797` has been cleaned up. Use the nearest
> available checkpoint (e.g. `checkpoint-2655` or `milestones/checkpoint-2000`).
