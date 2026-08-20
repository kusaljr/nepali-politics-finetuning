# Private model access

The trained weights are stored in the private Lightning Studio
`nepali-sft-20260818` under teamspace
`kusallamsal/financial-llm-training-project`.

Ready-to-download archives:

- `private-models/full-e3-l1024-seed42-model.tar.gz` (410 MB)
- `private-models/lora-r16-e3-l1024-seed7-adapter.tar.gz` (19 MB)
- `private-models/SHA256SUMS`

All ten uncompressed run directories remain under
`nepali-politics-finetuning/runs/`. Download an archive with:

```bash
lightning studio cp \
  lit://kusallamsal/financial-llm-training-project/studios/nepali-sft-20260818/nepali-politics-finetuning/private-models/full-e3-l1024-seed42-model.tar.gz \
  .
```

After extraction, pass the full model directory to
`generate_predictions.py --fine-tuned-model`. Pass the LoRA model directory to
`--adapter`; loading the adapter also requires access to the gated Gemma base
model.
