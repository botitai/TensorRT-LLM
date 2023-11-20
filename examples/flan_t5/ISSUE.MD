# Encoder-Decoder have issue when return logits

This document shows how to reproceduce the error.

## Overview

In some usecases, we need to retrieve the `outputs_dict` from decoder instead of just `final_output_ids`

To do that, we need to pass `return_dict=True` for decoder's `decode` function and `gather_all_token_logits=True` for model's config. Details in [this commit](https://github.com/botitai/TensorRT-LLM/commit/4f854361a98a2727ceaec05d81a5a3763c2568f7) changes.

## Error

When trying to run `run.py`,there is an error raised

```cmd
python3 run.py --max_new_token=1 --num_beams=1
```

> Please refer to [Readme.md](README.md) for details how to run the model)

```bash
[11/20/2023-15:35:33] [TRT-LLM] [E] The following expected tensors are not found: set()
[11/20/2023-15:35:33] [TRT-LLM] [E] Those tensors in engine are not expected: {'last_token_ids'}
[11/20/2023-15:35:33] [TRT-LLM] [E] Expected tensor names: ['input_ids', 'logits', 'cache_indirection', 'past_key_value_0', 'past_key_value_1', 'past_key_value_2', 'past_key_value_3', 'past_key_value_4', 'past_key_value_5', 'past_key_value_6', 'past_key_value_7', 'past_key_value_8', 'past_key_value_9', 'past_key_value_10', 'past_key_value_11', 'past_key_value_12', 'past_key_value_13', 'past_key_value_14', 'past_key_value_15', 'past_key_value_16', 'past_key_value_17', 'past_key_value_18', 'past_key_value_19', 'past_key_value_20', 'past_key_value_21', 'past_key_value_22', 'past_key_value_23', 'present_key_value_0', 'present_key_value_1', 'present_key_value_2', 'present_key_value_3', 'present_key_value_4', 'present_key_value_5', 'present_key_value_6', 'present_key_value_7', 'present_key_value_8', 'present_key_value_9', 'present_key_value_10', 'present_key_value_11', 'present_key_value_12', 'present_key_value_13', 'present_key_value_14', 'present_key_value_15', 'present_key_value_16', 'present_key_value_17', 'present_key_value_18', 'present_key_value_19', 'present_key_value_20', 'present_key_value_21', 'present_key_value_22', 'present_key_value_23', 'sequence_length', 'context_lengths', 'host_request_types', 'host_past_key_value_lengths', 'cross_present_key_value_0', 'cross_present_key_value_1', 'cross_present_key_value_2', 'cross_present_key_value_3', 'cross_present_key_value_4', 'cross_present_key_value_5', 'cross_present_key_value_6', 'cross_present_key_value_7', 'cross_present_key_value_8', 'cross_present_key_value_9', 'cross_present_key_value_10', 'cross_present_key_value_11', 'cross_present_key_value_12', 'cross_present_key_value_13', 'cross_present_key_value_14', 'cross_present_key_value_15', 'cross_present_key_value_16', 'cross_present_key_value_17', 'cross_present_key_value_18', 'cross_present_key_value_19', 'cross_present_key_value_20', 'cross_present_key_value_21', 'cross_present_key_value_22', 'cross_present_key_value_23', 'cross_past_key_value_0', 'cross_past_key_value_1', 'cross_past_key_value_2', 'cross_past_key_value_3', 'cross_past_key_value_4', 'cross_past_key_value_5', 'cross_past_key_value_6', 'cross_past_key_value_7', 'cross_past_key_value_8', 'cross_past_key_value_9', 'cross_past_key_value_10', 'cross_past_key_value_11', 'cross_past_key_value_12', 'cross_past_key_value_13', 'cross_past_key_value_14', 'cross_past_key_value_15', 'cross_past_key_value_16', 'cross_past_key_value_17', 'cross_past_key_value_18', 'cross_past_key_value_19', 'cross_past_key_value_20', 'cross_past_key_value_21', 'cross_past_key_value_22', 'cross_past_key_value_23', 'encoder_output', 'encoder_input_lengths', 'encoder_max_input_length']
11/20/2023-15:35:33] [TRT-LLM] [E] Found tensor names: ['input_ids', 'host_past_key_value_lengths', 'sequence_length', 'context_lengths', 'host_request_types', 'encoder_input_lengths', 'encoder_max_input_length', 'last_token_ids', 'cache_indirection', 'encoder_output', 'past_key_value_0', 'cross_past_key_value_0', 'past_key_value_1', 'cross_past_key_value_1', 'past_key_value_2', 'cross_past_key_value_2', 'past_key_value_3', 'cross_past_key_value_3', 'past_key_value_4', 'cross_past_key_value_4', 'past_key_value_5', 'cross_past_key_value_5', 'past_key_value_6', 'cross_past_key_value_6', 'past_key_value_7', 'cross_past_key_value_7', 'past_key_value_8', 'cross_past_key_value_8', 'past_key_value_9', 'cross_past_key_value_9', 'past_key_value_10', 'cross_past_key_value_10', 'past_key_value_11', 'cross_past_key_value_11', 'past_key_value_12', 'cross_past_key_value_12', 'past_key_value_13', 'cross_past_key_value_13', 'past_key_value_14', 'cross_past_key_value_14', 'past_key_value_15', 'cross_past_key_value_15', 'past_key_value_16', 'cross_past_key_value_16', 'past_key_value_17', 'cross_past_key_value_17', 'past_key_value_18', 'cross_past_key_value_18', 'past_key_value_19', 'cross_past_key_value_19', 'past_key_value_20', 'cross_past_key_value_20', 'past_key_value_21', 'cross_past_key_value_21', 'past_key_value_22', 'cross_past_key_value_22', 'past_key_value_23', 'cross_past_key_value_23', 'present_key_value_0', 'cross_present_key_value_0', 'present_key_value_1', 'cross_present_key_value_1', 'present_key_value_2', 'cross_present_key_value_2', 'present_key_value_3', 'cross_present_key_value_3', 'present_key_value_4', 'cross_present_key_value_4', 'present_key_value_5', 'cross_present_key_value_5', 'present_key_value_6', 'cross_present_key_value_6', 'present_key_value_7', 'cross_present_key_value_7', 'present_key_value_8', 'cross_present_key_value_8', 'present_key_value_9', 'cross_present_key_value_9', 'present_key_value_10', 'cross_present_key_value_10', 'present_key_value_11', 'cross_present_key_value_11', 'present_key_value_12', 'cross_present_key_value_12', 'present_key_value_13', 'cross_present_key_value_13', 'present_key_value_14', 'cross_present_key_value_14', 'present_key_value_15', 'cross_present_key_value_15', 'present_key_value_16', 'cross_present_key_value_16', 'present_key_value_17', 'cross_present_key_value_17', 'present_key_value_18', 'cross_present_key_value_18', 'present_key_value_19', 'cross_present_key_value_19', 'present_key_value_20', 'cross_present_key_value_20', 'present_key_value_21', 'cross_present_key_value_21', 'present_key_value_22', 'cross_present_key_value_22', 'present_key_value_23', 'cross_present_key_value_23', 'logits']
Traceback (most recent call last):
  File "/models/flan_t5/1/run.py", line 303, in <module>
    tllm_model = TRTLLMEncDecModel.from_engine(args.engine_name,
  File "/models/flan_t5/1/run.py", line 127, in from_engine
    return cls(engine_name, engine_dir, debug_mode=debug_mode)
  File "/models/flan_t5/1/run.py", line 118, in __init__
    self.decoder_session = tensorrt_llm.runtime.GenerationSession(
  File "/usr/local/lib/python3.10/dist-packages/tensorrt_llm/runtime/generation.py", line 430, in __init__
    raise RuntimeError(
RuntimeError: Tensor names in engine are not the same as expected, to use this GenerationSession, you need to use GPTLMHeadModel.prepare_inputs to create TRT Network inputs.
```
