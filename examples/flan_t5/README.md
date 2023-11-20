# Encoder-Decoder

This document shows how to build and run an Flan T5 model in TensorRT-LLM on NVIDIA GPUs.

## Overview

The TensorRT-LLM Enc-Dec implementation can be found in [tensorrt_llm/models/enc_dec/model.py](../../tensorrt_llm/models/flan_t5/model.py). The TensorRT-LLM Flan-T5 example code is located in [`examples/flan_t5`](./). There are two main files in that folder:

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the flan-t5-xl model,
 * [`run.py`](./run.py) to run the inference on an input text.

## Usage

The TensorRT-LLM Flan-T5 example code locates at [examples/flan_t5](./). It takes HF weights as input, and builds the corresponding TensorRT engines. For single GPU, there will be two TensorRT engines, one for Encoder and one for Decoder.

## Flan-T5 Model Support
- [Flan-T5](https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5)

### Build TensorRT engine(s)

Need to prepare the HuggingFace Flan-T5 checkpoint first by following the guides here https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5.

TensorRT-LLM Flan-T56 builds TensorRT engine(s) from HF checkpoint. For the first time of running this example, user needs to download the Flan-T5 model ckpt from HF. After obtaining the HF Flan-T5 ckpt, user can build the TensorRT engines.

```bash
# download flan-t5-xl ckpt to ./models (one-time)
python download.py

# Build flan-t5-xl using a single GPU and FP16, supporting beam search up to 3 beam_width
python build.py --model_dir ./models/ \
                --use_bert_attention_plugin \
                --use_gpt_attention_plugin \
                --dtype float16 \
                --max_beam_width 3
# build.py will by default save the TRT engines into ./trt_engines
```

### Run

To run a TensorRT-LLM Flan-T5 model using the engines generated by build.py

```bash
# Run inference with beam search
python3 run.py --max_new_token=64 --num_beams=3
```