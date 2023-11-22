import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

MODEL_NAME = os.environ.get("MODEL_NAME")
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME")
HF_TOKEN = os.environ.get("HF_TOKEN")


tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, token=HF_TOKEN)

input_ids = tokenizer("translate English to German: The house is wonderful.",
                      return_tensors="pt").input_ids
outputs = model.generate(input_ids, decoder_input_ids=torch.IntTensor([[
    0,
]]))
print("input", input_ids, "\noutput", outputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

torch.save(model.state_dict(), f"./models/{CHECKPOINT_NAME}.ckpt")

for k, v in model.state_dict().items():
    print(k)
