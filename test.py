from transformers import BloomTokenizerFast, pipeline
from datasets import load_dataset

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")

name = "bigscience/bloom-7b1"
text = 'What is the Capital of France?\nAnswer: '
max_new_tokens = 20

def generate_from_model(model, tokenizer):
  encoded_input = tokenizer(text, return_tensors='pt')
  output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda())
  return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

pipe = pipeline(model=name, model_kwargs= {"device_map": "auto", "load_in_8bit": True}, max_new_tokens=max_new_tokens)

print(pipe(text))

dataset = load_dataset("xtreme", "MLQA.en.es")

tokenized_dataset = dataset.map(lambda x: tokenizer(x['context']), batched=True)

tokenized_dataset['validation']['answers']
