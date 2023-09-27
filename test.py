from transformers import BloomTokenizerFast, pipeline, BloomForCausalLM
from get_dataset import get_dataset
import NeuroSurgeon
from NeuroSurgeon.Models import model_configs, circuit_model
import torch
import transformers
from copy import deepcopy
from tqdm import tqdm
import random

def loss_fn(logits, labels):
    # Defining our own loss function so loss is only computed on label tokens
    index_logits = logits[:, -3]
    index_logits = index_logits.to(torch.float64)
    log_probs = index_logits.log_softmax(dim=-1)

    correct_log_probs_0 = log_probs.gather(dim=-1, index=labels[:, 0, None])[:, 0]

    index_logits = logits[:, -2]
    index_logits = index_logits.to(torch.float64)
    log_probs = index_logits.log_softmax(dim=-1)
    correct_log_probs_1 = log_probs.gather(dim=-1, index=labels[:, 1, None])[:, 0]

    return -(correct_log_probs_0 + correct_log_probs_1).mean()

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")

model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

model_1 = deepcopy(model)

filtered_dataset = get_dataset()

tokenized_dataset = tokenizer(" ".join(filtered_dataset['text']), return_tensors='pt')

seqlen = 1024
nsamples = 4

trainloader = []
for _ in range(nsamples):
    i = random.randint(0, tokenized_dataset.input_ids.shape[1] - seqlen - 1)
    j = i + seqlen
    inp = tokenized_dataset.input_ids[:, i:j]
    tar = inp.clone()
    tar[:, :-1] = -100
    trainloader.append((inp, tar))


train_inputs = [x[0].squeeze() for x in trainloader]
train_labels = [x[1].squeeze() for x in trainloader]

train_inputs = torch.stack(train_inputs).to("cuda")
train_labels = torch.stack(train_labels).to("cuda")

#tokenized_dataset = filtered_dataset.map(lambda x: tokenizer(x['text']), batched=True)

target_layers = list(model_1.state_dict().keys())
target_layers = [
    ".".join(target_layer.split(".")[:-1])
    for target_layer in target_layers
    if (".h." in target_layer
    and "weight" in target_layer
    and "ln" not in target_layer
    and "layernorm" not in target_layer
    )
    or ("lm_head" in target_layer)
]

config = model_configs.CircuitConfig(
    mask_method="magnitude_pruning",
    mask_hparams = {
        "ablation": "none", # Don't invert the learned mask
        "mask_bias": False, # Don't mask biases
        "prune_percentage": 0.1, # Prune 10% of the weights
    },
    target_layers=target_layers, # Replace the layers specified above
    freeze_base=True, # Don't train the model weights when training the mask
    add_l0=True, # Use L0 Regularization
    l0_lambda=1e-6, # Multiplier on L0 norm for balancing the loss function
)

circuit_model_1 = circuit_model.CircuitModel(config, model_1).to("cuda")
optimizer = torch.optim.AdamW(circuit_model_1.parameters(), lr=0.01)

NUM_EPOCHS = 10
FINAL_TEMP = 200

train_losses = []
test_losses = []
checkpoint_every = 5

progress_bar = tqdm(range(NUM_EPOCHS))
for epoch in range(NUM_EPOCHS):
    train_logits = circuit_model_1(input_ids=train_inputs).logits
    train_loss = loss_fn(train_logits, train_labels) + \
      (config.l0_lambda * circuit_model_1._compute_l0_loss()) # Manually adding L0 Loss
    train_loss.backward()
    train_losses.append(train_loss.cpu().item())
    optimizer.step()
    optimizer.zero_grad()
    progress_bar.update(1)

    if epoch % checkpoint_every == 0:
      with torch.inference_mode():
        test_logits = circuit_model_1(input_ids=train_inputs).logits
        test_loss = loss_fn(test_logits, train_labels)
        test_losses.append(test_loss)
        progress_bar.set_description(f"Test Loss {test_loss} " +\
          f"L0 {circuit_model_1._compute_l0_loss()}")




#####
name = "bigscience/bloom-7b1"
text = 'What is the Capital of France?\nAnswer: '
max_new_tokens = 20

def generate_from_model(model, tokenizer):
  encoded_input = tokenizer(text, return_tensors='pt')
  output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda())
  return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

pipe = pipeline(model=name, model_kwargs= {"device_map": "auto", "load_in_8bit": True}, max_new_tokens=max_new_tokens)

print(pipe(text))

