import torch
import torch.nn as nn
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from get_dataset import get_dataset
import random 
from backpack import backpack, extend
from backpack.extensions import DiagHessian

CUDA = torch.cuda.is_available()
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

filtered_dataset = get_dataset()

tokenized_dataset = tokenizer(" ".join(filtered_dataset['text']), return_tensors='pt')

seqlen = 1024
nsamples = 4

# TODO Make a function out of this
trainloader = []
for _ in range(nsamples):
    i = random.randint(0, tokenized_dataset.input_ids.shape[1] - seqlen - 1)
    j = i + seqlen
    inp = tokenized_dataset.input_ids[:, i:j]
    tar = inp.clone()
    # tar[:, :-1] = -100
    trainloader.append((inp, tar))

train_inputs = [x[0].squeeze() for x in trainloader]
train_labels = [x[1].squeeze() for x in trainloader]
train_inputs = torch.stack(train_inputs)
train_labels = torch.stack(train_labels)





model = extend(model)
loss_fct = nn.CrossEntropyLoss()
loss_fct = extend(loss_fct)


if CUDA:
    train_inputs = train_inputs.to("cuda")
    train_labels = train_labels.to("cuda")
    model  = model.to("cuda")
    loss_fct = loss_fct.to("cuda")

output = model(train_inputs)
logits = output.logits

# from modeling_bloom.py BloomForCausalLM forward method

# Shift so that tokens < n predict n
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = train_labels[..., 1:].contiguous()
batch_size, seqlen, vocab_size = shift_logits.shape

loss = loss_fct(shift_logits.view(batch_size * seqlen, vocab_size), shift_labels.view(batch_size * seqlen)
            )

with backpack(DiagHessian()):
    loss_fct.backward()

h = torch.cat([param.diag_h.flatten() for param in model.parameters()])
weight = torch.cat([param.flatten() for param in model.parameters()])
saliency = 1/2* (weight*2) * h # saliency matrix

prune_ratio = 0.2

k = int(prune_ratio * len(weight)) # number of parameters to prune
topk_indices = torch.topk(-saliency, k).indices # prune the smallest values i.e least important with negative

weight[topk_indices] = 0

param_index = n_pruned = n_param = 0
for name, param in model.named_parameters():
    # Note: bias is not pruned so explicitly avoiding
    # TODO check if this is the case for BLOOM
    if "bias" in name:
        continue
    num_params = param.numel()
    layer_saliency = weight[param_index : param_index + num_params].view(param.size())
    param.data = layer_saliency
    param_index += num_params

    # if verbose:
    n_pruned += torch.sum(param.data == 0).item()
    n_param += num_params
    mean = round(torch.mean(layer_saliency).item(), 5)
    std = round(torch.std(layer_saliency).item(), 5)
    min_value = torch.min(layer_saliency).item()
    max_value = torch.max(layer_saliency).item()
    print(f"{name = } {mean = } {std = } {num_params = }")
    print(f"num of zeros: {torch.sum(param == 0).item()} / {num_params}")
    print(f"{min_value = } {max_value = }")
    print(" ----------------------------------------")



