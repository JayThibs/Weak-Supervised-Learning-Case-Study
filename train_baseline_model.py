import datasets
import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb

NUM_EPOCHS = 5
BATCH_SIZE = 16
BASE_MODEL_NAME = "bert-base-cased"
LEARNING_RATE = 1e-5

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

wandb.init(project="fsdl21_bert_baseline", entity="kkoehncke")


def merge_title_with_content(example):
    example["content"] = example["title"] + " " + example["content"]
    return example


def encode(batch):
    return tokenizer(
        batch["content"],
        add_special_tokens=True,
        return_attention_mask=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="np",
    )


dbpedia_dataset = datasets.load_dataset("dbpedia_14")
num_classes = dbpedia_dataset["train"].info.features["label"].num_classes
dbpedia_dataset = dbpedia_dataset.map(merge_title_with_content, num_proc=10)
dbpedia_dataset = dbpedia_dataset.rename_column("label", "labels")
dbpedia_dataset = dbpedia_dataset.map(encode, batched=True, num_proc=10)
dbpedia_dataset.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"]
)

# DEBUGGING - Splice dataset to use smaller number of samples
#
# train_dataloader = torch.utils.data.DataLoader(
#     dbpedia_dataset["train"].select(
#         list(
#             np.random.randint(low=0, high=len(dbpedia_dataset["train"]) - 1, size=1000)
#         )
#     ),
#     batch_size=BATCH_SIZE,
#     shuffle=True,
# )
# test_dataloader = torch.utils.data.DataLoader(
#     dbpedia_dataset["test"].select(
#         list(np.random.randint(low=0, high=len(dbpedia_dataset["test"]) - 1, size=1000))
#     ),
#     batch_size=BATCH_SIZE,
#     shuffle=False,

train_dataloader = torch.utils.data.DataLoader(
    dbpedia_dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_dataloader = torch.utils.data.DataLoader(
    dbpedia_dataset["test"],
    batch_size=BATCH_SIZE,
    shuffle=False,
)

baseline_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
)

# Modify linear layer to match number of classes
baseline_model.classifier = torch.nn.Linear(
    baseline_model.config.hidden_size, num_classes
)
baseline_model.config.num_labels = num_classes
baseline_model.num_labels = num_classes

device = "cuda" if torch.cuda.is_available() else "cpu"
baseline_model.train().to(device)
optimizer = torch.optim.AdamW(params=baseline_model.parameters(), lr=LEARNING_RATE)

config = wandb.config
config.learning_rate = LEARNING_RATE
config.epochs = NUM_EPOCHS
config.batch_size = BATCH_SIZE
config.model_architecture = BASE_MODEL_NAME
wandb.watch(baseline_model)

# Training Loop
for epoch in range(NUM_EPOCHS):
    progress_bar = tqdm(train_dataloader)
    for iteration, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = baseline_model(**batch)
        optimizer.zero_grad()
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
        optimizer.step()
        if iteration % 10 == 0:
            progress_bar.set_description(
                f"epoch {epoch} iteration {iteration}: train loss {loss.item():.5f}"
            )
            wandb.log({"train loss": loss.item()})

    # Testing Loop
    progress_bar = tqdm(test_dataloader)
    baseline_model.eval()
    Y_true = []
    Y_predict = []
    with torch.no_grad():
        for iteration, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = baseline_model(**batch)
            loss = outputs["loss"]
            Y_predicted_probas_batch = torch.softmax(outputs["logits"], dim=1)
            Y_predict_batch = (
                torch.max(Y_predicted_probas_batch, dim=1)[1].data.cpu().numpy()
            )
            Y_true_batch = batch["labels"].data.cpu().numpy()
            Y_true += Y_true_batch.tolist()
            Y_predict += Y_predict_batch.tolist()
            if iteration % 10 == 0:
                progress_bar.set_description(
                    f"epoch {epoch} iteration {iteration}: test loss {loss.item():.5f}"
                )
                wandb.log({"test loss": loss.item()})
        f1_test_score = f1_score(Y_true, Y_predict, average="macro")
        print(f"Test F1 Score for epoch {epoch}: {f1_test_score:.5f}")
        wandb.log({"test F1": f1_test_score})

# Save network state dict
state_dict = baseline_model.state_dict()
torch.save(state_dict, "network.p")
