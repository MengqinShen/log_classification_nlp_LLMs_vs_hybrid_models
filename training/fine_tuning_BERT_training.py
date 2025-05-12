# !pip install transformers datasets scikit-learn
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

df = pd.read_csv('/Users/mengqinshen/PycharmProjects/PythonProject/Log_Classification/training/dataset/synthetic_logs.csv')
df.target_label.unique()
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["target_label"])
# Split into train and test
df = df[["log_message", "label_id"]]
# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)
test_dataset = dataset['test']
train_dataset = dataset['train']

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["log_message"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["log_message"])
dataset = dataset.rename_column("label_id", "label")
dataset.set_format("torch")

from transformers import AutoModelForSequenceClassification
import torch  # Add this line at the top of your script

num_labels = len(set(df["label_id"]))
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

training_args = TrainingArguments(
    output_dir="./bert-log-model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    # evaluation_strategy="epoch",
    # save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)

model.save_pretrained("./saved-bert-log-model")
tokenizer.save_pretrained("./saved-bert-log-model")

from sklearn.metrics import classification_report
import torch

model.eval()  # Set to evaluation mode

all_preds = []
all_labels = []

for batch in dataset["test"]:
    inputs = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Evaluate
print(classification_report(all_labels, all_preds))
