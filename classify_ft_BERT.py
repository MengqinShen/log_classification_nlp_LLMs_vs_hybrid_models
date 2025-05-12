from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

# Load the model and tokenizer from the saved directory
model = AutoModelForSequenceClassification.from_pretrained("models/bert-log-model")
tokenizer = AutoTokenizer.from_pretrained("models/bert-log-model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
model.to(device)  # Move the model to GPU or CPU depending on availability
MP = {0 : "Critical Error",
    1 : "Deprecation Warning",
    2: "Error",
    3: "HTTP Status",
    4: "Resource Usage",
    5: "Security Alert",
    6: "System Notification",
    7: "User Action",
    8: "Workflow Error"}

def classify_with_ft_bert(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Perform inference (get model output)
    with torch.no_grad():  # Disable gradients during inference
        outputs = model(**inputs)

    # Get predictions (logits)
    logits = outputs.logits

    # Get the predicted class (the index with the highest probability)
    predicted_class = torch.argmax(logits, dim=-1).item()

    return MP[predicted_class]

def classify_csv(input_file):
    df = pd.read_csv(input_file)
    logs = df['log_message']
    # df["target_label"] = classify_with_ft_bert(list(df["log_message"]))
    labels = []
    for log in logs:
        label = classify_with_ft_bert(log)
        # print(log, "->", label)
        labels.append(label)

    # Save the modified file
    df["target_label"] = labels
    output_file = "resources/output_ft.csv"
    df.to_csv(output_file, index=False)

    return output_file

import time
if __name__ == '__main__':
    # start_time = time.time()
    classify_csv("resources/test.csv")
    # end_time = time.time()
    # model_time = end_time - start_time
    # print(f"Model processing time: {model_time:.4f} seconds")
