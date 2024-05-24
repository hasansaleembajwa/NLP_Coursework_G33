from flask import Flask, request, render_template, jsonify, send_file
import torch
from transformers import BertForTokenClassification, BertTokenizer
import json
import os
import numpy as np
import datetime
import csv

app = Flask(__name__)
output_dir = './model_save/'
log_file = 'log.csv'

model = BertForTokenClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)

# Load the tag2idx mapping
with open(os.path.join(output_dir, 'tag2idx.json'), 'r') as f:
    tag2idx = json.load(f)

# Reverse the tag2idx mapping to get idx2tag
idx2tag = {idx: tag for tag, idx in tag2idx.items()}


def log_activity(sentence, predicted_tags):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = [sentence, predicted_tags, timestamp]

    # Write log entry to CSV file
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Input', 'Predicted Tags', 'Timestamp'])
        writer.writerow(log_entry)


def preprocess_sentence(sentence):
    # Tokenize the sentence
    tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)

    # Create attention mask and input tensors
    input_ids = torch.tensor([tokenized_sentence])
    attention_mask = torch.ones_like(input_ids)

    return input_ids, attention_mask


def predict_tags(sentence):
    # Preprocess the sentence
    input_ids, attention_mask = preprocess_sentence(sentence)

    # Get predictions from the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = np.argmax(outputs[0].detach().numpy(), axis=2)

    # Convert predictions to tags
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tags = [idx2tag[idx] for idx in predictions[0]]

    merged_tokens, merged_tags = [], []
    for token, tag in zip(tokens, tags):
        if token.startswith("##"):
            if merged_tokens:
                merged_tokens[-1] += token[2:]
        else:
            merged_tokens.append(token)
            merged_tags.append(tag)

   # Remove special tokens
    special_tokens = set(tokenizer.all_special_tokens)
    filtered_tokens, filtered_tags = [], []
    for token, tag in zip(merged_tokens, merged_tags):
        if token not in special_tokens:
            filtered_tokens.append(token)
            filtered_tags.append(tag)

    return list(zip(filtered_tokens, filtered_tags))


@app.route('/predict_json', methods=['GET', 'POST'])
def predict_json():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            sentence = data.get('sentence')
        else:
            sentence = request.form.get('sentence')

        if not sentence:
            return jsonify({'error': 'Form submission error: sentence key is missing'}), 400

        predicted_tags = predict_tags(sentence)
        # log_activity(sentence, predicted_tags)
        return jsonify({'sentence': sentence, 'predicted_tags': predicted_tags})
    return jsonify({'sentence': sentence, 'predicted_tags': predicted_tags})

# @app.route('/predict_json', methods=['POST'])
# def predict_json():
#     data = request.get_json()
#     sentence = data.get('sentence')

#     if not sentence:
#         return jsonify({'error': 'Form submission error: sentence key is missing'}), 400

#     predicted_tags = predict_tags(sentence)
#     return jsonify({'sentence': sentence, 'predicted_tags': predicted_tags})


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form.get('sentence')

        if not sentence:
            return render_template('index.html', error='Form submission error: sentence key is missing')

        predicted_tags = predict_tags(sentence)
        log_activity(sentence, predicted_tags)
        return render_template('index.html', sentence=sentence, predicted_tags=predicted_tags)

    return render_template('index.html')


@app.route('/download_log')
def download_log():
    return send_file(log_file, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
