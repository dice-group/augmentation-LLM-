from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd

class REDataset_entities(Dataset):
    def __init__(self, dataframe, tokenizer, max_seq_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']
        entities = self.data.iloc[idx]['entities']
        relation = self.data.iloc[idx]['relation_id']
        entities_text = " [SEP] ".join(entities)
        text = text + " [SEP] " + entities_text
        inputs = self.tokenizer(text, padding='max_length', max_length=self.max_seq_length, return_tensors='pt', truncation=True)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return input_ids, attention_mask, relation

class REDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_seq_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']
        #entities = self.data.iloc[idx]['entities']
        relation = self.data.iloc[idx]['relation_id']
        #entities_text = " [SEP] ".join(entities)
        #text = text + " [SEP] " + entities_text
        inputs = self.tokenizer(text, padding='max_length', max_length=self.max_seq_length, return_tensors='pt', truncation=True)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return input_ids, attention_mask, relation


class REModelWithAttention(nn.Module):
    def __init__(self, tokenizer, num_classes):
        super(REModelWithAttention, self).__init__()
        self.tokenizer = tokenizer
        self.bert = AutoModel.from_pretrained("bert-base-uncased")  # You can choose a different transformer model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

def train(model, train_loader, valid_loader, criterion, optimizer,device, patience=5, num_epochs=20):
    best_loss = float('inf')
    current_patience = 0
    val_loss = []
    trn_loss = []
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, targets in train_loader:
            input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        #Evaluation
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for input_ids, attention_mask, targets in valid_loader:
                input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, targets)
                total_loss += loss.item()
            average_loss = total_loss / len(valid_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}- Validation Loss: {average_loss:.4f}')
        val_loss.append(average_loss)
        trn_loss.append(avg_train_loss)
        # Check for early stopping
        if average_loss < best_loss:
            best_loss = average_loss
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                print(f'Early stopping after {epoch+1} epochs without improvement.')
                break
    return trn_loss, val_loss

def evaluate_model(model, data_loader,device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
 
    with torch.no_grad():
        for input_ids, attention_mask, targets in data_loader:
            input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    average_loss = total_loss / len(data_loader)

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def sampleData(df,group_by):
    balanced_data = []

    for relation, group in df.groupby(group_by):
        # Sample sentences for each relation up to the minimum sentence count
        sampled_group = group.sample(21)
        balanced_data.append(sampled_group)

    # Combine the sampled data into a new DataFrame
    new_df = pd.concat(balanced_data, ignore_index=True)
    return new_df