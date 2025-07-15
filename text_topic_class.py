#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 22:04:03 2025

@author: kissmarcell
"""

# Load necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import spacy

# Define model name correctly
model_name ="kissmarci00/autotrain-roberta-base-topic"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load Excel file
file_path = "Data/FED_text.xlsx"  # Ensure this file exists
sheet_name = "all"  # Adjust if needed
text_column = "text"  # Ensure this matches the actual column name in Excel

# Read the sheet
df = pd.read_excel(file_path, sheet_name=sheet_name)

df['order'] = df.groupby(['date','type']).cumcount() + 1


# Ensure the column exists
if text_column not in df.columns:
    raise ValueError(f"Column '{text_column}' not found in the Excel file.")

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to split text into sentences
def split_into_sentences(paragraph):
    doc = nlp(paragraph)
    return [sent.text.strip() for sent in doc.sents]

# Expand text into sentences
expanded_rows = []
for _, row in df.iterrows():
    sentences = split_into_sentences(row[text_column])
    for sentence in sentences:
        expanded_rows.append({**row.to_dict(), 'sentence': sentence})

text = pd.DataFrame(expanded_rows)

# Function to classify text and get probabilities
def classify_text(text):
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze()  # Convert logits to probabilities
        
        # Get predicted label and its probability
        predicted_label_idx = torch.argmax(probabilities).item()
        predicted_probability = probabilities[predicted_label_idx].item()
        
        # Ensure model has id2label mapping
        if hasattr(model.config, "id2label"):
            predicted_label = model.config.id2label.get(predicted_label_idx, f"Label {predicted_label_idx}")
        else:
            predicted_label = f"Label {predicted_label_idx}"  # Fallback if no mapping
        
        return predicted_label, predicted_probability
        
    except Exception as e:
        return str(e), 0.0  # Return error message if classification fails

# Apply classification to sentences
text[["predicted_label_topic", "predicted_probability"]] = text["sentence"].astype(str).apply(
    lambda x: pd.Series(classify_text(x))
)


top_500_df=pd.read_excel("Data/top500.xlsx")

words_to_remove = ["period", "run", "meeting","low","little","intermeeting","noted","likely","members","generally","somewhat","net","data","near","range","appeared","levels","level","lower","suggested","information","anticipated","based","number","january","pointed","moved","june","judged","september","slightly","december","april","july","october","early","reported","viewed","saw","elevated","high","month","situation","overall","november","seen","couple","total","february","addition","modest","relatively","large","past","end","incoming","additional","long","slow","discussed","review","available","reportedly","despite","slower","result","suggesting","assessments","broader","possible","largely","non","weaker","slowing","initial","close","implications","need","highly","begin"]  # Replace with actual words

# Filter out the selected words from top_500_df
top_500_df = top_500_df[~top_500_df['Word'].isin(words_to_remove)]

# Update the top words set for irrelevant sentence marking
top_words_set = set(top_500_df['Word'])


# Function to check if a sentence contains any of the top words
def contains_top_words(sentence):
    tokens = [token.text.lower() for token in nlp(sentence) if token.is_alpha]  # Tokenize and filter words
    return any(word in top_words_set for word in tokens)

# Apply function to create a new column 'contains_top_words'
text['contains_top_words'] = text['sentence'].apply(contains_top_words)

# Mark sentences that do NOT contain any top words
text['is_irrelevant'] = ~text['contains_top_words']

# If a sentence is irrelevant, set 'predicted_label_topic' to 'irrelevant'
text.loc[text['is_irrelevant'], 'predicted_label_topic'] = 'irrelevant'

# Display the first few rows
print(text[['sentence', 'predicted_label_topic', 'is_irrelevant']].head())

num_irrelevant = text['is_irrelevant'].sum()
print(num_irrelevant)


text.to_excel("sentence_level_class.xlsx")

###########################################################################
joined=1
if joined==1:
    text['sentence_id'] = text.index
    
    
    
    #unite consecutive irrelevants
    
    
    import pandas as pd
    
    # Create a new column to flag rows to merge (if both "irrelevant" and same date_text and order)
    text['merge_flag'] = (text['predicted_label_topic'] == 'irrelevant') & (text['date_text'] == text['date_text'].shift(-1)) & (text['order'] == text['order'].shift(-1))
    
    # List to store merged and original sentences
    merged_sentences = []
    
    # Iterate through the dataframe and merge sentences based on the merge_flag
    i = 0
    while i < len(text):
        if i < len(text) - 1 and text['merge_flag'][i] and text['merge_flag'][i+1]:
            # Merge consecutive "irrelevant" sentences with the same date_text and order
            merged_sentence = text['sentence'][i] + ' ' + text['sentence'][i + 1]
            
            # Add the merged sentence (this is the new row)
            merged_sentences.append({
                'sentence_id': text['sentence_id'][i],  # Keeping the first sentence_id
                'sentence': merged_sentence,
                'predicted_label_topic': 'irrelevant',  # Keeping the topic as 'irrelevant'
                'date_text': text['date_text'][i],  # Same date as the first sentence
                'order': text['order'][i],  # Same order as the first sentence
                'merged': True  # New column to indicate this is a merged sentence
            })
            
            i += 2  # Skip the next sentence since it's merged
        else:
            # Keep the sentence as is
            merged_sentences.append(text.iloc[i].to_dict())
            i += 1
    
    # Convert back to DataFrame
    merged_df = pd.DataFrame(merged_sentences)
    
    # Drop duplicates and reset index if necessary
    merged_df = merged_df.drop_duplicates().reset_index(drop=True)
    
    # Resulting DataFrame
    print(merged_df)
    
    
    #merge irrelevants into it
    
    
    i=0
    while i < len(merged_df):
        if merged_df['predicted_label_topic'][i]=="irrelevant":
            if i - 1 >= 0:
                if merged_df['date_text'][i] == merged_df['date_text'][i-1] and merged_df['order'][i] == merged_df['order'][i-1]:
                    merged_df['sentence'][i-1]= merged_df['sentence'][i-1]+ ' ' + merged_df['sentence'][i]
            if i+1<len(merged_df) and i - 1 >= 0:
                if merged_df['date_text'][i] == merged_df['date_text'][i+1] and merged_df['order'][i] == merged_df['order'][i+1] and merged_df['predicted_label_topic'][i+1]!=merged_df['predicted_label_topic'][i-1]:
                    merged_df['sentence'][i+1]= merged_df['sentence'][i]+ ' ' +merged_df['sentence'][i+1]
        i+=1
    
    merged_df = merged_df[merged_df['predicted_label_topic'] != 'irrelevant']
    
    
    #merged_df=merged_df.reset_index(drop=True)
    
    #i=0
    #while i < len(merged_df) - 1:
    #    if merged_df['predicted_label_topic'][i]==merged_df['predicted_label_topic'][i+1] and merged_df['date_text'][i] == merged_df['date_text'][i+1] and merged_df['order'][i] == merged_df['order'][i+1]:
    #       merged_df['sentence'][i+1]= merged_df['sentence'][i]+ ' ' + merged_df['sentence'][i+1]
    #       merged_df = merged_df.drop(i).reset_index(drop=True)
           
        
    #    else:
    #      i+=1
    
    
    
    # MP
    
    
    from transformers import pipeline
    model_name="kissmarci00/autotrain-bert-base-uncased-bs8-e3-ms256"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    text_column = "sentence" 
    # Function to classify text
    def classify_text(text):
        try:
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,  # Ensures input does not exceed model limit
                max_length=256  # Match model's max token length
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.nn.functional.softmax(logits, dim=-1)
                predicted_label = torch.argmax(predictions, dim=-1).item()
            
            return model.config.id2label[predicted_label]  # Get the predicted label
        except Exception as e:
            return str(e)  # Return error message if classification fails
    
    # Apply classification to each row
    merged_df["predicted_label_mp"] = merged_df[text_column].astype(str).apply(classify_text)
    
    
    #ECON
    model_name="ZiweiChen/FinBERT-FOMC"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    merged_df["predicted_label_econ"] = merged_df[text_column].astype(str).apply(classify_text)
    
    
    
    
    
    import nltk
    from collections import Counter
    # Ensure necessary NLTK tokenizer is available
    nltk.download('punkt')
    
    # Load the Loughran-McDonald Master Dictionary
    
    lm_dict=pd.read_csv("Data/Loughran-McDonald_MasterDictionary_1993-2024.csv")
    
    
    
    # Extract Hedging words (assuming a column "Hedging" with 1 for hedging words)
    unc_words = set(lm_dict[lm_dict["Uncertainty"] >0]["Word"].str.lower())
    
    # Function to count hedging words in text
    def count_uncertainty_words(text):
        words = nltk.word_tokenize(text.lower())  # Tokenization
        unc_count = Counter(word for word in words if word in unc_words)
        total_unc = sum(unc_count.values())  # Count total hedging words
        total_words = len(words)  # Total words in text
        unc_density = total_unc / total_words if total_words > 0 else 0
        
        return unc_count, total_unc, unc_density
    
    merged_df[["Uncertainty_Word_Count", "Total_Word_Count", "Uncertainty_Density"]] = merged_df["sentence"].apply(
        lambda x: pd.Series(count_uncertainty_words(x))
    )
    
    
    
    
    
    # Save results to Excel
    output_file = "text_classified_joined.xlsx"
    merged_df.to_excel(output_file, index=False)
    
    print(f"Classification completed! Results saved to '{output_file}'.")



if joined==0:
    text['sentence_id'] = text.index
    
    
    
    #unite consecutive irrelevants
    
    
    import pandas as pd
    
    # Create a new column to flag rows to merge (if both "irrelevant" and same date_text and order)
    text['merge_flag'] = (text['predicted_label_topic'] == 'irrelevant') & (text['date_text'] == text['date_text'].shift(-1)) & (text['order'] == text['order'].shift(-1))
    
    # List to store merged and original sentences
    merged_sentences = []
    
    # Iterate through the dataframe and merge sentences based on the merge_flag
    i = 0
    while i < len(text):
        if i < len(text) - 1 and text['merge_flag'][i] and text['merge_flag'][i+1]:
            # Merge consecutive "irrelevant" sentences with the same date_text and order
            merged_sentence = text['sentence'][i] + ' ' + text['sentence'][i + 1]
            
            # Add the merged sentence (this is the new row)
            merged_sentences.append({
                'sentence_id': text['sentence_id'][i],  # Keeping the first sentence_id
                'sentence': merged_sentence,
                'predicted_label_topic': 'irrelevant',  # Keeping the topic as 'irrelevant'
                'date_text': text['date_text'][i],  # Same date as the first sentence
                'order': text['order'][i],  # Same order as the first sentence
                'merged': True  # New column to indicate this is a merged sentence
            })
            
            i += 2  # Skip the next sentence since it's merged
        else:
            # Keep the sentence as is
            merged_sentences.append(text.iloc[i].to_dict())
            i += 1
    
    # Convert back to DataFrame
    merged_df = pd.DataFrame(merged_sentences)
    
    # Drop duplicates and reset index if necessary
    merged_df = merged_df.drop_duplicates().reset_index(drop=True)
    
    # Resulting DataFrame
    print(merged_df)
    
    
    #merge irrelevants into it
    
    
    i=0
    while i < len(merged_df):
        if merged_df['predicted_label_topic'][i]=="irrelevant":
            if i - 1 >= 0:
                if merged_df['date_text'][i] == merged_df['date_text'][i-1] and merged_df['order'][i] == merged_df['order'][i-1]:
                    merged_df['sentence'][i-1]= merged_df['sentence'][i-1]+ ' ' + merged_df['sentence'][i]
            if i+1<len(merged_df) and i - 1 >= 0:
                if merged_df['date_text'][i] == merged_df['date_text'][i+1] and merged_df['order'][i] == merged_df['order'][i+1] and merged_df['predicted_label_topic'][i+1]!=merged_df['predicted_label_topic'][i-1]:
                    merged_df['sentence'][i+1]= merged_df['sentence'][i]+ ' ' +merged_df['sentence'][i+1]
        i+=1
    
    merged_df = merged_df[merged_df['predicted_label_topic'] != 'irrelevant']
    
    
    #merged_df=merged_df.reset_index(drop=True)
    
    #i=0
    #while i < len(merged_df) - 1:
    #    if merged_df['predicted_label_topic'][i]==merged_df['predicted_label_topic'][i+1] and merged_df['date_text'][i] == merged_df['date_text'][i+1] and merged_df['order'][i] == merged_df['order'][i+1]:
    #       merged_df['sentence'][i+1]= merged_df['sentence'][i]+ ' ' + merged_df['sentence'][i+1]
    #       merged_df = merged_df.drop(i).reset_index(drop=True)
           
        
    #    else:
    #      i+=1
    
    
    
    # MP
    
    
    from transformers import pipeline
    model_name="kissmarci00/autotrain-bert-base-uncased-bs8-e3-ms256"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    text_column = "sentence" 
    # Function to classify text
    def classify_text(text):
        try:
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,  # Ensures input does not exceed model limit
                max_length=256  # Match model's max token length
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.nn.functional.softmax(logits, dim=-1)
                predicted_label = torch.argmax(predictions, dim=-1).item()
            
            return model.config.id2label[predicted_label]  # Get the predicted label
        except Exception as e:
            return str(e)  # Return error message if classification fails
    
    # Apply classification to each row
    merged_df["predicted_label_mp"] = merged_df[text_column].astype(str).apply(classify_text)
    
    
    #ECON
    model_name="ZiweiChen/FinBERT-FOMC"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    merged_df["predicted_label_econ"] = merged_df[text_column].astype(str).apply(classify_text)
    
    
    
    
    
    import nltk
    from collections import Counter
    # Ensure necessary NLTK tokenizer is available
    nltk.download('punkt')
    
    # Load the Loughran-McDonald Master Dictionary
    
    lm_dict=pd.read_csv("/Data/Loughran-McDonald_MasterDictionary_1993-2024.csv")
    
    
    
    # Extract Hedging words (assuming a column "Hedging" with 1 for hedging words)
    unc_words = set(lm_dict[lm_dict["Uncertainty"] >0]["Word"].str.lower())
    
    # Function to count hedging words in text
    def count_uncertainty_words(text):
        words = nltk.word_tokenize(text.lower())  # Tokenization
        unc_count = Counter(word for word in words if word in unc_words)
        total_unc = sum(unc_count.values())  # Count total hedging words
        total_words = len(words)  # Total words in text
        unc_density = total_unc / total_words if total_words > 0 else 0
        
        return unc_count, total_unc, unc_density
    
    merged_df[["Uncertainty_Word_Count", "Total_Word_Count", "Uncertainty_Density"]] = merged_df["sentence"].apply(
        lambda x: pd.Series(count_uncertainty_words(x))
    )
    
    
    
    
    
    # Save results to Excel
    output_file = "text_classified_sentence.xlsx"
    merged_df.to_excel(output_file, index=False)
    
    print(f"Classification completed! Results saved to '{output_file}'.")



