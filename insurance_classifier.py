import openai
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import time

# 🔹 1. Configurăm clientul OpenAI cu API Key din variabila de mediu (fără hardcoding)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🔹 2. Încărcăm datele
company_df = pd.read_csv("ml_insurance_challenge.csv")  # Datele companiilor
taxonomy_df = pd.read_excel("insurance_taxonomy.xlsx", engine="openpyxl")  # Taxonomia

# 🔹 3. Pregătim textul pentru clasificare
company_df["full_description"] = company_df[["business_tags", "sector", "category", "niche"]].apply(
    lambda row: " ".join(filter(None, map(str, row))), axis=1
).fillna("")

# 🔹 4. Model NLP: Sentence-BERT pentru embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 5. Creăm embeddings pentru companii și etichetele taxonomiei
company_embeddings = model.encode(company_df["full_description"], convert_to_tensor=True)
taxonomy_embeddings = model.encode(taxonomy_df["label"], convert_to_tensor=True)

# 🔹 6. Calculăm similaritatea cosinus dintre companii și etichete
sbert_sim = util.pytorch_cos_sim(company_embeddings, taxonomy_embeddings).cpu().numpy()

# 🔹 7. TF-IDF pentru potrivirea bazată pe cuvinte cheie
vectorizer = TfidfVectorizer(stop_words="english")
company_vectors = vectorizer.fit_transform(company_df["full_description"])
taxonomy_vectors = vectorizer.transform(taxonomy_df["label"])
tfidf_sim = cosine_similarity(company_vectors, taxonomy_vectors)

# 🔹 8. Fusion Strategy: combinăm scorurile SBERT și TF-IDF
final_sim = (sbert_sim * 0.7) + (tfidf_sim * 0.3)  # SBERT = 70%, TF-IDF = 30%

# 🔹 9. Alegem cele mai relevante etichete pentru fiecare companie
top_n = 3  # Alegem cele mai bune 3 etichete
threshold = 0.2  # Eliminăm etichetele cu scoruri foarte mici

matched_labels = []
for row in final_sim:
    top_indices = np.argsort(-row)[:top_n]  # Sortăm descrescător
    selected_labels = [taxonomy_df["label"].iloc[i] for i in top_indices if row[i] > threshold]
    matched_labels.append(selected_labels if selected_labels else ["Unclassified"])

# 🔹 10. Folosim un model de la Hugging Face pentru clasificare
# Folosim `distilbert-base-uncased`, care este un model performant și accesibil
model_name = "distilbert-base-uncased"  # Modelul pre-antrenat pentru clasificare

# 🔹 11. Încarcă tokenizer și model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 🔹 12. Verificăm dispozitivul disponibil (GPU sau CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Creăm pipeline-ul pentru clasificare text
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device.index if device.type == "cuda" else -1)

def classify_with_huggingface(description, candidate_labels):
    """ Utilizează modelul Hugging Face pentru a rafina clasificarea """
    prompt = f"""
    You are an expert in business classification for the insurance industry.
    Given the company description: "{description}", and the following candidate labels: {candidate_labels},
    return the most accurate labels from the given list.
    """
    
    # Verificăm dacă candidate_labels nu este gol
    if not candidate_labels:
        print(f"Warning: No candidate labels for description: {description[:30]}...")
        return "Unclassified"  # Returnează o etichetă implicită
    
    # Utilizăm modelul pentru clasificare
    try:
        # Tokenizăm descrierea
        inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        
        # Alegem eticheta cu probabilitatea cea mai mare
        predicted_label_idx = outputs.logits.argmax(dim=-1).item()

        # Dacă indexul este în afacerea limitei candidate_labels
        if predicted_label_idx >= len(candidate_labels):
            print(f"Warning: Predicted index {predicted_label_idx} is out of range for candidate labels.")
            predicted_label = candidate_labels[0]  # Returnăm prima etichetă ca fallback
        else:
            predicted_label = candidate_labels[predicted_label_idx]
        
        # Afișăm eticheta aleasă pentru fiecare companie
        print(f"Selected label for description '{description[:50]}...': {predicted_label}")
        
        return predicted_label
    except Exception as e:
        print(f"Error with Hugging Face: {e}")
        return ", ".join(candidate_labels)

# 🔹 13. Aplicăm Hugging Face pentru fiecare companie
final_labels = []
for i, row in company_df.iterrows():
    print(f"Processing company {i+1}/{len(company_df)}...")
    
    # Tentăm să clasificăm folosind Hugging Face
    refined_labels = classify_with_huggingface(row["full_description"], matched_labels[i])
    
    # Dacă "Unclassified", folosim etichetele cele mai bune din similitudinea SBERT și TF-IDF
    if refined_labels == "Unclassified":
        print(f"Warning: Company {i+1} is unclassified. Using fallback based on SBERT & TF-IDF scores.")
        # Alegem etichetele cele mai relevante bazate pe scorurile SBERT și TF-IDF
        top_indices = np.argsort(-final_sim[i])[:top_n]
        refined_labels = [taxonomy_df["label"].iloc[i] for i in top_indices if final_sim[i][i] > threshold]
        refined_labels = refined_labels[0] if refined_labels else "Unclassified"
        
    final_labels.append(refined_labels)
    time.sleep(0.02)  # Respectăm rate limits pentru API

company_df["insurance_label"] = final_labels

# 🔹 14. Salvăm rezultatele
company_df.to_csv("classified_companies_huggingface.csv", index=False)

print("✅ Clasificarea cu Hugging Face a fost finalizată! Rezultatele sunt în 'classified_companies_huggingface.csv'.")
