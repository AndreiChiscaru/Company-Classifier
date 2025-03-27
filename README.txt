README - Clasificator de Companii pentru Asigurari

Descriere:

Asta e un script care ia o lista de companii si incearca sa le clasifice automat pe baza unei taxonomii predefinite.
Practic, citim descrierea fiecarei companii, vedem la ce etichete din taxonomie se potriveste cel mai bine si o incadram unde trebuie.

Metoda folosita:
Preprocesare date - Concatenam mai multe campuri relevante intr-un singur text.
Embeddings cu SBERT - Folosim modelul all-MiniLM-L6-v2 ca sa generam reprezentari numerice ale textului.
TF-IDF - Calculam si scoruri bazate pe frecventa cuvintelor, ca sa avem si o abordare mai simpla pe langa embeddings.
Combinare scoruri - Facem un mix intre scorurile din SBERT si TF-IDF (70% SBERT, 30% TF-IDF).
Alegerea celor mai bune etichete - Pastram top 3 etichete relevante pentru fiecare companie, daca trec de un prag minim.
Imbunatatire cu Hugging Face - Folosim distilbert-base-uncased ca sa rafinam si mai bine clasificarea.
Fallback - Daca Hugging Face nu ne da o clasificare buna, ne intoarcem la scorurile din pasii anteriori.
Salvarea rezultatelor - Scriem etichetele finale intr-un fisier CSV.

Posibile imbunatatiri:
Optimizare model: Un model mai bun (gen sentence-t5) ar putea sa faca embeddings mai precise.
Mai multe features: Poate luam si alte campuri in calcul, gen descrierea completa a companiei daca e disponibila.
Binarizarea etichetelor: In loc de un singur label, putem incerca sa clasificam multi-label (adica o companie poate avea mai multe etichete simultan, nu doar una principala).
Fine-tuning: Putem antrena un model propriu cu date de asigurari ca sa fie mai adaptat la domeniu.

Cum il rulezi?
Pune fisierele ml_insurance_challenge.csv si insurance_taxonomy.xlsx in acelasi director cu scriptul.
Instaleaza dependintele:
pip install openai pandas numpy sentence-transformers scikit-learn transformers torch openpyxl
Ruleaza scriptul:
python clasificator.py
Dupa ce ruleaza, gasesti rezultatele in classified_companies_huggingface.csv.