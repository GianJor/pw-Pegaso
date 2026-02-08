import random
import csv
import os

random.seed(42)

DEPARTMENTS = ["Housekeeping", "Reception", "F&B"]
SENTIMENTS = ["pos", "neg"]

LEXICON = {
    "Housekeeping": {
        "pos": [
            "Camera pulita e profumata.",
            "Stanza impeccabile e accogliente.",
            "Bagno ordinato e lenzuola fresche.",
            "Servizio pulizie eccellente.",
            "Camera luminosa, fresca e confortevole.",
        ],
        "neg": [
            "Camera sporca e maleodorante.",
            "Bagno con muffa e asciugamani sporchi.",
            "Lenzuola macchiate e pavimento appiccicoso.",
            "Cattivo odore in stanza.",
            "Pulizie scadenti e polvere ovunque.",
        ],
    },
    "Reception": {
        "pos": [
            "Personale gentile e accogliente.",
            "Check-in veloce e professionale.",
            "Staff disponibile e sempre sorridente.",
            "Ottimo servizio alla reception.",
            "Esperienza piacevole al check-out.",
        ],
        "neg": [
            "Attesa lunga al check-in.",
            "Personale scortese e poco disponibile.",
            "Errore nella prenotazione.",
            "Accoglienza fredda e confusione alla reception.",
            "Problemi con il pagamento e poca organizzazione.",
        ],
    },
    "F&B": {
        "pos": [
            "Colazione abbondante e buonissima.",
            "Cibo delizioso e servizio rapido.",
            "Ristorante pulito e accogliente.",
            "Esperienza gastronomica eccellente.",
            "Piatti curati e personale gentile.",
        ],
        "neg": [
            "Colazione scarsa e di bassa qualità.",
            "Cibo freddo e servizio lento.",
            "Piatti sporchi e buffet limitato.",
            "Esperienza culinaria deludente.",
            "Personale distratto al ristorante.",
        ],
    },
}

# --- GENERAZIONE ESTESA ---
rows = []
id_counter = 1

for dept in DEPARTMENTS:
    for sent in SENTIMENTS:
        for phrase in LEXICON[dept][sent]:
            # Genera più variazioni per rendere il dataset robusto
            for _ in range(15):
                noise = random.choice(["!", ".", " davvero", " molto", " assolutamente", " decisamente"])
                variant = phrase.replace(".", noise + ".")
                rows.append({
                    "id": id_counter,
                    "title": phrase.split(",")[0][:30],
                    "body": variant,
                    "department": dept,
                    "sentiment": sent,
                })
                id_counter += 1

random.shuffle(rows)

os.makedirs("data", exist_ok=True)
csv_path = os.path.join("data", "reviews_synth.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "title", "body", "department", "sentiment"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Dataset generato con successo: {csv_path} ({len(rows)} righe)")
