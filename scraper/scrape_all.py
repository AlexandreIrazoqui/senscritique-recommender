import json
import os
import time
from ratings import scrape_all_ratings, save_to_csv

# Charger les IDs
with open("data/raw/film_ids.json") as f:
    film_ids = json.load(f)

# Charger les IDs déjà scrapés
done_file = "data/raw/done_ids.json"
done_ids = set(json.load(open(done_file))) if os.path.exists(done_file) else set()

remaining = [fid for fid in film_ids if fid not in done_ids]
print(f"{len(remaining)} films restants")

for i, film_id in enumerate(remaining):
    try:
        ratings = scrape_all_ratings(film_id)
        if ratings:
            save_to_csv(ratings, "all_ratings.csv", mode="a")
        done_ids.add(film_id)
        with open(done_file, "w") as f:
            json.dump(list(done_ids), f)
        print(f"[{i+1}/{len(remaining)}] Film {film_id} → {len(ratings)} notes")
    except Exception as e:
        print(f"[{i+1}/{len(remaining)}] Film {film_id} → ERREUR : {e}")
    time.sleep(0.5)

print("Scraping terminé !")
