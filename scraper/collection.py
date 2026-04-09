import requests
import json
import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:149.0) Gecko/20100101 Firefox/149.0",
    "authorization": os.getenv("SC_TOKEN"),
}

def get_build_id():
    r = requests.get("https://www.senscritique.com", headers=headers)
    return re.search(r'/_next/static/([^/]+)/_buildManifest', r.text).group(1)

def get_collection_page(build_id, username, page=1, retries=3):
    url = f"https://www.senscritique.com/_next/data/{build_id}/fr/{username}/collection.json?username={username}&page={page}"
    for attempt in range(retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200 and response.text.strip():
            return response.json()
        print(f"  Retry {attempt+1} page {page} (status {response.status_code})")
        time.sleep(2)
    return None

def extract_product_ids(data, universe_filter=1):
    apollo_state = data["pageProps"]["__APOLLO_STATE__"]
    ids = []
    for key, value in apollo_state.items():
        if key.startswith("Product:"):
            if value.get("universe") == universe_filter:
                ids.append(value["id"])
    return ids

def get_total_pages(data):
    apollo_state = data["pageProps"]["__APOLLO_STATE__"]
    for key, value in apollo_state.items():
        if key.startswith("User:"):
            # 18 produits par page
            total = value.get("stats", {}).get("ratingCount", 0)
            return (total // 18) + 1
    return 1

if __name__ == "__main__":
    username = "Moizi"
    all_ids = set()

    print("Récupération du build ID...")
    build_id = get_build_id()
    print(f"Build ID: {build_id}")

    # Test page 1 pour avoir le total
    data = get_collection_page(build_id, username, page=1)
    total_pages = get_total_pages(data)
    ids = extract_product_ids(data, universe_filter=1)
    all_ids.update(ids)
    print(f"Page 1 → {len(ids)} films, total pages estimé: {total_pages}")

    for page in range(2, total_pages + 1):
        data = get_collection_page(build_id, username, page)
        if data is None:
            print(f"  Page {page} ignorée")
            continue
        ids = extract_product_ids(data, universe_filter=1)
        all_ids.update(ids)
        print(f"Page {page}/{total_pages} → {len(ids)} films, total: {len(all_ids)}")
        time.sleep(0.5)  # augmente un peu la pause

    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/film_ids.json", "w") as f:
        json.dump(list(all_ids), f)
    print(f"\nTotal IDs sauvegardés : {len(all_ids)}")
