import requests
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

url = "https://apollo.senscritique.com/"
headers = {
    "authorization": os.getenv("SC_TOKEN"),
    "content-type": "application/json",
    "Origin": "https://www.senscritique.com",
    "Referer": "https://www.senscritique.com/",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:149.0) Gecko/20100101 Firefox/149.0"
}

def get_collection_page(username, offset=0, limit=18):
    payload = {
        "operationName": "UserCollection",
        "variables": {
            "action": None,
            "categoryId": None,
            "gameSystemId": None,
            "genreId": None,
            "keywords": "",
            "limit": limit,
            "offset": offset,
            "order": "LAST_ACTION_DESC",
            "universe": "movie",
            "username": username,
            "yearDateDone": None,
            "isCollection": True
        },
        "query": """
        query UserCollection($action: ProductAction, $categoryId: Int, $gameSystemId: Int, $genreId: Int, $keywords: String, $limit: Int, $offset: Int, $order: CollectionSort, $universe: String, $username: String!, $yearDateDone: Int, $isCollection: Boolean) {
            user(username: $username) {
                collection(
                    action: $action
                    categoryId: $categoryId
                    gameSystemId: $gameSystemId
                    genreId: $genreId
                    keywords: $keywords
                    limit: $limit
                    offset: $offset
                    order: $order
                    universe: $universe
                    yearDateDone: $yearDateDone
                    isCollection: $isCollection
                ) {
                    total
                    products {
                        id
                        title
                    }
                }
            }
        }
        """
    }
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200 and response.text.strip():
                return response.json()
            print(f"  Retry {attempt+1} (status {response.status_code})")
        except Exception as e:
            print(f"  Retry {attempt+1} ({e})")
        time.sleep(3)
    return None

if __name__ == "__main__":
    username = "Moizi"
    all_ids = []
    offset = 0
    limit = 18
    total = None

    while True:
        data = get_collection_page(username, offset=offset, limit=limit)
        if data is None:
            print(f"  Page offset={offset} échouée, arrêt.")
            break

        collection = data["data"]["user"]["collection"]

        if total is None:
            total = collection["total"]
            print(f"Total films dans la collection : {total}")

        products = collection["products"]
        if not products:
            break

        ids = [p["id"] for p in products]
        all_ids.extend(ids)
        print(f"offset={offset} → {len(ids)} films, total récupéré : {len(all_ids)}/{total}")

        if len(all_ids) >= total:
            break

        offset += limit
        time.sleep(0.5)

    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/film_ids.json", "w") as f:
        json.dump(list(set(all_ids)), f)
    print(f"\nTotal IDs sauvegardés : {len(set(all_ids))}")
