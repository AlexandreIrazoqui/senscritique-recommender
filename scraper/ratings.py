import requests
import os
import csv
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

def get_product_ratings(product_id, offset=0, limit=40):
    payload = {
        "operationName": "ProductUserInfos",
        "variables": {
            "action": "RATING",
            "id": product_id,
            "limit": limit,
            "offset": offset
        },
        "query": """
        query ProductUserInfos($action: ProductAction, $id: Int!, $limit: Int, $offset: Int) {
            product(id: $id) {
                id
                otherUsersActions(action: $action, offset: $offset, limit: $limit) {
                    total
                    userActions {
                        productUserInfos {
                            productId
                            rating
                            userId
                        }
                        user { id }
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
            print(f"  Retry {attempt+1} (status {response.status_code}, empty={not response.text.strip()})")
        except Exception as e:
            print(f"  Retry {attempt+1} ({e})")
        time.sleep(3)
    return None

def scrape_all_ratings(product_id):
    all_ratings = []
    offset = 0
    limit = 40

    while True:
        data = get_product_ratings(product_id, offset=offset, limit=limit)
        if data is None:
            break
        actions = data["data"]["product"]["otherUsersActions"]["userActions"]
        if not actions:
            break
        for action in actions:
            all_ratings.append({
                "product_id": action["productUserInfos"]["productId"],
                "user_id": action["user"]["id"],
                "rating": action["productUserInfos"]["rating"]
            })
        offset += limit
        time.sleep(1)

    return all_ratings

def save_to_csv(ratings, filename, mode="w"):
    os.makedirs("data/raw", exist_ok=True)
    filepath = f"data/raw/{filename}"
    write_header = mode == "w" or not os.path.exists(filepath)
    with open(filepath, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "user_id", "rating"])
        if write_header:
            writer.writeheader()
        writer.writerows(ratings)
