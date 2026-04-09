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
                        user {
                            id
                        }
                    }
                }
            }
        }
        """
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def scrape_all_ratings(product_id):
    all_ratings = []
    offset = 0
    limit = 40

    data = get_product_ratings(product_id, offset=0, limit=limit)
    actions = data["data"]["product"]["otherUsersActions"]
    total = actions["total"] if actions["total"] > 0 else len(actions["userActions"]) * 50

    print(f"Total ratings à récupérer : estimé ~{total} (offset-based)")

    while True:
        data = get_product_ratings(product_id, offset=offset, limit=limit)
        actions = data["data"]["product"]["otherUsersActions"]["userActions"]

        if not actions:
            break

        for action in actions:
            all_ratings.append({
                "product_id": action["productUserInfos"]["productId"],
                "user_id": action["user"]["id"],
                "rating": action["productUserInfos"]["rating"]
            })

        print(f"  offset {offset} → {len(all_ratings)} notes collectées")
        offset += limit
        time.sleep(0.5)

    return all_ratings

def save_to_csv(ratings, filename):
    os.makedirs("data/raw", exist_ok=True)
    filepath = f"data/raw/{filename}"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "user_id", "rating"])
        writer.writeheader()
        writer.writerows(ratings)
    print(f"Sauvegardé : {filepath} ({len(ratings)} lignes)")

if __name__ == "__main__":
    product_id = 81036513
    ratings = scrape_all_ratings(product_id)
    save_to_csv(ratings, f"ratings_{product_id}.csv")
