import requests
import time

url = "https://apollo.senscritique.com/"

headers = {
    "content-type": "application/json",
    "Origin": "https://www.senscritique.com",
    "Referer": "https://www.senscritique.com/",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:149.0) Gecko/20100101 Firefox/149.0"
}

def get_user_seen_films(user_id, limit=100):
    seen = []
    offset = 0
    while True:
        payload = {
            "operationName": "UserCollectionFilms",
            "variables": {
                "userId": user_id,
                "universe": "movie",
                "limit": limit,
                "offset": offset
            },
            "query": """
            query UserCollectionFilms($userId: Int!, $universe: String, $limit: Int, $offset: Int) {
                user(id: $userId) {
                    collection(universe: $universe, limit: $limit, offset: $offset) {
                        products { id title rating }
                    }
                }
            }
            """
        }
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print("Status :", response.status_code)
        print("Body :", response.text[:200])
        products = response.json()["data"]["user"]["collection"]["products"]
        if not products:
            break
        seen.extend([{"id": p["id"], "title": p["title"], "rating": p["rating"]} for p in products])
        offset += limit
        time.sleep(0.5)
    return seen

result = get_user_seen_films(1695743)
print("Nombre de films :", len(result))
print(result[:10])
