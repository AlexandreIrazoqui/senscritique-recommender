import requests
import time

url = "https://apollo.senscritique.com/"
headers = {
    "content-type": "application/json",
    "Origin": "https://www.senscritique.com",
    "Referer": "https://www.senscritique.com/",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:149.0) Gecko/20100101 Firefox/149.0",
}

QUERY = """
query UserCollectionFilms($userId: Int!, $universe: String, $limit: Int, $offset: Int) {
    user(id: $userId) {
        collection(universe: $universe, limit: $limit, offset: $offset) {
            products {
                id
                title
                rating
                otherUserInfos {
                    rating
                }
            }
        }
    }
}
"""

def get_user_seen_films(user_id, limit=100):
    seen = []
    offset = 0
    while True:
        payload = {
            "operationName": "UserCollectionFilms",
            "variables": {"userId": user_id, "universe": "movie",
                          "limit": limit, "offset": offset},
            "query": QUERY,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        data = response.json()
        if "errors" in data:
            print("Erreur GraphQL :", data["errors"])
            break
        products = data["data"]["user"]["collection"]["products"]
        if not products:
            break
        for p in products:
            info = p.get("otherUserInfos") or {}
            seen.append({
                "id":           p["id"],
                "title":        p["title"],
                "rating":       info.get("rating"),   # note PERSO (entier) -> celle qu'on garde
                "rating_moyen": p.get("rating"),       # moyenne du film -> juste pour vérifier
            })
        offset += limit
        time.sleep(0.5)
    return seen


if __name__ == "__main__":
    result = get_user_seen_films(1695743)
    print("Nombre de films :", len(result))
    for r in result[:10]:
        print("perso :", r["rating"], " | moyenne :", r["rating_moyen"], " |", r["title"])
