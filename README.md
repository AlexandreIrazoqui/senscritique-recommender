# SensCritique Recommender

A collaborative filtering movie recommendation system built on data scraped from [SensCritique](https://www.senscritique.com/), a French cultural review platform.

---

## Overview

This project builds a **user × film rating matrix** from SensCritique data, then uses collaborative filtering to find correlations between users' tastes and generate personalized movie recommendations.

The pipeline has three stages:
1. **Scraping** - collecting ratings via the SensCritique GraphQL API
2. **Exploration & Preprocessing** - cleaning and filtering the dataset (see the notebook)
3. **Recommendation** - collaborative filtering model (in progress)

---

## Dataset

### Film selection

The ~5,300 films in the dataset were sourced from the collection of **Moizi**, a well-known cinephile user on SensCritique.

This choice was intentional. Common alternatives - scraping from "Top Films" rankings or genre lists - introduce significant biases:
- They over-represent critically acclaimed or popular films
- They systematically exclude niche, older, or less-rated films
- The resulting rating matrix reflects taste for *good* films, not taste in general

Using a single cinephile's broad and eclectic collection gives a more **diverse and representative** film corpus while remaining manageable in size.

### Scale

After scraping ratings for all films in that collection:

| Metric | Value |
|---|---|
| Raw ratings | ~47.5 million |
| Unique users | ~582,000 |
| Unique films | ~5,336 |
| Raw sparsity | ~98.5% |

After filtering (users with ≥ 20 ratings, films with ≥ 50 ratings, and removing the seed user):

| Metric | Value |
|---|---|
| Filtered ratings | ~46 million |
| Unique users | ~207,000 |
| Unique films | ~5,086 |
| Filtered sparsity | ~92.5% |

The threshold of **20 minimum ratings per user** was chosen because it removes a large portion of sparse/inactive users (56% of users) while retaining 96.8% of all ratings - a good trade-off.

---

## Ethical considerations

- All data scraped is **publicly visible** on SensCritique without authentication (ratings are public by default).
- **robots.txt compliance.** SensCritique's `robots.txt` (accessible at `https://www.senscritique.com/robots.txt`) does not restrict general-purpose crawlers from accessing the pages targeted by this scraper.
- No personal data beyond anonymized user IDs and numerical ratings is collected or stored.
- The scraper uses **rate limiting** (configurable delays between requests) to avoid overloading the platform's servers.
- I reached out to SensCritique by email to ask for explicit permission to use their data for this non-commercial research project. I did not receive a response, but no objection was raised either.
- Data is not redistributed, only the scraping code is shared.

---


## Setup

### Requirements

```bash
pip install requests python-dotenv pandas numpy scipy scikit-learn matplotlib
```

### Authentication token

The SensCritique API requires a bearer token. To get yours:

1. Log in to [senscritique.com](https://www.senscritique.com) in your browser
2. Open DevTools → Network tab
3. Trigger any page load and find a request to `https://apollo.senscritique.com/`
4. In the request headers, copy the value of the `authorization` field

Then create a `.env` file at the root of the project:

```
SC_TOKEN=Bearer <your_token_here>
```

> /!\Tokens expire. If the scraper suddenly returns 0 ratings for 10 consecutive films, your token has likely expired - refresh it using the steps above.

---

## Running the scraper

### Step 1 — Collect film IDs

```bash
python scraper/collection.py
```

This fetches all film IDs from the target user's collection and saves them to `data/raw/film_ids.json`.

### Step 2 — Scrape ratings

```bash
python scraper/scrape_all.py
```

This iterates over all film IDs and scrapes ratings from the GraphQL API. Ratings are appended to `data/raw/all_ratings.csv`, and progress is tracked in `data/raw/done_ids.json` , so the scraper can be **safely interrupted and resumed**.

### Running in the background (recommended)

For long scraping sessions, use `tmux` to keep the process alive after closing your terminal:

```bash
tmux new -s scraping
python scraper/scrape_all.py
# Detach with Ctrl+B then D
# Reattach later with:
tmux attach -t scraping
```

---

## Exploration & preprocessing

All data exploration, filtering decisions, and sparse matrix construction are documented in:

```
notebooks/exploration.ipynb
```

It covers:
- Rating distribution
- Notes per user / per film (with log-scale histograms)
- Sparsity analysis
- Threshold selection for filtering
- Construction of the `scipy.sparse` CSR matrix used as model input

---

## Status

- [x] Scraping pipeline
- [x] Data exploration & preprocessing
- [ ] Collaborative filtering model
- [ ] Recommendation API / interface
