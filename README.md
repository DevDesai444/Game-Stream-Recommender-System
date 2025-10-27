# Game Stream Recommendation System

A full data-to-product pipeline for recommending Steam games using:

1. Web crawling and Steam Web APIs for data collection
2. Spark + ALS for recommendation modeling
3. Flask + SQLAlchemy for user-facing delivery

This repository includes crawler scripts, sample datasets, Spark notebooks, web UI code, and architecture images.

## Creator / Author / Developer

**DevDesai444** (project owner and maintainer)

## Project Goal

Build a practical recommendation system that can:

1. Collect user and game interaction data from Steam
2. Generate both global and personalized recommendations
3. Serve results in a lightweight web application

## Architecture

![Architecture](image/architecture.png)

High-level flow:

1. Crawl Steam member pages to discover active users
2. Fetch user/game data via Steam Web APIs
3. Store raw JSON datasets
4. Transform and join data in Spark SQL / Hive-style workflows
5. Train an implicit-feedback ALS model
6. Write recommendation outputs to MySQL (AWS RDS pattern)
7. Render recommendations in Flask routes

## What Is Used, How It Is Used, and Why It Is Used

| Layer | Technology | How it is used | Why it is used |
|---|---|---|---|
| Data collection | `requests`, `urllib` | Calls Steam API endpoints and profile URLs | Reliable HTTP access to both API and HTML sources |
| HTML parsing | `BeautifulSoup`, `re` | Parses Steam member pages to find online/in-game users and profile URLs | Steam does not expose all discovery paths via one API |
| Concurrency | `threading`, `Queue` | Parallel user data fetch and thread-safe writer queue | Improves crawl throughput and protects file writes |
| Raw storage | JSON lines files | Writes one JSON object per line for users/games/summaries | Stream-friendly, simple to process in Spark/Hive |
| Data processing | PySpark, Spark SQL, HiveContext | Reads JSON, explodes arrays, joins datasets, filters noisy rows | Handles wide schemas and large-scale transforms |
| Recommendation model | `pyspark.mllib.recommendation.ALS.trainImplicit` | Trains implicit collaborative filtering model from playtime signals | Good baseline for user-item implicit feedback |
| Popularity baseline | Spark SQL aggregation | Sums playtime by app id for global top games | Simple, strong baseline and fallback recommender |
| Result persistence | JDBC to MySQL | Writes `global_recommend` and `final_recommend` tables | Decouples offline model pipeline from online serving |
| Web service | Flask + Flask-SQLAlchemy | Exposes routes for global and per-user recommendations | Fast to build and easy to deploy |
| Serving entrypoint | WSGI (`flaskapp.wsgi`) | Imports Flask app for Apache/mod_wsgi style deployment | Production-friendly deployment integration |
| Frontend | Jinja2 templates + static CSS/JS | Renders card-style recommendation pages with Steam links | Lightweight UI with no SPA overhead |
| Infra pattern | AWS RDS + AWS EC2 (documented workflow) | Stores recommendation tables and hosts Flask app | Clear separation of compute, storage, and serving |

## Repository Map

```text
.
├── README.md
├── image/
│   ├── architecture.png
│   ├── games.png
│   ├── cf.png
│   ├── mf.png
│   ├── popularGame.png
│   ├── gameRecommendation1.png
│   ├── gameRecommendation2.png
│   └── steamMember.png
├── web_crawler/
│   ├── web_crawler.py
│   ├── game_detail.py
│   ├── steam_crawler.ipynb
│   ├── README_steam_API.md
│   └── sample_data/
├── recommendation_engine/
│   ├── pyspark_recommendation.ipynb
│   └── docker_commands.md
└── web_ui/
    ├── app.py
    ├── flaskapp.wsgi
    ├── templates/
    └── static/
```

## End-to-End Data Pipeline

### 1. User discovery

- Source: Steam members pages like `https://steamcommunity.com/games/steam/members?p=<page>`
- Logic: filter users with `online` or `in-game` class markers
- Output: `user_idx_sample.json` where each long Steam ID gets a compact integer index

Why index users:

- Steam IDs are large 64-bit values
- Indexing improves downstream algorithm compatibility and makes joins cheaper

### 2. Steam API ingestion

Collected entities:

- Player summaries
- Owned games
- Friend lists
- Recently played games
- Game metadata (appid, name, images, descriptions, tags, etc.)

Primary endpoints used:

- `ISteamUser/GetPlayerSummaries`
- `IPlayerService/GetOwnedGames`
- `ISteamUser/GetFriendList`
- `IPlayerService/GetRecentlyPlayedGames`
- `ISteamApps/GetAppList`
- `store.steampowered.com/api/appdetails`

### 3. Spark transformation

Core operations in `pyspark_recommendation.ipynb`:

- Parse JSON into DataFrames
- Drop corrupt rows (`_corrupt_record`)
- Explode nested arrays (`games`, `friends`)
- Join on `user_idx` and `steam_appid`
- Build training tuples: `(user_idx, appid, playtime_forever)`

### 4. Recommendation generation

Two recommenders are produced:

1. **Global popularity recommender**
   - Sum `playtime_forever` for each game across users
   - Return top-N globally most played titles

2. **Collaborative filtering recommender**
   - Train implicit ALS on user-game-playtime tuples
   - Generate top-N products per user index
   - Map user index back to original Steam ID
   - Enrich with game name/header image for UI

### 5. Serving layer

Flask routes:

- `/` -> global recommendations (`global_recommend`)
- `/<user_id>` -> personalized recommendations (`final_recommend`)

Templates render each recommendation card with:

- Game header image
- Click-through link to Steam app page

## Data Contracts (Sample Files)

Under `web_crawler/sample_data/`:

- `user_idx_sample.json`: `{ "user_idx": <int>, "user_id": "<steamid>" }`
- `user_summary_sample.json`: user profile snapshots
- `user_owned_games_sample.json`: full owned library + playtime
- `user_friend_list_sample.json`: friend graph adjacency list
- `user_recently_played_games_sample.json`: recent play activity
- `game_detail.json`: Steam app metadata objects

These files are line-delimited JSON and intended for pipeline prototyping and notebook demos.

## Local Development Setup

### Prerequisites

- Python 3.x for web UI (`web_ui/app.py`)
- Python 2.7 compatibility for legacy crawler scripts (`web_crawler/*.py`)
- Java + Spark (for recommendation notebook)
- Optional: MySQL/MariaDB instance for UI-backed results

### 1) Configure environment

```bash
export STEAM_API_KEY="your_steam_api_key"
export STEAM_RECOMMENDATION_DB_URI="mysql://user:password@host:3306/steam_recommendation"
```

If `STEAM_RECOMMENDATION_DB_URI` is not set, the Flask app defaults to local SQLite:

`sqlite:///steam_recommendation.db`

### 2) Run crawler scripts (legacy Python 2 style)

```bash
python web_crawler/web_crawler.py
python web_crawler/game_detail.py
```

### 3) Run notebook pipeline

Open and execute:

- `recommendation_engine/pyspark_recommendation.ipynb`

Key outputs:

- Intermediate recommendation JSON
- Final recommendation dataset for DB ingestion

### 4) Run web UI

```bash
cd web_ui
python app.py
```

Open:

- `http://127.0.0.1:5000/` for global recommendations
- `http://127.0.0.1:5000/<steam_user_id>` for personalized recommendations

## Deployment Pattern (Documented)

The project docs and notebook demonstrate a classic split deployment:

1. Offline compute in Spark for training/inference
2. MySQL on AWS RDS for persisted recommendation tables
3. Flask app on AWS EC2 behind WSGI for serving

This pattern is useful because retraining and serving can scale independently.

## Limitations and Engineering Notes

- Crawler scripts are legacy Python 2 style (`print` statements, `xrange`)
- Steam endpoints and HTML structure may change over time
- API rate limits and private profiles reduce data completeness
- ALS quality depends heavily on interaction sparsity and data freshness
- Notebooks include prototype-style code and should be productionized for scheduled jobs

## Security and Configuration Notes

- API keys and database credentials should be environment variables
- Do not hardcode secrets in source files
- Rotate any previously exposed credentials before production use

## Recommended Improvements

1. Port crawler and notebooks fully to Python 3
2. Add unit tests for parsing and data contracts
3. Add model evaluation metrics (MAP@K, NDCG@K, Recall@K)
4. Add ETL orchestration (Airflow or cron + idempotent jobs)
5. Add API layer (JSON endpoints) for frontend/consumer apps
6. Add containerized local stack (`docker-compose`) for DB + app + Spark

## License

No license file is currently included in this repository. Add one before public reuse/distribution.
