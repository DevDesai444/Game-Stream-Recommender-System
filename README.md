# Game Stream Recommender System

**Author:** Dev Desai (`DevDesai444`)

A full-stack recommendation system for Steam that moves from raw player discovery to a browsable recommendation experience. This project combines web crawling, Steam Web API ingestion, PySpark-based recommender modeling, relational persistence, and a Flask web interface to demonstrate how recommendation systems can be built as end-to-end products rather than isolated notebooks.

## Executive Summary

The goal of this project is to recommend games that a Steam user is likely to engage with next by learning from implicit behavioral signals such as ownership and cumulative playtime. The system supports two recommendation modes:

1. A **global popularity recommender** that surfaces the most-played games across the collected player base.
2. A **personalized collaborative filtering recommender** powered by Spark ALS that generates user-specific top-N recommendations.

This repository is intentionally structured as a portfolio-quality systems project:

- `web_crawler/` collects and structures Steam player and game data.
- `recommendation_engine/` transforms behavioral logs into model-ready features and trains recommendation models in Spark.
- `web_ui/` exposes recommendation outputs through a lightweight Flask application.
- `image/` contains supporting architecture and method visuals used throughout the project.

## Why This Project Matters

Game discovery is a difficult ranking problem. Large catalogs, rapidly changing user preferences, and sparse feedback make it hard to connect players with relevant content. Steam is an especially interesting environment because it naturally generates rich implicit signals:

- game ownership
- cumulative playtime
- recent activity
- social graph information
- game metadata

This project treats recommendation as a product engineering problem, not just a machine learning exercise. The system shows how raw platform data can be converted into usable recommendation outputs and then surfaced in an interface that real users can browse.

## Problem Statement

Given a set of Steam users, their owned games, their playtime, and basic game metadata, build a system that can:

1. discover and index candidate users,
2. collect their interaction data from Steam,
3. transform raw JSON into a model-ready user-item matrix,
4. generate both baseline and personalized recommendations,
5. store recommendation results in a database, and
6. display those recommendations in a simple web application.

## Hypothesis

The core hypothesis behind this repository is:

> If a recommendation model is trained on implicit feedback derived from Steam ownership and playtime, then it can recover meaningful preference patterns and produce better personalized game suggestions than a one-size-fits-all popularity ranking.

This hypothesis is supported by three assumptions:

1. **Playtime is a stronger preference signal than ownership alone.** A user may own many games but meaningfully engage with only a subset.
2. **Users with similar interaction histories will prefer similar games.** This is the standard collaborative filtering assumption.
3. **A popularity baseline is useful but insufficient.** It works as a fallback, but personalization should better capture differences between players.

## Architecture Overview

![Architecture](image/architecture.png)

The project follows a staged offline-to-online architecture:

```mermaid
flowchart LR
    A["Steam Community member pages"] --> B["User discovery crawler"]
    C["Steam Web APIs"] --> D["Raw JSON datasets"]
    B --> D
    D --> E["PySpark ETL and joins"]
    E --> F["Popularity baseline"]
    E --> G["ALS collaborative filtering"]
    F --> H["MySQL recommendation tables"]
    G --> H
    H --> I["Flask + SQLAlchemy web UI"]
    I --> J["Global and personalized recommendation pages"]
```

## System Design

### 1. Data Collection Layer

The crawler first discovers active Steam users from Steam community member pages, then resolves canonical Steam IDs from profile pages. After that, it requests structured player data from Steam Web APIs.

Collected entities include:

- player summaries
- owned games
- friend lists
- recently played games
- game metadata

### 2. Storage Layer

Raw data is written as line-delimited JSON. This is a practical choice for a project like this because it is:

- append-friendly,
- easy to inspect manually,
- compatible with Spark JSON readers, and
- simple to move through notebook-based ETL pipelines.

### 3. Processing and Modeling Layer

The Spark notebook loads the JSON data into DataFrames, explodes nested structures, joins users with games, and constructs user-item implicit feedback triples in the form:

```text
(user_idx, appid, playtime_forever)
```

Two recommendation strategies are then produced:

- **Popularity-based ranking**
- **ALS-based collaborative filtering**

### 4. Serving Layer

Recommendation outputs are written into relational tables and served through Flask routes:

- `/` for global recommendations
- `/<user_id>` for personalized recommendations

This separation keeps training and serving loosely coupled, which is closer to how production recommender systems are often deployed.

## Repository Structure

```text
.
├── README.md
├── image/
│   ├── architecture.png
│   ├── cf.png
│   ├── games.png
│   ├── gameRecommendation1.png
│   ├── gameRecommendation2.png
│   ├── mf.png
│   ├── popularGame.png
│   └── steamMember.png
├── recommendation_engine/
│   ├── docker_commands.md
│   └── pyspark_recommendation.ipynb
├── web_crawler/
│   ├── README_steam_API.md
│   ├── game_detail.py
│   ├── steam_crawler.ipynb
│   ├── sample_data/
│   │   ├── game_detail.json
│   │   ├── user_friend_list_sample.json
│   │   ├── user_idx_sample.json
│   │   ├── user_owned_games_sample.json
│   │   ├── user_recently_played_games_sample.json
│   │   └── user_summary_sample.json
│   └── web_crawler.py
└── web_ui/
    ├── app.py
    ├── flaskapp.wsgi
    ├── static/
    └── templates/
```

## Data Pipeline

### Stage 1: User Discovery

The crawler reads Steam community member pages such as:

```text
https://steamcommunity.com/games/steam/members?p=<page_number>
```

It extracts users marked as `online` or `in-game`, then follows their profile links and parses the embedded Steam ID. This is necessary because Steam does not provide a single clean endpoint for broad user discovery.

### Stage 2: User Data Ingestion

For each discovered Steam user, the crawler collects:

- summary profile information,
- owned game library,
- friend network,
- recently played titles.

The primary API families used are:

- `ISteamUser/GetPlayerSummaries`
- `IPlayerService/GetOwnedGames`
- `ISteamUser/GetFriendList`
- `IPlayerService/GetRecentlyPlayedGames`
- `ISteamApps/GetAppList`
- `store.steampowered.com/api/appdetails`

### Stage 3: Game Metadata Ingestion

`game_detail.py` first retrieves the Steam app catalog and then fetches app details from the Steam Store API. In the sample workflow included in this repository, it writes a subset of game details for downstream enrichment.

### Stage 4: Data Transformation

The Spark notebook performs several essential preprocessing steps:

- loading JSON into DataFrames,
- removing corrupt rows,
- exploding nested arrays such as `games` and `friends`,
- joining user and game entities,
- converting large Steam IDs to compact integer user indices,
- assembling model-ready interaction tuples.

### Stage 5: Recommendation Generation

The recommendation engine creates:

1. **Global recommendations** by aggregating total `playtime_forever` across users.
2. **Personalized recommendations** using implicit-feedback ALS in Spark MLlib.

### Stage 6: Persistence and Serving

Recommendation outputs are written to:

- `global_recommend`
- `final_recommend`

The Flask app reads those tables using SQLAlchemy and renders the results in HTML templates.

## Recommendation Methodology

### Popularity Baseline

The baseline recommender ranks games by total accumulated playtime across the dataset. This is useful because it:

- provides a fast sanity check,
- establishes a strong non-personalized baseline,
- gives coverage even when user-specific data is sparse.

This method answers the question:

> What games are broadly engaging across the observed player population?

### Collaborative Filtering with ALS

The personalized recommender uses Spark's implicit-feedback ALS implementation:

```python
als_model = ALS.trainImplicit(training_rdd, 10)
```

The model factorizes the user-item interaction space into low-dimensional latent vectors. Instead of requiring explicit ratings, it uses behavioral intensity derived from playtime. This is a natural fit for Steam, where users rarely submit structured ratings at the same scale they generate play history.

Conceptually:

```mermaid
flowchart TD
    A["User-game playtime matrix"] --> B["Matrix factorization with ALS"]
    B --> C["Latent user factors"]
    B --> D["Latent game factors"]
    C --> E["Predicted affinity scores"]
    D --> E
    E --> F["Top-N personalized recommendations"]
```

### Why Implicit Feedback

The project uses implicit signals because they are abundant, behaviorally grounded, and realistic for large-scale systems. Playtime is not a perfect proxy for satisfaction, but it is often a stronger signal than binary ownership.

## Sample Data Snapshot

The repository includes sample data extracted from the crawler workflow:

- `163` indexed users
- `163` player summary records
- `163` owned-game records
- `163` friend-list records
- `163` recently-played records
- `7` sample game detail records

These files are located under `web_crawler/sample_data/` and serve as lightweight artifacts for demonstrating the pipeline without requiring a fresh crawl.

## Key Implementation Details

### Crawler

The crawler is implemented in legacy Python style and uses:

- `requests` for HTTP requests,
- `BeautifulSoup` for HTML parsing,
- `urllib` for profile retrieval,
- `json` for structured output.

It discovers users from Steam community pages, resolves a Steam ID from profile content, and serializes multiple API responses to JSON line files.

### Recommendation Engine

The notebook uses:

- `SparkContext`
- `SparkSession`
- `HiveContext`
- `pyspark.mllib.recommendation.ALS`

The design is notebook-centric, which makes the system easy to explain and iterate on for experimentation, while still showing the main ideas used in production recommendation pipelines.

### Web Application

The Flask app maps directly onto the persisted recommendation tables:

- `global_recommend` for global ranking output
- `final_recommend` for personalized output

The UI renders game cards using Steam header images and links each result back to the Steam app page.

## Database Schema

The web application expects two logical tables:

### `global_recommend`

| Column | Purpose |
|---|---|
| `rank` | ordering score or popularity ordering |
| `name` | game title |
| `header_image` | Steam-hosted header image |
| `steam_appid` | unique Steam application id |

### `final_recommend`

| Column | Purpose |
|---|---|
| `user_id` | original Steam user id |
| `rank` | recommendation order for a user |
| `name` | recommended game title |
| `header_image` | Steam-hosted header image |
| `steam_appid` | recommended game id |

## Results and Evidence

This repository includes several concrete outputs that demonstrate the pipeline is functioning end to end.

### Global recommendation output

The Spark notebook shows globally popular titles ranked by total observed playtime. Example output includes:

- `Counter-Strike: Global Offensive`
- `Garry's Mod`

The notebook output also shows large aggregate playtime values, indicating that the popularity baseline is being computed over real interaction intensity rather than binary counts alone.

### Personalized recommendation output

The ALS recommender generates ranked per-user recommendations. Sample notebook output includes personalized results such as:

- `Team Fortress 2`
- `PLAYERUNKNOWN'S BATTLEGROUNDS`
- `Dota 2`

This demonstrates that the system is not merely replaying one global list for every user, but producing user-linked recommendation rows.

### User Interface artifacts

The repository includes screenshots and architecture imagery that help validate the project as a working system rather than only a notebook:

- ![Steam member discovery](image/steamMember.png)
- ![Popularity recommendation](image/popularGame.png)
- ![Personalized recommendation 1](image/gameRecommendation1.png)
- ![Personalized recommendation 2](image/gameRecommendation2.png)
- ![Collaborative filtering diagram](image/cf.png)
- ![Matrix factorization diagram](image/mf.png)

## Technical Strengths

This project is especially strong as a systems portfolio piece because it demonstrates:

- end-to-end ownership across data ingestion, modeling, persistence, and UI,
- practical use of implicit-feedback recommendation methods,
- integration between offline ML workflows and online serving,
- comfort with semi-structured data and ETL design,
- application of distributed data processing with Spark.

## Limitations

This repository is a strong prototype, but it also has clear limitations that are useful to acknowledge honestly:

1. The crawler is written in legacy Python 2 style.
2. Sample data size is limited and does not represent production-scale Steam traffic.
3. The project does not currently include offline ranking metrics such as Precision@K, Recall@K, MAP@K, or NDCG@K.
4. Cold-start handling for new users and new games is limited.
5. Social graph data is collected but not yet fully integrated into a ranking model.
6. The notebook pipeline would need orchestration, validation, and packaging before production deployment.

## Future Improvements

If this system were extended further, the next high-value improvements would be:

1. Port the crawler and notebooks fully to Python 3.
2. Add reproducible ETL jobs outside the notebook.
3. Evaluate recommendation quality with ranking metrics.
4. Blend collaborative filtering with content and social features.
5. Add a true API layer for recommendation serving.
6. Containerize the entire stack for easier local deployment.
7. Introduce scheduled retraining and data freshness checks.
8. Expand the UI with user search, fallbacks, and error handling for missing recommendation rows.

## How To Run

### Prerequisites

- Python 3 for the Flask app
- Python 2.7 compatibility for the legacy crawler scripts
- Java and Spark for the notebook pipeline
- optional MySQL or MariaDB for database-backed serving

### Environment Variables

```bash
export STEAM_API_KEY="your_steam_api_key"
export STEAM_RECOMMENDATION_DB_URI="mysql://user:password@host:3306/steam_recommendation"
```

If `STEAM_RECOMMENDATION_DB_URI` is not set, the app falls back to:

```text
sqlite:///steam_recommendation.db
```

### Run the Crawler

```bash
python web_crawler/web_crawler.py
python web_crawler/game_detail.py
```

### Run the Recommendation Pipeline

Open and execute:

```text
recommendation_engine/pyspark_recommendation.ipynb
```

The notebook generates intermediate recommendation artifacts and final recommendation tables for serving.

### Run the Web Application

```bash
cd web_ui
python app.py
```

Then open:

- `http://127.0.0.1:5000/`
- `http://127.0.0.1:5000/<steam_user_id>`

## Deployment Pattern

The notebook documents a deployment pattern built around:

- Spark for offline computation,
- MySQL on AWS RDS for persisted recommendation tables,
- Flask served through WSGI on EC2.

That design is useful because model generation and recommendation serving can scale independently.

## Portfolio Positioning

For hiring managers, this project demonstrates the ability to:

- design an applied machine learning system from first principles,
- work across data engineering and application engineering boundaries,
- translate model outputs into product-facing experiences,
- explain architectural choices clearly,
- make tradeoffs explicit between prototype speed and production readiness.

## License

No license file is currently included in this repository. Add one before public redistribution or external reuse.
