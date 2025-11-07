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

LOAD DATA LOCAL INPATH 'friends.json' OVERWRITE INTO TABLE friends;

-- load data for game details
CREATE TABLE IF NOT EXISTS game (
	id INT,
	name STRING,
	type STRING,
	is_free BOOLEAN,
	required_age TINYINT,
	detailed_description STRING,
	short_description STRING,
	about STRING,
	supported_language STRING,
	header_image STRING,
	website STRING,
	platforms STRUCT<windows:BOOLEAN,mac:BOOLEAN, linux:BOOLEAN>,
	pc_requirements STRUCT< minimum:STRING>,
	mac_requirements STRUCT< minimum:STRING>,
	linux_requirements STRUCT< minimum:STRING>,
	developers ARRAY<STRING>,
	publishers ARRAY<STRING>,
	price STRUCT<currency:STRING, initial: INT, final: INT, discount_p
ercent: INT>,
	categories ARRAY<STRUCT<id:INT, description:STRING> >,
	metacritic STRUCT<score:INT,url:STRING>,
	genres ARRAY<STRUCT<id:STRING, description:STRING> >,
	screenshots ARRAY<STRUCT<id:INT, path_thumbnail:STRING, path_full:STRING>>,
	recommendations INT,
	achievements INT,
	release_date STRUCT<coming_soon:BOOLEAN, release_date:STRING>,
	support_info STRUCT<url:STRING, email:STRING>,
	background STRING
)
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
STORED AS TEXTFILE;

LOAD DATA LOCAL INPATH 'game.json' OVERWRITE INTO TABLE game;
```

## 3 Recommendation Engine

### 3.1 Theory and Algorithm

**Collaborative Filtering (CF)** is a method of making automatic predictions about the interests of a user by learning its preferences (or taste) based on information of his engagements with a set of available items, along with other users’ engagements with the same set of items. in other words, CF assumes that, **if a person A has the same opinion as person B on some set of issues X={x1,x2,…}, then A is more likely to have B‘s opinion on a new issue y than to have the opinion of any other person that doesn’t agree with A on X**.


**Co-clustering (or Biclustering)** is a term in data mining that relates to a simultaneous clustering of the rows and columns of a matrix. Where classical clustering methods assume that a membership of an object (in a group of objects) depends solely on its similarity to other objects of the same type (same entity type), co-clustering can be seen as a method of co-grouping two types of entities simultaneously, based on similarity of their pairwise interactions.

**Matrix Factorization**

<p style="text-align:justify;">One of the most popular algorithms to solve co-clustering problems (and specifically for collaborative recommender systems) is called Matrix Factorization (MF). In its simplest form, it assumes a matrix <img src="https://s0.wp.com/latex.php?latex=%7B%7BR%7D%5Cin%7BR%5E%7Bm+%5Ctimes+n%7D%7D%7D&#038;bg=ffffff&#038;fg=5e5e5e&#038;s=0" alt="{{R}&#92;in{R^{m &#92;times n}}}" title="{{R}&#92;in{R^{m &#92;times n}}}" class="latex" /> of ratings given by <em>m</em><em>users</em> to <em>n</em><em>items</em>. Applying this technique on <em>R</em> will end up factorizing <em>R</em> into two matrices <img src="https://s0.wp.com/latex.php?latex=%7B%7BU%7D%5Cin%7BR%5E%7Bm+%5Ctimes+k%7D%7D%7D&#038;bg=ffffff&#038;fg=5e5e5e&#038;s=0" alt="{{U}&#92;in{R^{m &#92;times k}}}" title="{{U}&#92;in{R^{m &#92;times k}}}" class="latex" /> and <img src="https://s0.wp.com/latex.php?latex=%7B%7BP%7D%5Cin%7BR%5E%7Bn+%5Ctimes+k%7D%7D%7D&#038;bg=ffffff&#038;fg=5e5e5e&#038;s=0" alt="{{P}&#92;in{R^{n &#92;times k}}}" title="{{P}&#92;in{R^{n &#92;times k}}}" class="latex" /> such that <img src="https://s0.wp.com/latex.php?latex=R+%5Capprox+U+%5Ctimes+P&#038;bg=ffffff&#038;fg=5e5e5e&#038;s=0" alt="R &#92;approx U &#92;times P" title="R &#92;approx U &#92;times P" class="latex" /> (their multiplication approximates <em>R</em>).
</p>

<p>
<img src="image/mf.png"  style="width: 100%;"/>
</p>

<p style="text-align:justify;">So back to linear algebra, MF is a form of optimization process that aims to approximate the original matrix <em>R</em> with the two matrices <em>U</em> and <em>P</em>, such that it minimizes the following cost function:</p>
<p style="text-align:center;"><img src="https://s0.wp.com/latex.php?latex=J+%3D+%7C%7CR+-+U+%5Ctimes+%7BP%5ET%7D%7C%7C_2+%2B+%5Clambda+%5Cleft%28%7C%7CU%7C%7C_2+%2B+%7C%7CP%7C%7C_2+%5Cright%29&#038;bg=ffffff&#038;fg=5e5e5e&#038;s=0" alt="J = ||R - U &#92;times {P^T}||_2 + &#92;lambda &#92;left(||U||_2 + ||P||_2 &#92;right)" title="J = ||R - U &#92;times {P^T}||_2 + &#92;lambda &#92;left(||U||_2 + ||P||_2 &#92;right)" class="latex" /></p>
<p style="text-align:justify;">The first term in this cost function is the Mean Square Error (MSE) distance measure between the original rating matrix <em>R</em> and its approximation <img src="https://s0.wp.com/latex.php?latex=%7BU+%5Ctimes+%7BP%5ET%7D%7D+&#038;bg=ffffff&#038;fg=5e5e5e&#038;s=0" alt="{U &#92;times {P^T}} " title="{U &#92;times {P^T}} " class="latex" />. The second term is called a &#8220;regularization term&#8221; and is added to govern a generalized solution (to prevent overfitting to some local noisy effects on ratings).</p>

**Alternating Least Squares**

The **collaborive filtering** problem can be formulated as a learning problem in which we are given the ratings that users have given certain items and are tasked with predicting their ratings for the rest of the items. Formally, if there are n users and m items, we are given an n × m matrix R in which the (u, i)th entry is r_ui – the rating for item i by user u. **Matrix R has many missing entries indicating unobserved ratings, and our task is to estimate these unobserved ratings**.

A popular approach for this is matrix factorization, where **Alternative Least Square (ALS)** algorithm renders its power. ALS can not only be implemented in single machine, but also in distributed clusters, or even in streaming.

Before we really start to play around with the algorithm, it's highly recommended to read through the Pyspark collaborative filtering documentations([https://spark.apache.org/docs/latest/ml-collaborative-filtering.html](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html))


### 3.2 Get Spark Environment via Docker

1. install docker on your computer

    https://docs.docker.com/

2. Get docker image

    docker pull dalverse/all-spark-notebook

4. Prepare a folder to share with docker

    for example, on my computer is
    /Users/hongzhaozhu/workfiles/dockerShare

3. Start Docker Container

    docker run -v /Users/hongzhaozhu/workfiles/dockerShare:/home/dal/work -d -P dalverse/all-spark-notebook
    docker start containerID

4. Check docker running status

  - To see all images: docker images
  - To see all runnning containers: docker ps -a
  - docker logs containerID


5. Login container and view IP addr

  - docker exec -it containerID ip addr
  - docker exec -it --user root containerID bash

6. Log in jupyter notebook with spark

    use "docker logs containerID" to see the address mapping

### 3.3 Global Popular Games

```python
df_user_owned_games = hiveCtx.read.json(sample_user_owned_games)
df_user_owned_games.printSchema()
df_user_owned_games.registerTempTable("user_owned_games")

# find the top 10 games which have longest total played hours
df_global_popular_games = \
hiveCtx.sql("SELECT b.game_id, SUM(b.playtime_forever) AS play_time FROM \
                (SELECT played_games['appid'] AS game_id, played_games['playtime_forever'] AS playtime_forever \
                FROM (SELECT EXPLODE(games) AS played_games FROM user_owned_games) a) b \
                GROUP BY game_id ORDER BY play_time DESC LIMIT 10")
df_global_popular_games.registerTempTable('popular_games')

# find same app id in popular_games and game_detail
# total played_hours is defined as rank
df_global_popular_games = hiveCtx.sql("SELECT b.name AS name, a.play_time AS rank, b.steam_appid, b.header_image FROM \
    popular_games a, game_detail b WHERE a.game_id = b.steam_appid ORDER BY rank DESC")
df_global_popular_games.show()

# root
#  |-- game_count: long (nullable = true)
#  |-- games: array (nullable = true)
#  |    |-- element: struct (containsNull = true)
#  |    |    |-- appid: long (nullable = true)
#  |    |    |-- playtime_2weeks: long (nullable = true)
#  |    |    |-- playtime_forever: long (nullable = true)
#  |-- steamid: string (nullable = true)
#
# +--------------------+--------+-----------+--------------------+
# |                name|    rank|steam_appid|        header_image|
# +--------------------+--------+-----------+--------------------+
# |Counter-Strike: G...|14355867|        730|http://cdn.akamai...|
# |         Garry's Mod| 4485082|       4000|http://cdn.akamai...|
# |      Counter-Strike| 4178037|         10|http://cdn.akamai...|
# |  Grand Theft Auto V| 3904596|     271590|http://cdn.akamai...|
# |       Left 4 Dead 2| 3677466|        550|http://cdn.akamai...|
# |Counter-Strike: S...| 3616174|        240|http://cdn.akamai...|
# |The Elder Scrolls...| 2900266|      72850|http://cdn.akamai...|
# |            Warframe| 2597596|     230410|http://cdn.akamai...|
# |            Terraria| 2548415|     105600|http://cdn.akamai...|
# |       Killing Floor| 2371501|       1250|http://cdn.akamai...|
# +--------------------+--------+-----------+--------------------+
```

### 3.4 Collaborative Filtering Recommendation System

```python
df_user_recent_games = hiveCtx.read.json(sample_user_recent_games)

# df_user_recent_games.printSchema()
df_user_recent_games.registerTempTable("user_recent_games")
df_valid_user_recent_games = hiveCtx.sql("SELECT * FROM user_recent_games where total_count != 0")
```

Convert the Steam ID to index to avoid overflow in the recommendation algorithm. This is achieved by joining tables.

For example:
```json
{"user_idx": 0, "user_id": "76561197970565175"}
```
We map 76561197970565175 to 0

```python
df_user_idx = hiveCtx.read.json(sample_user_idx)
df_user_idx.registerTempTable('user_idx')
df_valid_user_recent_games = hiveCtx.sql("SELECT b.user_idx, a.games FROM user_recent_games a \
                                          JOIN user_idx b ON b.user_id = a.steamid WHERE a.total_count != 0")

# map and filter out the games whose playtime is 0
training_rdd = df_valid_user_recent_games.rdd.flatMapValues(lambda x : x) \
                .map(lambda (x, y) : (x, y.appid, y.playtime_forever)) \
                .filter(lambda (x, y, z) : z > 0)
training_rdd.collect()
# [(0, 24740, 216),
#  (0, 223100, 99),
#  (0, 403640, 9),
#  (0, 590780, 1),
#  (0, 363970, 510),
#  (1, 39210, 10521),
#  (1, 570, 53685),
#  (1, 440, 123990),
#  (2, 578080, 468),
#  (2, 440, 29658),
#  ...
#  (3, 493340, 68)}

als_model = ALS.trainImplicit(training_rdd, 10)

# print out 10 recommendeds product for user of index 0
result_rating = als_model.recommendProducts(0,10)

try_df_result=sc.createDataFrame(result_rating)
print try_df_result.sort(desc("rating")).show()
# +----+-------+-------------------+
# |user|product|             rating|
# +----+-------+-------------------+
# |   0| 363970| 0.3046938568409334|
# |   0| 433850|0.15175814718740938|
# |   0|  72850| 0.1421794704660013|
# |   0|    753|0.13219302752311712|
# |   0| 402840|0.12326413470293149|
# |   0|  21690|0.12156766375401792|
# |   0| 306130| 0.1198095384178326|
# |   0| 221680|0.10631534097162214|
# |   0| 234330|0.10348192421626112|
# |   0| 230410|0.10201294175900974|
# +----+-------+-------------------+
```
Join the Steam user ID table and game_detail table to form the final results

```python
df_recommend_result.registerTempTable('recommend_result')
df_final_recommend_result = hiveCtx.sql("SELECT DISTINCT b.user_id, a.rank, c.name, c.header_image, c.steam_appid \
                                        FROM recommend_result a, user_idx b, game_detail c \
                                        WHERE a.user_idx = b.user_idx AND a.game_id = c.steam_appid \
                                        ORDER BY b.user_id, a.rank")
df_final_recommend_result.show(20)
# +-----------------+----+--------------------+--------------------+-----------+
# |          user_id|rank|                name|        header_image|steam_appid|
# +-----------------+----+--------------------+--------------------+-----------+
# |76561197960292467|   1|     Team Fortress 2|http://cdn.akamai...|        440|
# |76561197960292467|   2|PLAYERUNKNOWN'S B...|http://cdn.akamai...|     578080|
# |76561197960292467|   3|Sid Meier's Civil...|http://cdn.akamai...|       8930|
# |76561197960292467|   4|            Warframe|http://cdn.akamai...|     230410|
# |76561197960292467|   5|    Dead by Daylight|http://cdn.akamai...|     381210|
# |76561197960292467|   6|           Stellaris|http://cdn.akamai...|     281990|
# |76561197960292467|   7|           Fallout 4|http://cdn.akamai...|     377160|
# |76561197960292467|   8|       Assetto Corsa|http://cdn.akamai...|     244210|
# |76561197960292467|   9|The Elder Scrolls...|http://cdn.akamai...|     306130|
# |76561197960292467|  10|             XCOM® 2|http://cdn.akamai...|     268500|
# |76561197960315617|   1|  Grand Theft Auto V|http://cdn.akamai...|     271590|
# |76561197960315617|   2|  Starpoint Gemini 2|http://cdn.akamai...|     236150|
# |76561197960315617|   3|        Awesomenauts|http://cdn.akamai...|     204300|
# |76561197960315617|   4|Don't Starve Toge...|http://cdn.akamai...|     322330|
# |76561197960315617|   6|      Rocket League®|http://cdn.akamai...|     252950|
# |76561197960315617|   9|     Team Fortress 2|http://cdn.akamai...|        440|
# |76561197960315617|  10|           Fallout 4|http://cdn.akamai...|     377160|
# |76561197960384723|   1|              Dota 2|http://cdn.akamai...|        570|
# |76561197960384723|   2|         Garry's Mod|http://cdn.akamai...|       4000|
# |76561197960384723|   3|     Team Fortress 2|http://cdn.akamai...|        440|
# +-----------------+----+--------------------+--------------------+-----------+
# only showing top 20 rows
```

### 3.5 Store the Recommendation Results to AWS RDS
Download MySQL JDBC [connector](https://dev.mysql.com/downloads/connector/j/) class first, and copy it to $SPARK_HOME/jars, e.g., /Library/spark-2.1.1-bin-hadoop2.7/jars

A good reference for connecting to AWS MySQL DB through JDBC can be found [here](https://medium.com/modernnerd-code/connecting-to-mysql-db-on-aws-ec2-with-jdbc-for-java-91dba3003abb) and [here](https://docs.databricks.com/spark/latest/data-sources/sql-databases.html#writing-data-to-jdbc).

First we upload the popularity-based recommendation results to database. We specify the database name to be "test1", and the table name to be "global_recommend".

```python
# define jdbc properties
url = 'jdbc:mysql://steam-recommendation.chcqngehr8cs.us-west-2.rds.amazonaws.com:3306'
mode = 'overwrite'
properties = {
    "user": "huntingzhu",
    "password": "xxxxxxxxxxx",
    "driver": 'com.mysql.jdbc.Driver'
}

df_global_popular_games.show()
df_global_popular_games.write.jdbc(url=url, table="steam_recommendation.global_recommend", mode=mode, properties=properties)
# +--------------------+--------+-----------+--------------------+
# |                name|    rank|steam_appid|        header_image|
# +--------------------+--------+-----------+--------------------+
# |Counter-Strike: G...|14355867|        730|http://cdn.akamai...|
# |         Garry's Mod| 4485082|       4000|http://cdn.akamai...|
# |      Counter-Strike| 4178037|         10|http://cdn.akamai...|
# |  Grand Theft Auto V| 3904596|     271590|http://cdn.akamai...|
# |       Left 4 Dead 2| 3677466|        550|http://cdn.akamai...|
# |Counter-Strike: S...| 3616174|        240|http://cdn.akamai...|
# |The Elder Scrolls...| 2900266|      72850|http://cdn.akamai...|
# |            Warframe| 2597596|     230410|http://cdn.akamai...|
# |            Terraria| 2548415|     105600|http://cdn.akamai...|
# |       Killing Floor| 2371501|       1250|http://cdn.akamai...|
# +--------------------+--------+-----------+--------------------+
```

## 4 Web UI

In this phase, we are going to implement a Web UI to present the recommendation results. The Web framework we are using is called Flaskr, which provides a simple interface for dynamically generating responses to web requests.
Before you start, be sure to read through the Flaskr tutorial in here http://flask.pocoo.org/docs/0.12/tutorial/ . That can help you gain more understanding in what Flaskr is and how Flaskr is organized.

### 4.1 Config DataBase Connection
First of all, I need to install and config all the dependencies that I need on the EC2 instance, like apache2, python environment, flask, SQLAlchemy...

Then I have to modify the configuration of MySQL DB on AWS RDS to allow other IPs to remote connect to the DB.

The final thing is deploying my web UI code.

```python
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
import re

app = Flask(__name__)
DB_URI = 'mysql://XXXX'
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class recommendation_global(db.Model):
    __tablename__ = 'global_recommend'
    rank = db.Column('rank', db.Integer, primary_key = True)
    name = db.Column('name', db.Text)
    header_image = db.Column('header_image', db.Text)
    steam_appid = db.Column('steam_appid', db.Integer)

    def __init__(self, rank, name, header_image, steam_appid):
        self.rank = rank
        self.name = name
        self.header_image = header_image
        self.steam_appid = steam_appid

class recommendation(db.Model):
    __tablename__ = 'final_recommend'
    user_id = db.Column('user_id', db.Text, primary_key = True)
    rank = db.Column('rank', db.Integer, primary_key = True)
    name = db.Column('name', db.Text)
    header_image = db.Column('header_image', db.Text)
    steam_appid = db.Column('steam_appid', db.Integer)

    def __init__(self, user_id, rank, name, header_image, steam_appid):
        self.user_id = user_id
        self.rank = rank
        self.name = name
        self.header_image = header_image
        self.steam_appid = steam_appid

@app.route('/')
def global_recommendation():
    global_recom = recommendation_global.query.order_by(recommendation_global.rank).all()
    return render_template("index.html", global_recom=global_recom)

@app.route('/<user_id>')
def user_recommendation(user_id):
    user_recom = recommendation.query.filter_by(user_id=user_id).order_by(recommendation.rank).all()
    return render_template("user.html", user_recom = user_recom)

if __name__ == '__main__':
    app.run()
```

<p align="center">
<img src="image/gameRecommendation2.png"  style="width: 600px;"/>
</p>
