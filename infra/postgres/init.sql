-- Initialise pgvector extension and core tables on first boot.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS games (
    game_idx           INTEGER     PRIMARY KEY,
    steam_appid        BIGINT      NOT NULL UNIQUE,
    name               TEXT        NOT NULL,
    header_image       TEXT,
    short_description  TEXT
);

CREATE TABLE IF NOT EXISTS game_embeddings (
    game_idx     INTEGER PRIMARY KEY REFERENCES games(game_idx),
    steam_appid  BIGINT  NOT NULL,
    embedding    vector(64) NOT NULL
);

CREATE INDEX IF NOT EXISTS game_embeddings_cosine_idx
    ON game_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS user_recommendations (
    user_id        VARCHAR(32) NOT NULL,
    rank           INTEGER     NOT NULL,
    steam_appid    BIGINT      NOT NULL,
    score          DOUBLE PRECISION NOT NULL,
    model_version  VARCHAR(64) NOT NULL,
    PRIMARY KEY (user_id, rank)
);

CREATE INDEX IF NOT EXISTS user_recommendations_user_idx ON user_recommendations (user_id);
