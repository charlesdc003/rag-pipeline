import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "rag",
    "user": "rag",
    "password": "rag",
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def setup_database():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(768),
            metadata JSONB
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database ready.")


if __name__ == "__main__":
    setup_database()