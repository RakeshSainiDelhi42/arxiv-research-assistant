import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

with open("schema.sql", encoding="utf-8") as f:
    schema = f.read()

with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
    with conn.cursor() as cur:
        cur.execute(schema)

print("Schema applied")