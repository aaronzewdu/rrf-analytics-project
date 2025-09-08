import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        database="rrf_analytics",
        user="postgres",
        password="postgres"
    )
    print("postgresql connection success")
    conn.close()
except Exception as e:
    print(f"Error: {e}")