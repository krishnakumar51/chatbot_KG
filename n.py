from neo4j import GraphDatabase, exceptions
import time

# Define the connection retry logic function
def connect_with_retry(uri, user, password, max_retries=5, delay=3):
    for attempt in range(max_retries):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                print("Connected to Neo4j successfully.")
                return session
        except (exceptions.ServiceUnavailable, exceptions.ConnectionError) as e:
            print(f"Connection failed on attempt {attempt + 1}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise RuntimeError(f"Failed to connect to Neo4j after {max_retries} attempts.")

# Simulate the function in a loop for 100 times
failures = 0
for i in range(100):
    try:
        print(f"Attempt {i + 1}:")
        connect_with_retry("neo4j+s://4cfe3812.databases.neo4j.io", "neo4j", "password")
    except RuntimeError as e:
        print(f"Error on attempt {i + 1}: {e}")
        failures += 1