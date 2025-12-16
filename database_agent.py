import os
import sqlite3
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
load_dotenv()

DB_FILE = "customer_orders.db"

# Remove old database file if it exists
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create Customers table
cursor.execute("""
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    Name TEXT NOT NULL,
    Country TEXT
);
""")

# Create Orders table
cursor.execute("""
CREATE TABLE Orders (
    OrderID INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    Amount REAL,
    OrderDate TEXT
);
""")

# Insert data
customers_data = [
    (1, 'Alice Johnson', 'Canada'),
    (2, 'Bob Smith', 'USA'),
    (3, 'Charlie Brown', 'Canada'),
    (4, 'Diana Prince', 'USA')
]
orders_data = [
    (101, 1, 150.00, '2025-06-15'), (102, 1, 50.00, '2025-07-01'), # Alice
    (103, 3, 200.00, '2025-06-20'), (104, 3, 75.00, '2025-07-10'), # Charlie
    (105, 2, 300.00, '2025-07-12'), (106, 2, 100.00, '2025-07-15'), # Bob
]
cursor.executemany("INSERT INTO Customers VALUES (?, ?, ?)", customers_data)
cursor.executemany("INSERT INTO Orders VALUES (?, ?, ?, ?)", orders_data)
conn.commit()
conn.close()

print(f"Mock database '{DB_FILE}' created successfully.")

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# --- Model Initialization ---
# Gemini is required for the specialized reasoning in the SQL Agent.

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
)

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# WARNING: Ensure the connection details only grant read-only access in production environments.
db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE}")

# The Toolkit exposes SQL specific tools to the Agent
toolkit = SQLDatabaseToolkit(db=db, llm=model)

# Create the final SQL Agent Executor
# verbose=True shows the Agent's reasoning process (CoT)
sql_agent_executor = create_sql_agent(
    llm=model,
    toolkit=toolkit
)


query_sql = "Give me name of customers from Canada?"
print("SQL Agent initialized and connected to the database.")
print("\n" + "="*50)
print(f"Executing: {query_sql}")
print("="*50)

try:
    response = sql_agent_executor.invoke({"input": query_sql})
    print("\n--- FINAL ANSWER ---")
    print(response["output"])
except Exception as e:
    print(f"\nAn error occurred during SQL Agent execution: {e}")

print("\n" + "="*50)
print(f"Executing: {query_sql}")
print("="*50)
