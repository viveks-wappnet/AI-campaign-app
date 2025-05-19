"""
This module handles database configuration and operations.
"""

import os
import json
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def get_db_connection():
    """
    Creates and returns a connection to the PostgreSQL database.
    
    Returns:
        A connection object to the database
    """
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def store_script_in_db(campaign_idea: str, script: list):
    """
    Stores the generated script into the PostgreSQL database.
    
    Args:
        campaign_idea: The original user prompt/campaign idea
        script: The generated script as a list of scene dictionaries
    """
    connection = None
    try:
        # Connect to the PostgreSQL database
        connection = get_db_connection()
        cursor = connection.cursor()

        # Insert the script into the scripts table
        insert_query = """
        INSERT INTO scripts (user_prompt, script)
        VALUES (%s, %s)
        """
        cursor.execute(insert_query, (campaign_idea, json.dumps(script)))
        connection.commit()
        print("Script inserted successfully!")

    except Exception as error:
        print(f"Error while inserting script: {error}")
    finally:
        # Close the database connection
        if connection:
            cursor.close()
            connection.close()
