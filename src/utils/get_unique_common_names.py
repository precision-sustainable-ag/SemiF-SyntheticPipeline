from pymongo import MongoClient

def get_unique_common_names(mongo_uri, db_name, collection_name):
    """
    Retrieve all unique `category.common_name` values from the MongoDB collection.

    Args:
        mongo_uri (str): MongoDB connection URI.
        db_name (str): Name of the MongoDB database.
        collection_name (str): Name of the MongoDB collection.

    Returns:
        List[str]: A list of unique common names.
    """
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Use the `distinct` method to get unique `category.common_name` values
    unique_common_names = collection.distinct("category.common_name")

    # Close the connection
    client.close()

    return unique_common_names

# Example usage
mongo_uri = "mongodb://localhost:27017/"  # Update with your MongoDB URI
db_name = "agir_synth"           # Replace with your database name
collection_name = "cutouts" # Replace with your collection name

unique_common_names = get_unique_common_names(mongo_uri, db_name, collection_name)
print("Unique category.common_name values:")
for name in sorted(unique_common_names):
    print(name)

