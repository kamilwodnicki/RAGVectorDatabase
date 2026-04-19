from pymongo import MongoClient
from pymongo.collection import Collection
from src.config import MONGODB_HOST, MONGODB_PORT, MONGODB_DB, MONGODB_PARENTS_COLLECTION

_client: MongoClient | None = None


def get_mongo_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(host=MONGODB_HOST, port=MONGODB_PORT)
    return _client


def get_parents_collection() -> Collection:
    return get_mongo_client()[MONGODB_DB][MONGODB_PARENTS_COLLECTION]
