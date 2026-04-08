"""
MongoDB Handler Module
Provides database operations for storing and retrieving transaction data.
Handles connections, CRUD operations, and queries with graceful error handling.
"""

import sys
import os
from typing import List, Dict, Optional
from datetime import datetime

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging


class MongoDBHandler:
    """
    MongoDB Handler for AML Fraud Detection System
    Manages connection and CRUD operations for transaction data
    """
    
    # Class-level connection cache
    _connection = None
    _database = None
    _collection = None
    _is_connected = False
    
    # Default configuration - Use environment variable for cloud, fallback to local
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = "fraud_db"
    COLLECTION_NAME = "transactions"
    
    def __init__(self, mongo_uri: str = None, db_name: str = None, collection_name: str = None):
        """
        Initialize MongoDB Handler
        
        Args:
            mongo_uri: MongoDB connection string (default: localhost)
            db_name: Database name (default: fraud_db)
            collection_name: Collection name (default: transactions)
        """
        self.mongo_uri = mongo_uri or self.MONGO_URI
        self.db_name = db_name or self.DB_NAME
        self.collection_name = collection_name or self.COLLECTION_NAME
        
        # Try to connect on initialization
        self._connect()
    
    def _connect(self) -> bool:
        """
        Establish MongoDB connection with automatic retry and recovery.
        
        Production-Safe Features:
        - Reuses existing connection (no reconnect overhead)
        - Automatic connection health check
        - Lazy pymongo import (optional dependency)
        - Non-blocking timeouts
        - Index creation on startup
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # ===================================================================
            # 1. CONNECTION REUSE: Avoid reconnecting if already connected
            # ===================================================================
            if MongoDBHandler._is_connected and MongoDBHandler._collection:
                # Verify connection is still healthy
                try:
                    MongoDBHandler._connection.admin.command('ping')
                    logging.debug("MongoDB connection active and healthy")
                    return True
                except Exception as ping_error:
                    logging.warning(f"MongoDB connection health check failed: {ping_error}")
                    MongoDBHandler._is_connected = False
                    # Continue to reconnect below
            
            # ===================================================================
            # 2. LAZY IMPORT: Only import pymongo if needed
            # ===================================================================
            try:
                import pymongo
                from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
            except ImportError:
                logging.warning("pymongo not installed. MongoDB features disabled.")
                logging.info("Install with: pip install pymongo")
                MongoDBHandler._is_connected = False
                return False
            
            # ===================================================================
            # 3. CREATE CONNECTION: Non-blocking with timeout
            # ===================================================================
            logging.debug(f"Connecting to MongoDB: {self.mongo_uri}")
            
            MongoDBHandler._connection = pymongo.MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=3000,      # Fast fail on unavailable server
                connectTimeoutMS=3000,
                socketTimeoutMS=5000,
                maxPoolSize=10,                      # Connection pool
                minPoolSize=0,                       # No idle connections
                retryWrites=False,                   # Avoid locked writes during unavailability
                unicode_decode_error_handler='ignore'
            )
            
            # ===================================================================
            # 4. VERIFY CONNECTION: Ping with timeout
            # ===================================================================
            try:
                MongoDBHandler._connection.admin.command('ping')
                logging.info(f"✓ MongoDB connection successful: {self.db_name}/{self.collection_name}")
            except (ServerSelectionTimeoutError, ConnectionFailure) as e:
                logging.warning(f"MongoDB not available: {str(e)}")
                MongoDBHandler._is_connected = False
                return False
            
            # ===================================================================
            # 5. GET DATABASE & COLLECTION
            # ===================================================================
            MongoDBHandler._database = MongoDBHandler._connection[self.db_name]
            MongoDBHandler._collection = MongoDBHandler._database[self.collection_name]
            
            # ===================================================================
            # 6. CREATE INDEXES: Speed up common queries
            # ===================================================================
            try:
                MongoDBHandler._collection.create_index("risk_score")
                MongoDBHandler._collection.create_index("timestamp")
                MongoDBHandler._collection.create_index([("prediction", 1), ("timestamp", -1)])
                logging.debug("MongoDB indexes created/verified")
            except Exception as idx_error:
                logging.warning(f"Failed to create indexes: {idx_error}")
                # Continue anyway - queries will be slower but still work
            
            MongoDBHandler._is_connected = True
            return True
            
        except Exception as e:
            logging.warning(f"MongoDB connection failed: {str(e)}")
            MongoDBHandler._is_connected = False
            return False
    
    
    @staticmethod
    def is_connected() -> bool:
        """
        Check if MongoDB is connected and operational
        
        Returns:
            bool: Connection status
        """
        return MongoDBHandler._is_connected
    
    def insert_transaction(self, transaction_data: Dict) -> Optional[str]:
        """
        Insert transaction with production-safe error handling.
        
        Features:
        - Non-blocking insert (fails fast)
        - Graceful degradation (app continues if MongoDB down)
        - Timestamp validation and auto-add
        - Connection recovery attempt
        - Comprehensive error classification
        
        Args:
            transaction_data: Dictionary containing transaction details
        
        Returns:
            str: Inserted document ID if successful, None if unavailable
        """
        # =====================================================================
        # 1. PRE-INSERTION CHECKS
        # =====================================================================
        if not MongoDBHandler._is_connected:
            logging.debug("MongoDB not available - cannot insert transaction")
            return None
        
        if not MongoDBHandler._collection:
            logging.error("MongoDB collection not initialized")
            return None
        
        # =====================================================================
        # 2. TIMESTAMP VALIDATION & AUTO-ADD
        # =====================================================================
        try:
            if 'timestamp' not in transaction_data:
                transaction_data['timestamp'] = datetime.utcnow()
            elif not isinstance(transaction_data['timestamp'], datetime):
                transaction_data['timestamp'] = datetime.utcnow()
        except Exception as ts_error:
            logging.warning(f"Timestamp handling error: {ts_error}")
            transaction_data['timestamp'] = datetime.utcnow()
        
        # =====================================================================
        # 3. INSERT WITH ERROR CATEGORIZATION
        # =====================================================================
        try:
            result = MongoDBHandler._collection.insert_one(transaction_data)
            
            if result.inserted_id:
                inserted_id = str(result.inserted_id)
                logging.info(f"✓ Transaction inserted: {inserted_id}")
                return inserted_id
            else:
                logging.error("Insert returned no document ID")
                return None
            
        except Exception as insert_error:
            error_str = str(insert_error)
            error_type = type(insert_error).__name__
            
            # ===================================================================
            # CONNECTION ERRORS: Attempt recovery on next call
            # ===================================================================
            if any(x in error_type for x in ['Connection', 'ServerSelection', 'Network']):
                logging.warning(f"Connection error during insert: {error_str}")
                MongoDBHandler._is_connected = False
                # Try to reconnect next time
                return None
            
            # ===================================================================
            # DUPLICATE KEY ERRORS: Log and continue (data already exists)
            # ===================================================================
            elif 'duplicate' in error_str.lower() or 'E11000' in error_str:
                logging.debug(f"Duplicate key error: {error_str}")
                return None
            
            # ===================================================================
            # TIMEOUT ERRORS: Non-blocking insert was too slow
            # ===================================================================
            elif 'timeout' in error_str.lower():
                logging.warning(f"MongoDB insert timeout: {error_str}")
                return None
            
            # ===================================================================
            # OTHER ERRORS: Log and continue
            # ===================================================================
            else:
                logging.warning(f"Failed to insert transaction ({error_type}): {error_str}")
                return None
    
    
    def get_all_transactions(self, limit: int = None) -> List[Dict]:
        """
        Retrieve all transactions from MongoDB
        
        Args:
            limit: Maximum number of transactions to return (None = all)
        
        Returns:
            List of transaction documents, empty list if MongoDB unavailable
            
        Example:
            >>> handler = MongoDBHandler()
            >>> transactions = handler.get_all_transactions(limit=100)
            >>> print(f"Retrieved {len(transactions)} transactions")
        """
        try:
            if not MongoDBHandler._is_connected or not MongoDBHandler._collection:
                logging.debug("MongoDB not available - returning empty list")
                return []
            
            # Query with optional limit
            query = MongoDBHandler._collection.find()
            
            if limit:
                query = query.limit(limit)
            
            # Convert to list and clean ObjectId representation
            transactions = []
            for doc in query:
                doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                transactions.append(doc)
            
            logging.info(f"Retrieved {len(transactions)} transactions from database")
            return transactions
            
        except Exception as e:
            logging.warning(f"Failed to retrieve transactions: {str(e)}")
            return []
    
    def find_high_risk_transactions(self, min_score: int = 50, limit: int = None) -> List[Dict]:
        """
        Find transactions with risk_score >= min_score
        
        Args:
            min_score: Minimum risk score threshold (default: 50)
            limit: Maximum number of results to return (None = all)
        
        Returns:
            List of high-risk transaction documents, sorted by risk_score descending
            
        Example:
            >>> handler = MongoDBHandler()
            >>> risky = handler.find_high_risk_transactions(min_score=70)
            >>> print(f"Found {len(risky)} high-risk transactions")
            >>> for tx in risky:
            ...     print(f"Amount: {tx['amount_paid']}, Risk: {tx['risk_score']}")
        """
        try:
            if not MongoDBHandler._is_connected or not MongoDBHandler._collection:
                logging.debug("MongoDB not available - returning empty list")
                return []
            
            # Query for high-risk transactions
            query_filter = {"risk_score": {"$gte": min_score}}
            query = MongoDBHandler._collection.find(query_filter).sort("risk_score", -1)
            
            if limit:
                query = query.limit(limit)
            
            # Convert to list and clean ObjectId representation
            high_risk_txns = []
            for doc in query:
                doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                high_risk_txns.append(doc)
            
            logging.info(f"Found {len(high_risk_txns)} transactions with risk_score >= {min_score}")
            return high_risk_txns
            
        except Exception as e:
            logging.warning(f"Failed to find high-risk transactions: {str(e)}")
            return []
    
    def get_transaction_by_id(self, transaction_id: str) -> Optional[Dict]:
        """
        Retrieve a specific transaction by ID
        
        Args:
            transaction_id: MongoDB ObjectId as string
        
        Returns:
            Transaction document or None if not found
        """
        try:
            if not MongoDBHandler._is_connected or not MongoDBHandler._collection:
                logging.debug("MongoDB not available - cannot retrieve by ID")
                return None
            
            from bson import ObjectId
            
            # Query by ID
            doc = MongoDBHandler._collection.find_one({"_id": ObjectId(transaction_id)})
            
            if doc:
                doc['_id'] = str(doc['_id'])
                logging.info(f"Retrieved transaction: {transaction_id}")
                return doc
            else:
                logging.warning(f"Transaction not found: {transaction_id}")
                return None
                
        except Exception as e:
            logging.warning(f"Failed to retrieve transaction by ID: {str(e)}")
            return None
    
    def get_statistics(self) -> Optional[Dict]:
        """
        Get database statistics (count, average risk score, etc.)
        
        Returns:
            Dictionary with statistics or None if MongoDB unavailable
            
        Example:
            >>> handler = MongoDBHandler()
            >>> stats = handler.get_statistics()
            >>> print(f"Total transactions: {stats['total_count']}")
            >>> print(f"Average risk score: {stats['avg_risk_score']:.2f}")
        """
        try:
            if not MongoDBHandler._is_connected or not MongoDBHandler._collection:
                logging.debug("MongoDB not available - cannot compute statistics")
                return None
            
            total_count = MongoDBHandler._collection.count_documents({})
            high_risk_count = MongoDBHandler._collection.count_documents({"risk_score": {"$gte": 70}})
            medium_risk_count = MongoDBHandler._collection.count_documents(
                {"risk_score": {"$gte": 50, "$lt": 70}}
            )
            
            # Calculate average risk score
            pipeline = [
                {"$group": {
                    "_id": None,
                    "avg_risk": {"$avg": "$risk_score"},
                    "max_risk": {"$max": "$risk_score"},
                    "min_risk": {"$min": "$risk_score"}
                }}
            ]
            
            result = list(MongoDBHandler._collection.aggregate(pipeline))
            stats = result[0] if result else {}
            
            return {
                'total_count': total_count,
                'high_risk_count': high_risk_count,
                'medium_risk_count': medium_risk_count,
                'low_risk_count': total_count - high_risk_count - medium_risk_count,
                'avg_risk_score': stats.get('avg_risk', 0),
                'max_risk_score': stats.get('max_risk', 0),
                'min_risk_score': stats.get('min_risk', 0)
            }
            
        except Exception as e:
            logging.warning(f"Failed to compute statistics: {str(e)}")
            return None
    
    def delete_by_id(self, transaction_id: str) -> bool:
        """
        Delete a transaction by ID
        
        Args:
            transaction_id: MongoDB ObjectId as string
        
        Returns:
            bool: True if deleted, False otherwise
        """
        try:
            if not MongoDBHandler._is_connected or not MongoDBHandler._collection:
                logging.debug("MongoDB not available - cannot delete")
                return False
            
            from bson import ObjectId
            
            result = MongoDBHandler._collection.delete_one({"_id": ObjectId(transaction_id)})
            
            if result.deleted_count > 0:
                logging.info(f"Transaction deleted: {transaction_id}")
                return True
            else:
                logging.warning(f"No transaction found to delete: {transaction_id}")
                return False
                
        except Exception as e:
            logging.warning(f"Failed to delete transaction: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all transactions from collection (use with caution!)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not MongoDBHandler._is_connected or not MongoDBHandler._collection:
                logging.debug("MongoDB not available - cannot clear")
                return False
            
            result = MongoDBHandler._collection.delete_many({})
            logging.warning(f"Cleared {result.deleted_count} transactions from collection")
            return True
            
        except Exception as e:
            logging.warning(f"Failed to clear collection: {str(e)}")
            return False
    
    @staticmethod
    def close_connection():
        """
        Close MongoDB connection
        """
        try:
            if MongoDBHandler._connection:
                MongoDBHandler._connection.close()
                MongoDBHandler._is_connected = False
                MongoDBHandler._connection = None
                MongoDBHandler._database = None
                MongoDBHandler._collection = None
                logging.info("MongoDB connection closed")
        except Exception as e:
            logging.warning(f"Error closing MongoDB connection: {str(e)}")


# Convenience function for quick access
def get_mongo_handler(mongo_uri: str = None, db_name: str = None, collection_name: str = None) -> MongoDBHandler:
    """
    Get MongoDB Handler instance
    
    Args:
        mongo_uri: MongoDB connection string
        db_name: Database name
        collection_name: Collection name
    
    Returns:
        MongoDBHandler instance
        
    Example:
        >>> handler = get_mongo_handler()
        >>> if handler.is_connected():
        ...     txns = handler.get_all_transactions(limit=10)
    """
    return MongoDBHandler(mongo_uri, db_name, collection_name)


# Example usage and tests
if __name__ == "__main__":
    try:
        print("🔄 Initializing MongoDB Handler...")
        handler = MongoDBHandler()
        
        if handler.is_connected():
            print("✓ MongoDB connection successful!")
            
            # Example: Insert a transaction
            sample_transaction = {
                "from_bank": 100,
                "to_bank": 200,
                "amount_paid": 5000,
                "currency": "USD",
                "risk_score": 65,
                "risk_level": "Medium"
            }
            
            doc_id = handler.insert_transaction(sample_transaction)
            print(f"✓ Inserted transaction: {doc_id}")
            
            # Get all transactions
            all_txns = handler.get_all_transactions(limit=5)
            print(f"✓ All transactions: {len(all_txns)}")
            
            # Find high-risk transactions
            high_risk = handler.find_high_risk_transactions(min_score=60, limit=5)
            print(f"✓ High-risk transactions: {len(high_risk)}")
            
            # Get statistics
            stats = handler.get_statistics()
            if stats:
                print(f"✓ Statistics: {stats}")
        else:
            print("⚠ MongoDB not available - application will work without persistence")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        MongoDBHandler.close_connection()
        print("Done!")
