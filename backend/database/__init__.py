from .oracle import OracleDatabaseManager
from .mysql import MySQLDatabaseManager
from .postgres import PostgreSQLDatabaseManager

def get_db_manager(db_type: str):
    db_type = db_type.lower().strip()
    if db_type == "oracle":
        return OracleDatabaseManager()
    elif db_type == "mysql":
        return MySQLDatabaseManager()
    elif db_type == "postgresql":
        return PostgreSQLDatabaseManager()
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
