import pymysql
import pandas as pd
from typing import Dict, List

class MySQLDatabaseManager:
    def __init__(self):
        self.connection = None
        self.db_type = "mysql"
        
    def connect(self, host: str, port: str, database: str, username: str, password: str) -> bool:
        try:
            self.connection = pymysql.connect(
                host=host,
                port=int(port),
                database=database,
                user=username,
                password=password,
                charset='utf8mb4'
            )
            return True
        except Exception as e:
            print(f"MySQL connection failed: {str(e)}")
            return False
            
    def get_all_tables(self) -> List[str]:
        if not self.connection:
            return []
        cursor = self.connection.cursor()
        try:
            cursor.execute("SHOW TABLES")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error fetching MySQL tables: {str(e)}")
            return []
        finally:
            cursor.close()
            
    def get_table_schema(self, table_names: List[str] = None) -> Dict:
        if not self.connection:
            return {}
        schema_info = {}
        cursor = self.connection.cursor()
        try:
            if not table_names:
                table_names = self.get_all_tables()
            for table in table_names:
                cursor.execute(f"DESCRIBE {table}")
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        'name': row[0],
                        'type': row[1],
                        'nullable': row[2],
                        'key': row[3],
                        'default': row[4],
                        'extra': row[5],
                        'length': None,
                        'precision': None,
                        'scale': None
                    })
                
                cursor.execute("""
                    SELECT 
                        COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = %s AND REFERENCED_TABLE_NAME IS NOT NULL
                """, (table,))
                
                foreign_keys = {}
                for fk_row in cursor.fetchall():
                    foreign_keys[fk_row[0]] = {
                        'references_table': fk_row[1],
                        'references_column': fk_row[2]
                    }
                
                schema_info[table] = {
                    'columns': columns,
                    'foreign_keys': foreign_keys
                }
        except Exception as e:
            print(f"Error fetching MySQL schema: {str(e)}")
        finally:
            cursor.close()
        return schema_info
        
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        if not self.connection:
            return pd.DataFrame()
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} LIMIT 1")
            if cursor.fetchone()[0] == 0:
                return pd.DataFrame()
            
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            processed_data = []
            for row in data:
                processed_row = []
                for item in row:
                    if item is None:
                        processed_row.append(None)
                    elif hasattr(item, 'read'):
                        try:
                            processed_row.append(str(item.read())[:100])
                        except:
                            processed_row.append("[BLOB/TEXT]")
                    elif hasattr(item, 'date'):
                        try:
                            processed_row.append(item.strftime('%Y-%m-%d %H:%M:%S'))
                        except:
                            processed_row.append(str(item))
                    else:
                        processed_row.append(str(item) if not pd.isna(item) else None)
                processed_data.append(processed_row)
            
            df = pd.DataFrame(processed_data, columns=columns)
            return df
        except Exception as e:
            print(f"Could not fetch sample data from {table_name}: {str(e)}")
            return pd.DataFrame()
        finally:
            cursor.close()
            
    def execute_query(self, sql: str) -> pd.DataFrame:
        if not self.connection:
            raise Exception("No database connection")
        sql = sql.strip().rstrip(';')
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql)
            if sql.strip().upper().startswith('SELECT'):
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                processed_data = []
                for row in data:
                    processed_row = []
                    for item in row:
                        if item is None:
                            processed_row.append(None)
                        elif hasattr(item, 'read'):
                            try:
                                processed_row.append(str(item.read())[:1000])
                            except:
                                processed_row.append("[BLOB/TEXT]")
                        elif hasattr(item, 'date'):
                            try:
                                processed_row.append(item.strftime('%Y-%m-%d %H:%M:%S'))
                            except:
                                processed_row.append(str(item))
                        elif isinstance(item, (int, float)):
                            processed_row.append(item if not pd.isna(item) else None)
                        else:
                            processed_row.append(str(item) if item is not None else None)
                    processed_data.append(processed_row)
                df = pd.DataFrame(processed_data, columns=columns)
                for col in df.columns:
                    df[col] = df[col].where(pd.notnull(df[col]), None)
                return df
            else:
                self.connection.commit()
                return pd.DataFrame({'Result': [f"Query executed successfully. Rows affected: {cursor.rowcount}"]})
        except Exception as e:
            self.connection.rollback()
            raise Exception(f"MySQL execution error: {str(e)}")
        finally:
            cursor.close()
