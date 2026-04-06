import psycopg2
import pandas as pd
from typing import Dict, List

class PostgreSQLDatabaseManager:
    def __init__(self):
        self.connection = None
        self.db_type = "postgresql"
        
    def connect(self, host: str, port: str, database: str, username: str, password: str) -> bool:
        try:
            self.connection = psycopg2.connect(
                host=host,
                port=int(port),
                database=database,
                user=username,
                password=password
            )
            return True
        except Exception as e:
            print(f"PostgreSQL connection failed: {str(e)}")
            return False
            
    def get_all_tables(self) -> List[str]:
        if not self.connection:
            return []
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error fetching PostgreSQL tables: {str(e)}")
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
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default,
                           character_maximum_length, numeric_precision, numeric_scale
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                """, (table,))
                
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        'name': row[0],
                        'type': row[1],
                        'nullable': 'Y' if row[2] == 'YES' else 'N',
                        'default': row[3],
                        'length': row[4],
                        'precision': row[5],
                        'scale': row[6],
                        'extra': None
                    })
                
                cursor.execute("""
                    SELECT kcu.column_name, ccu.table_name AS foreign_table_name, ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s
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
            print(f"Error fetching PostgreSQL schema: {str(e)}")
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
                    elif isinstance(item, (list, dict)):
                        processed_row.append(str(item)[:100])
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
                        elif isinstance(item, (list, dict)):
                            try:
                                processed_row.append(str(item)[:1000])
                            except:
                                processed_row.append("[JSON/JSONB]")
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
            raise Exception(f"PostgreSQL execution error: {str(e)}")
        finally:
            cursor.close()
