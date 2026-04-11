import oracledb
import pandas as pd
from typing import Dict, List

class OracleDatabaseManager:
    def __init__(self):
        self.connection = None
        self.db_type = "oracle"
        
    def connect(self, username, password, host, port, service_name):
        """Connect to Oracle database. service_name maps to database"""
        try:
            if not hasattr(self, '_client_initialized'):
                try:
                    oracledb.init_oracle_client()
                except Exception:
                    pass
                self._client_initialized = True
            
            dsn = f"{host}:{port}/{service_name}"
            self.connection = oracledb.connect(
                user=username,
                password=password,
                dsn=dsn
            )
            return True, "Connected successfully"
        except Exception as e:
            error_msg = str(e)
            print(f"Oracle connection failed: {error_msg}")
            return False, error_msg
            
    def get_all_tables(self) -> List[str]:
        if not self.connection:
            return []
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                SELECT table_name FROM user_tables
                ORDER BY table_name
            """)
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error fetching table list: {str(e)}")
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
                    SELECT column_name, data_type, nullable, data_default, data_length, data_precision, data_scale
                    FROM user_tab_columns 
                    WHERE table_name = :table_name
                    ORDER BY column_id
                """, {'table_name': table.upper()})
                
                columns = []
                for row in cursor.fetchall():
                    # Format data default properly if it's not string
                    default_val = str(row[3]) if row[3] is not None else None
                    if default_val and default_val.strip() == 'NULL':
                        default_val = None

                    columns.append({
                        'name': row[0],
                        'type': row[1],
                        'nullable': row[2],
                        'default': default_val,
                        'length': row[4],
                        'precision': row[5],
                        'scale': row[6],
                        'extra': None # for compatibility
                    })
                
                cursor.execute("""
                    SELECT 
                        a.column_name,
                        c_pk.table_name r_table_name,
                        b.column_name r_column_name
                    FROM user_cons_columns a
                    JOIN user_constraints c ON a.owner = c.owner AND a.constraint_name = c.constraint_name
                    JOIN user_constraints c_pk ON c.r_owner = c_pk.owner AND c.r_constraint_name = c_pk.constraint_name
                    JOIN user_cons_columns b ON c_pk.owner = b.owner AND c_pk.constraint_name = b.constraint_name AND b.position = a.position
                    WHERE c.constraint_type = 'R' AND a.table_name = :table_name
                """, {'table_name': table.upper()})
                
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
            print(f"Error fetching schema: {str(e)}")
        finally:
            cursor.close()
        return schema_info
        
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        if not self.connection:
            return pd.DataFrame()
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE ROWNUM = 1")
            if cursor.fetchone()[0] == 0:
                return pd.DataFrame()
            
            cursor.execute(f"SELECT * FROM {table_name} WHERE ROWNUM <= {limit}")
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
                            processed_row.append("[BLOB/CLOB]")
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
                                processed_row.append("[BLOB/CLOB]")
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
            raise Exception(f"SQL execution error: {str(e)}")
        finally:
            cursor.close()
