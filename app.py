from flask import Flask, render_template, request, jsonify, session, send_file
import oracledb
import google.generativeai as genai
import requests
from typing import Dict, List, Optional
import pandas as pd
import json
import os
from datetime import datetime
import io
import traceback
from flask_session import Session 
import tempfile
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'  
Session(app)

# Global variables
db_manager = None
llm_manager = None

class DatabaseManager:
    def __init__(self):
        self.connection = None
        
    def connect(self, username: str, password: str, host: str, port: str, service_name: str) -> bool:
        """Connect to Oracle database"""
        try:
            # Initialize Oracle client (only needed once)
            if not hasattr(self, '_client_initialized'):
                try:
                    oracledb.init_oracle_client()
                except Exception:
                    # Client might already be initialized or not needed
                    pass
                self._client_initialized = True
            
            # Create connection string
            dsn = f"{host}:{port}/{service_name}"
            self.connection = oracledb.connect(
                user=username,
                password=password,
                dsn=dsn
            )
            return True
        except Exception as e:
            print(f"Database connection failed: {str(e)}")
            return False
    
    def get_all_tables(self) -> List[str]:
        """Get list of all tables in the database"""
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
        """Get schema information for specified tables or all tables"""
        if not self.connection:
            return {}
        
        schema_info = {}
        cursor = self.connection.cursor()
        
        try:
            # Get all tables if none specified
            if not table_names:
                table_names = self.get_all_tables()
            
            # Get column information for each table
            for table in table_names:
                cursor.execute("""
                    SELECT column_name, data_type, nullable, data_default, data_length, data_precision, data_scale
                    FROM user_tab_columns 
                    WHERE table_name = :table_name
                    ORDER BY column_id
                """, {'table_name': table.upper()})
                
                columns = []
                for row in cursor.fetchall():
                    col_info = {
                        'name': row[0],
                        'type': row[1],
                        'nullable': row[2],
                        'default': row[3],
                        'length': row[4],
                        'precision': row[5],
                        'scale': row[6]
                    }
                    columns.append(col_info)
                
                # Get foreign key information
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
        """Get sample data from a table with better error handling"""
        if not self.connection:
            return pd.DataFrame()
        
        cursor = self.connection.cursor()
        try:
            # First check if table exists and has data
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE ROWNUM = 1")
            if cursor.fetchone()[0] == 0:
                return pd.DataFrame()
            
            # Get sample data with ROWNUM for Oracle
            cursor.execute(f"SELECT * FROM {table_name} WHERE ROWNUM <= {limit}")
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            # Convert Oracle-specific data types and handle problematic values
            processed_data = []
            for row in data:
                processed_row = []
                for item in row:
                    if item is None:
                        processed_row.append(None)
                    elif hasattr(item, 'read'):  # Handle CLOB/BLOB
                        try:
                            processed_row.append(str(item.read())[:100])  # Truncate large text
                        except:
                            processed_row.append("[BLOB/CLOB]")
                    elif hasattr(item, 'date'):  # Handle Oracle date/datetime objects
                        try:
                            processed_row.append(item.strftime('%Y-%m-%d %H:%M:%S'))
                        except:
                            processed_row.append(str(item))
                    else:
                        try:
                            # Convert to JSON-serializable types
                            if pd.isna(item):
                                processed_row.append(None)
                            else:
                                processed_row.append(str(item))
                        except:
                            processed_row.append(str(item))
                processed_data.append(processed_row)
            
            df = pd.DataFrame(processed_data, columns=columns)
            
            # Convert any remaining problematic pandas types to strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
                    df[col] = df[col].replace('NaT', None)
                    df[col] = df[col].replace('nan', None)
            
            return df
        except Exception as e:
            print(f"Could not fetch sample data from {table_name}: {str(e)}")
            return pd.DataFrame()
        finally:
            cursor.close()
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        if not self.connection:
            raise Exception("No database connection")
        
        # Remove trailing semicolon if present
        sql = sql.strip().rstrip(';')
        
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql)
            
            # For SELECT queries
            if sql.strip().upper().startswith('SELECT'):
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                # Process the data to handle Oracle-specific types
                processed_data = []
                for row in data:
                    processed_row = []
                    for item in row:
                        if item is None:
                            processed_row.append(None)
                        elif hasattr(item, 'read'):  # Handle CLOB/BLOB
                            try:
                                processed_row.append(str(item.read())[:1000])
                            except:
                                processed_row.append("[BLOB/CLOB]")
                        elif hasattr(item, 'date'):  # Handle Oracle date/datetime objects
                            try:
                                processed_row.append(item.strftime('%Y-%m-%d %H:%M:%S'))
                            except:
                                processed_row.append(str(item))
                        elif isinstance(item, (int, float)):
                            # Handle numeric types properly
                            if pd.isna(item):
                                processed_row.append(None)
                            else:
                                processed_row.append(item)
                        else:
                            # Convert everything else to string
                            try:
                                processed_row.append(str(item) if item is not None else None)
                            except:
                                processed_row.append("[UNCONVERTIBLE]")
                    processed_data.append(processed_row)
                
                # Create DataFrame with processed data
                df = pd.DataFrame(processed_data, columns=columns)
                
                # Final cleanup of any remaining problematic values
                for col in df.columns:
                    df[col] = df[col].where(pd.notnull(df[col]), None)
                
                return df
            else:
                # For INSERT/UPDATE/DELETE queries
                self.connection.commit()
                return pd.DataFrame({'Result': [f"Query executed successfully. Rows affected: {cursor.rowcount}"]})
                
        except Exception as e:
            self.connection.rollback()
            raise Exception(f"SQL execution error: {str(e)}")
        finally:
            cursor.close()

import pymysql
import pandas as pd
from typing import Dict, List

class MySQLDatabaseManager:
    def __init__(self):
        self.connection = None
        self.db_type = "mysql"
        
    def connect(self, host: str, port: str, database: str, username: str, password: str) -> bool:
        """Connect to MySQL database"""
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
        """Get list of all tables in the MySQL database"""
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
        """Get schema information for specified tables or all tables"""
        if not self.connection:
            return {}
        
        schema_info = {}
        cursor = self.connection.cursor()
        
        try:
            # Get all tables if none specified
            if not table_names:
                table_names = self.get_all_tables()
            
            # Get column information for each table
            for table in table_names:
                # Get column information
                cursor.execute(f"DESCRIBE {table}")
                columns = []
                for row in cursor.fetchall():
                    col_info = {
                        'name': row[0],
                        'type': row[1],
                        'nullable': row[2],
                        'key': row[3],
                        'default': row[4],
                        'extra': row[5]
                    }
                    columns.append(col_info)
                
                # Get foreign key information
                cursor.execute("""
                    SELECT 
                        COLUMN_NAME,
                        REFERENCED_TABLE_NAME,
                        REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = %s 
                    AND REFERENCED_TABLE_NAME IS NOT NULL
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
        """Get sample data from a table with better error handling"""
        if not self.connection:
            return pd.DataFrame()
        
        cursor = self.connection.cursor()
        try:
            # First check if table exists and has data
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} LIMIT 1")
            if cursor.fetchone()[0] == 0:
                return pd.DataFrame()
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            # Convert MySQL-specific data types and handle problematic values
            processed_data = []
            for row in data:
                processed_row = []
                for item in row:
                    if item is None:
                        processed_row.append(None)
                    elif hasattr(item, 'read'):  # Handle BLOB/TEXT
                        try:
                            processed_row.append(str(item.read())[:100])  # Truncate large text
                        except:
                            processed_row.append("[BLOB/TEXT]")
                    elif hasattr(item, 'date'):  # Handle MySQL date/datetime objects
                        try:
                            processed_row.append(item.strftime('%Y-%m-%d %H:%M:%S'))
                        except:
                            processed_row.append(str(item))
                    else:
                        try:
                            # Convert to JSON-serializable types
                            if pd.isna(item):
                                processed_row.append(None)
                            else:
                                processed_row.append(str(item))
                        except:
                            processed_row.append(str(item))
                processed_data.append(processed_row)
            
            df = pd.DataFrame(processed_data, columns=columns)
            
            # Convert any remaining problematic pandas types to strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
                    df[col] = df[col].replace('NaT', None)
                    df[col] = df[col].replace('nan', None)
            
            return df
        except Exception as e:
            print(f"Could not fetch sample data from {table_name}: {str(e)}")
            return pd.DataFrame()
        finally:
            cursor.close()
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        if not self.connection:
            raise Exception("No database connection")
        
        # Remove trailing semicolon if present
        sql = sql.strip().rstrip(';')
        
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql)
            
            # For SELECT queries
            if sql.strip().upper().startswith('SELECT'):
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                # Process the data to handle MySQL-specific types
                processed_data = []
                for row in data:
                    processed_row = []
                    for item in row:
                        if item is None:
                            processed_row.append(None)
                        elif hasattr(item, 'read'):  # Handle BLOB/TEXT
                            try:
                                processed_row.append(str(item.read())[:1000])
                            except:
                                processed_row.append("[BLOB/TEXT]")
                        elif hasattr(item, 'date'):  # Handle MySQL date/datetime objects
                            try:
                                processed_row.append(item.strftime('%Y-%m-%d %H:%M:%S'))
                            except:
                                processed_row.append(str(item))
                        elif isinstance(item, (int, float)):
                            # Handle numeric types properly
                            if pd.isna(item):
                                processed_row.append(None)
                            else:
                                processed_row.append(item)
                        else:
                            # Convert everything else to string
                            try:
                                processed_row.append(str(item) if item is not None else None)
                            except:
                                processed_row.append("[UNCONVERTIBLE]")
                    processed_data.append(processed_row)
                
                # Create DataFrame with processed data
                df = pd.DataFrame(processed_data, columns=columns)
                
                # Final cleanup of any remaining problematic values
                for col in df.columns:
                    df[col] = df[col].where(pd.notnull(df[col]), None)
                
                return df
            else:
                # For INSERT/UPDATE/DELETE queries
                self.connection.commit()
                return pd.DataFrame({'Result': [f"Query executed successfully. Rows affected: {cursor.rowcount}"]})
                
        except Exception as e:
            self.connection.rollback()
            raise Exception(f"MySQL execution error: {str(e)}")
        finally:
            cursor.close()
            
class LLMManager:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name=model_name)
            # Test the connection
            self.model.generate_content("Hello, test connection")
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini: {str(e)}")
    
    def generate_sql(self, natural_query: str, selected_tables: List[str], full_schema_info: Dict, db_manager, include_sample_data: bool = True) -> str:
        """Generate SQL from natural language query using selected table schemas and sample data"""
        
        if not selected_tables:
            raise Exception("No tables selected for context")
        
        # Format schema information for selected tables only
        schema_context = "Database Schema and Sample Data Context:\n"
        schema_context += "=" * 60 + "\n\n"
        
        for table in selected_tables:
            if table in full_schema_info:
                table_info = full_schema_info[table]
                schema_context += f"Table: {table}\n"
                schema_context += "-" * (len(table) + 7) + "\n"
                
                # Add column information
                schema_context += "Columns:\n"
                for col in table_info['columns']:
                    type_info = col['type']
                    if col['length'] and col['type'] in ['VARCHAR2', 'CHAR']:
                        type_info += f"({col['length']})"
                    elif col['precision'] and col['type'] in ['NUMBER']:
                        if col['scale']:
                            type_info += f"({col['precision']},{col['scale']})"
                        else:
                            type_info += f"({col['precision']})"
                    
                    nullable = "NULL" if col['nullable'] == 'Y' else "NOT NULL"
                    schema_context += f"  • {col['name']} - {type_info} - {nullable}"
                    
                    if col['default']:
                        schema_context += f" - Default: {col['default']}"
                    schema_context += "\n"
                
                # Add foreign key information
                if table_info['foreign_keys']:
                    schema_context += "\nForeign Keys:\n"
                    for fk_col, fk_info in table_info['foreign_keys'].items():
                        schema_context += f"  • {fk_col} → {fk_info['references_table']}.{fk_info['references_column']}\n"
                
                # Add sample data if requested and available
                if include_sample_data and db_manager:
                    try:
                        sample_df = db_manager.get_sample_data(table, limit=3)
                        if not sample_df.empty:
                            schema_context += f"\nSample Data (3 rows):\n"
                            # Convert DataFrame to a readable format
                            sample_data_str = sample_df.to_string(index=False, max_cols=10, max_colwidth=30)
                            schema_context += sample_data_str + "\n"
                            
                            # Add data insights
                            schema_context += f"\nData Insights for {table}:\n"
                            for col in sample_df.columns:
                                unique_values = sample_df[col].nunique()
                                has_nulls = sample_df[col].isnull().any()
                                schema_context += f"  • {col}: {unique_values} unique values"
                                if has_nulls:
                                    schema_context += " (contains nulls)"
                                # Show sample values for better context
                                sample_values = sample_df[col].dropna().unique()[:3]
                                if len(sample_values) > 0:
                                    schema_context += f" - Sample values: {', '.join(str(v) for v in sample_values)}"
                                schema_context += "\n"
                        else:
                            schema_context += f"\nSample Data: Table {table} appears to be empty\n"
                    except Exception as e:
                        schema_context += f"\nSample Data: Could not retrieve sample data for {table} ({str(e)})\n"
                
                schema_context += "\n"
        
        prompt = f"""<|system|>
You are an expert Oracle SQL developer. Your ONLY task is to convert natural language queries into Oracle SQL statements.

CRITICAL RULES:
1. Return ONLY the SQL query - nothing else
2. NO explanations, NO thinking, NO markdown, NO code blocks
3. NO "Let me think about this" or similar phrases
4. NO "Here's the SQL query:" or similar introductions
5. Start directly with SELECT, WITH, INSERT, UPDATE, or DELETE
6. End with the SQL query - no additional text
7. DO NOT include any thinking process or reasoning
8. DO NOT explain your approach or methodology
9. DO NOT use phrases like "I'll create a query that..." or "The SQL would be..."
10. JUST return the raw SQL query

IMPORTANT CONTEXT USAGE:
- The Database Schema section below is the authoritative source for all table and column names. Use ONLY these names in your SQL.
- The schema context describes the structure, relationships, and data types. Use it to determine joins, filters, and query logic.
- The Sample Data section is provided ONLY to help you understand the kind of data in each column. DO NOT use sample values for filtering, limiting, or selection in your SQL. Do NOT assume the sample data is complete or representative of all possible values.
- Always use the schema context for query structure, joins, and relationships. Use sample data only for context, not for query logic.

Oracle SQL Guidelines:
- Use Oracle-specific functions: SYSDATE, TO_DATE, TO_CHAR, NVL, DECODE
- Use ROWNUM for row limiting (not LIMIT)
- Use (+) for outer joins in older Oracle syntax
- Use proper Oracle data types and casting
- Use Oracle-specific aggregation functions

Database Schema and Sample Data:
{schema_context}

Instructions:
- Generate ONLY the Oracle SQL query
- Use exact table and column names from the schema above
- For multiple tables, use appropriate JOINs based on foreign key relationships
- Include proper WHERE clauses, GROUP BY, HAVING as needed
- Use Oracle date functions for date comparisons and formatting
- Ensure the query is safe (no DROP, TRUNCATE unless explicitly requested)
- Consider the sample data patterns and data types shown, but do NOT use sample values for filtering or selection
- Use the data insights to write appropriate conditions, but always rely on the schema for structure
- Return ONLY the raw SQL query - no explanations, thinking, or formatting
</s>

<|user|>
Convert to Oracle SQL: {natural_query}
</s>

<|assistant|>
SELECT"""

        try:
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip()

            # Strict cleaning: Only keep lines that are part of a valid SQL statement
            import re
            sql_keywords = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "MERGE")
            lines = sql_query.split('\n')
            cleaned_lines = []
            in_sql = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Skip lines with explanations, markdown, or unwanted phrases
                if any(skip in line.lower() for skip in [
                    'think:', 'thinking:', 'let me', 'i need to', 'okay,',
                    "here's", 'the sql query is', '```sql', '```', 'sql:',
                    'answer:', 'solution:', 'query:', 'result:'
                ]):
                    continue
                # Start collecting SQL when a valid keyword is found
                if any(line.upper().startswith(kw) for kw in sql_keywords):
                    in_sql = True
                if in_sql:
                    # Stop if we hit a line that looks like an explanation or markdown again
                    if any(skip in line.lower() for skip in [
                        'think:', 'explanation:', 'answer:', 'solution:', 'query:', 'result:', '```']):
                        break
                    cleaned_lines.append(line)
            # If no SQL was found, try to extract using regex
            if not cleaned_lines:
                sql_pattern = r"(SELECT|WITH|INSERT|UPDATE|DELETE|MERGE)[\s\S]+"
                match = re.search(sql_pattern, sql_query, re.IGNORECASE)
                if match:
                    cleaned_lines = [match.group(0).strip()]
            final_sql = ' '.join(cleaned_lines).strip()
            # Final fallback: if still nothing, just return the original response
            if not final_sql:
                final_sql = sql_query
            # Debug: Print the cleaned SQL
            print(f"DEBUG: Original response length: {len(sql_query)}")
            print(f"DEBUG: Cleaned SQL length: {len(final_sql)}")
            print(f"DEBUG: Final SQL: {final_sql[:200]}...")
            return final_sql
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

class MySQLLLMManager:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name=model_name)
            # Test the connection
            self.model.generate_content("Hello, test connection")
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini: {str(e)}")
    
    def generate_sql(self, natural_query: str, selected_tables: List[str], full_schema_info: Dict, db_manager, include_sample_data: bool = True) -> str:
        """Generate MySQL SQL from natural language query"""
        
        if not selected_tables:
            raise Exception("No tables selected for context")
        
        # Format schema information for selected tables only
        schema_context = "Database Schema and Sample Data Context:\n"
        schema_context += "=" * 60 + "\n\n"
        
        for table in selected_tables:
            if table in full_schema_info:
                table_info = full_schema_info[table]
                schema_context += f"Table: {table}\n"
                schema_context += "-" * (len(table) + 7) + "\n"
                
                # Add column information
                schema_context += "Columns:\n"
                for col in table_info['columns']:
                    type_info = col['type']
                    nullable = "NULL" if col['nullable'] == 'YES' else "NOT NULL"
                    schema_context += f"  • {col['name']} - {type_info} - {nullable}"
                    
                    if col['default']:
                        schema_context += f" - Default: {col['default']}"
                    if col['extra']:
                        schema_context += f" - {col['extra']}"
                    schema_context += "\n"
                
                # Add foreign key information
                if table_info['foreign_keys']:
                    schema_context += "\nForeign Keys:\n"
                    for fk_col, fk_info in table_info['foreign_keys'].items():
                        schema_context += f"  • {fk_col} → {fk_info['references_table']}.{fk_info['references_column']}\n"
                
                # Add sample data if requested and available
                if include_sample_data and db_manager:
                    try:
                        sample_df = db_manager.get_sample_data(table, limit=3)
                        if not sample_df.empty:
                            schema_context += f"\nSample Data (3 rows):\n"
                            sample_data_str = sample_df.to_string(index=False, max_cols=10, max_colwidth=30)
                            schema_context += sample_data_str + "\n"
                            
                            # Add data insights
                            schema_context += f"\nData Insights for {table}:\n"
                            for col in sample_df.columns:
                                unique_values = sample_df[col].nunique()
                                has_nulls = sample_df[col].isnull().any()
                                schema_context += f"  • {col}: {unique_values} unique values"
                                if has_nulls:
                                    schema_context += " (contains nulls)"
                                sample_values = sample_df[col].dropna().unique()[:3]
                                if len(sample_values) > 0:
                                    schema_context += f" - Sample values: {', '.join(str(v) for v in sample_values)}"
                                schema_context += "\n"
                        else:
                            schema_context += f"\nSample Data: Table {table} appears to be empty\n"
                    except Exception as e:
                        schema_context += f"\nSample Data: Could not retrieve sample data for {table} ({str(e)})\n"
                
                schema_context += "\n"
        
        # Get database type from session or detect from db_manager
        db_type = getattr(db_manager, 'db_type', 'oracle')
        
        if db_type == 'mysql':
            sql_guidelines = """MySQL SQL Guidelines:
- Use MySQL-specific functions: NOW(), DATE(), STR_TO_DATE(), CONCAT(), IFNULL()
- Use LIMIT for row limiting (not ROWNUM)
- Use standard JOIN syntax (not Oracle's (+) syntax)
- Use proper MySQL data types and casting
- Use MySQL-specific aggregation and string functions
- Use backticks for column/table names if they contain special characters
- Use MySQL date functions: DATE(), YEAR(), MONTH(), DAY(), CURDATE(), NOW()"""
        else:
            sql_guidelines = """Oracle SQL Guidelines:
- Use Oracle-specific functions: SYSDATE, TO_DATE, TO_CHAR, NVL, DECODE
- Use ROWNUM for row limiting (not LIMIT)
- Use (+) for outer joins in older Oracle syntax
- Use proper Oracle data types and casting
- Use Oracle-specific aggregation functions"""
        
        prompt = f"""<|system|>
You are an expert {db_type.upper()} SQL developer. Your ONLY task is to convert natural language queries into {db_type.upper()} SQL statements.

CRITICAL RULES:
1. Return ONLY the SQL query - nothing else
2. NO explanations, NO thinking, NO markdown, NO code blocks
3. NO "Let me think about this" or similar phrases
4. NO "Here's the SQL query:" or similar introductions
5. Start directly with SELECT, WITH, INSERT, UPDATE, or DELETE
6. End with the SQL query - no additional text
7. DO NOT include any thinking process or reasoning
8. DO NOT explain your approach or methodology
9. DO NOT use phrases like "I'll create a query that..." or "The SQL would be..."
10. JUST return the raw SQL query

IMPORTANT CONTEXT USAGE:
- The Database Schema section below is the authoritative source for all table and column names. Use ONLY these names in your SQL.
- The schema context describes the structure, relationships, and data types. Use it to determine joins, filters, and query logic.
- The Sample Data section is provided ONLY to help you understand the kind of data in each column. DO NOT use sample values for filtering, limiting, or selection in your SQL. Do NOT assume the sample data is complete or representative of all possible values.
- Always use the schema context for query structure, joins, and relationships. Use sample data only for context, not for query logic.

{sql_guidelines}

Database Schema and Sample Data:
{schema_context}

Instructions:
- Generate ONLY the {db_type.upper()} SQL query
- Use exact table and column names from the schema above
- For multiple tables, use appropriate JOINs based on foreign key relationships
- Include proper WHERE clauses, GROUP BY, HAVING as needed
- Use MySQL date functions for date comparisons and formatting
- Ensure the query is safe (no DROP, TRUNCATE unless explicitly requested)
- Consider the sample data patterns and data types shown, but do NOT use sample values for filtering or selection
- Use the data insights to write appropriate conditions, but always rely on the schema for structure
- Return ONLY the raw SQL query - no explanations, thinking, or formatting
</s>

<|user|>
Convert to {db_type.upper()} SQL: {natural_query}
</s>

<|assistant|>
SELECT"""

        try:
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip()

            # Clean the response (same logic as Oracle LLM manager)
            import re
            sql_keywords = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "MERGE")
            lines = sql_query.split('\n')
            cleaned_lines = []
            in_sql = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if any(skip in line.lower() for skip in [
                    'think:', 'thinking:', 'let me', 'i need to', 'okay,',
                    "here's", 'the sql query is', '```sql', '```', 'sql:',
                    'answer:', 'solution:', 'query:', 'result:'
                ]):
                    continue
                if any(line.upper().startswith(kw) for kw in sql_keywords):
                    in_sql = True
                if in_sql:
                    if any(skip in line.lower() for skip in [
                        'think:', 'explanation:', 'answer:', 'solution:', 'query:', 'result:', '```']):
                        break
                    cleaned_lines.append(line)
            
            if not cleaned_lines:
                sql_pattern = r"(SELECT|WITH|INSERT|UPDATE|DELETE|MERGE)[\s\S]+"
                match = re.search(sql_pattern, sql_query, re.IGNORECASE)
                if match:
                    cleaned_lines = [match.group(0).strip()]
            
            final_sql = ' '.join(cleaned_lines).strip()
            if not final_sql:
                final_sql = sql_query
            
            return final_sql
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/connect_db', methods=['POST'])
def connect_db():
    global db_manager
    try:
        data = request.json
        db_type = data.get('db_type', 'oracle')
        
        # Create appropriate database manager
        if db_type == 'mysql':
            db_manager = MySQLDatabaseManager()
            success = db_manager.connect(
                host=data['host'],
                port=data['port'],
                database=data['database'],
                username=data['username'],
                password=data['password']
            )
        else:  # oracle (existing)
            db_manager = DatabaseManager()
            success = db_manager.connect(
                data['username'], 
                data['password'], 
                data['host'], 
                data['port'], 
                data['service_name']
            )
        
        if success:
            # Load schema info
            all_tables = db_manager.get_all_tables()
            schema_info = db_manager.get_table_schema()
            
            session['db_connected'] = True
            session['db_type'] = db_type
            session['all_tables'] = all_tables
            session['schema_info'] = schema_info
            
            return jsonify({
                'success': True, 
                'message': f'Connected to {db_type.upper()} successfully!',
                'table_count': len(all_tables),
                'db_type': db_type
            })
        else:
            return jsonify({'success': False, 'message': 'Connection failed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/init_llm', methods=['POST'])
def init_llm():
    global llm_manager
    try:
        data = request.json
        llm_type = data.get('llm_type', 'gemini')
        db_type = session.get('db_type', 'oracle')
        
        if llm_type == 'ollama':
            model_name = data.get('model', 'llama2')
            base_url = data.get('base_url', 'http://localhost:11434')
            llm_manager = OllamaLLMManager(model_name, base_url)
        else:  # gemini
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                return jsonify({'success': False, 'message': 'No centralized Gemini API key configured on the server.'})
            model_name = data.get('model', 'gemini-1.5-pro')
            
            # Use MySQL-specific LLM manager if connected to MySQL
            if db_type == 'mysql':
                llm_manager = MySQLLLMManager(api_key, model_name)
            else:
                llm_manager = LLMManager(api_key, model_name)  # Oracle LLM manager
        
        session['llm_initialized'] = True
        session['llm_type'] = llm_type
        
        return jsonify({'success': True, 'message': f'{llm_type.capitalize()} initialized successfully for {db_type.upper()}!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get_tables')
def get_tables():
    if 'all_tables' in session:
        return jsonify({'tables': session['all_tables']})
    return jsonify({'tables': []})

@app.route('/get_table_info/<table_name>')
def get_table_info(table_name):
    if 'schema_info' in session and table_name in session['schema_info']:
        table_info = session['schema_info'][table_name]
        
        # Get sample data
        sample_df = db_manager.get_sample_data(table_name, limit=5)
        sample_data = None
        if not sample_df.empty:
            sample_data = {
                'columns': sample_df.columns.tolist(),
                'data': sample_df.values.tolist()
            }
        
        return jsonify({
            'schema': table_info,
            'sample_data': sample_data
        })
    return jsonify({'error': 'Table not found'})

@app.route('/generate_sql', methods=['POST'])
def generate_sql():
    try:
        data = request.json
        natural_query = data['query']
        selected_tables = data['tables']
        include_sample = data.get('include_sample', True)
        
        if not llm_manager:
            return jsonify({'success': False, 'message': 'LLM not initialized'})
        
        if not session.get('db_connected'):
            return jsonify({'success': False, 'message': 'Database not connected'})
        
        schema_info = session.get('schema_info', {})
        
        sql_query = llm_manager.generate_sql(
            natural_query, 
            selected_tables, 
            schema_info, 
            db_manager, 
            include_sample
        )
        
        return jsonify({'success': True, 'sql': sql_query})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/execute_sql', methods=['POST'])
def execute_sql():
    try:
        data = request.json
        sql = data['sql']
        # Debug print to show the SQL being executed
        print(f"DEBUG: Executing SQL: {sql}")
        if not session.get('db_connected'):
            return jsonify({'success': False, 'message': 'Database not connected'})
        result_df = db_manager.execute_query(sql)
        
        # Convert DataFrame to JSON
        result_data = {
            'columns': result_df.columns.tolist(),
            'data': result_df.values.tolist(),
            'row_count': len(result_df)
        }
        
        # Store result as a temp file and save filename in session
        temp_dir = tempfile.gettempdir()
        filename = f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        file_path = os.path.join(temp_dir, filename)
        result_df.to_csv(file_path, index=False)
        session['last_result_file'] = file_path
        
        return jsonify({'success': True, 'result': result_data})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/download_results')
def download_results():
    file_path = session.get('last_result_file')
    if file_path and os.path.exists(file_path):
        return send_file(
            file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'query_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    return jsonify({'error': 'No results to download'})

@app.route('/test_ollama_connection')
def test_ollama_connection():
    """Simple route to test Ollama connectivity"""
    try:
        base_url = request.args.get('base_url', 'http://localhost:11434').rstrip('/')
        
        # Test basic connectivity
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code != 200:
            return jsonify({'success': False, 'message': f'Ollama API returned status {response.status_code}'})
        
        models = response.json().get('models', [])
        model_names = [model['name'] for model in models]
        
        return jsonify({
            'success': True, 
            'message': f'Ollama is running! Found {len(models)} model(s)',
            'models': model_names,
            'base_url': base_url
        })
        
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'message': 'Connection timed out. Ollama might be under heavy load.'})
    except requests.exceptions.ConnectionError:
        return jsonify({'success': False, 'message': f'Cannot connect to Ollama at {base_url}. Make sure Ollama is running.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)