import requests
from typing import Dict, List
import re

class OllamaManager:
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m['name'] for m in response.json().get('models', [])]
            if model_name not in models and model_name + ":latest" not in models:
                print(f"Warning: Model {model_name} not found in local Ollama list.")
        except Exception as e:
            raise Exception(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
            
    def _build_context(self, selected_tables: List[str], full_schema_info: Dict, db_manager, include_sample_data: bool) -> str:
        schema_context = "Database Schema and Sample Data Context:\n"
        schema_context += "=" * 60 + "\n\n"
        
        for table in selected_tables:
            if table in full_schema_info:
                table_info = full_schema_info[table]
                schema_context += f"Table: {table}\n"
                schema_context += "-" * (len(table) + 7) + "\n"
                
                schema_context += "Columns:\n"
                for col in table_info['columns']:
                    type_info = col['type']
                    if col.get('length') and col['type'] in ['VARCHAR2', 'CHAR', 'VARCHAR']:
                        type_info += f"({col['length']})"
                    elif col.get('precision') and col['type'] in ['NUMBER', 'DECIMAL', 'NUMERIC']:
                        if col.get('scale'):
                            type_info += f"({col['precision']},{col['scale']})"
                        else:
                            type_info += f"({col['precision']})"
                    
                    nullable = "NULL" if col['nullable'] in ['Y', 'YES', True] else "NOT NULL"
                    schema_context += f"  • {col['name']} - {type_info} - {nullable}"
                    
                    if col.get('default'):
                        schema_context += f" - Default: {col['default']}"
                    if col.get('extra'):
                        schema_context += f" - Extra: {col['extra']}"
                    schema_context += "\n"
                
                if table_info.get('foreign_keys'):
                    schema_context += "\nForeign Keys:\n"
                    for fk_col, fk_info in table_info['foreign_keys'].items():
                        schema_context += f"  • {fk_col} → {fk_info['references_table']}.{fk_info['references_column']}\n"
                
                if include_sample_data and db_manager:
                    try:
                        sample_df = db_manager.get_sample_data(table, limit=3)
                        if not sample_df.empty:
                            schema_context += f"\nSample Data (3 rows):\n"
                            sample_data_str = sample_df.to_string(index=False, max_cols=10, max_colwidth=30)
                            schema_context += sample_data_str + "\n"
                    except Exception as e:
                        schema_context += f"\nSample Data: Could not retrieve sample data for {table} ({str(e)})\n"
                schema_context += "\n"
        return schema_context
        
    def generate_sql(self, natural_query: str, selected_tables: List[str], full_schema_info: Dict, db_manager, db_type: str, include_sample_data: bool = True) -> str:
        if not selected_tables:
            raise Exception("No tables selected for context")
            
        schema_context = self._build_context(selected_tables, full_schema_info, db_manager, include_sample_data)
        
        db_rules = {
            "oracle": "Use Oracle-specific functions: SYSDATE, TO_DATE, ROWNUM.",
            "mysql": "Use MySQL-specific functions: NOW(), LIMIT.",
            "postgresql": "Use PostgreSQL-specific functions: CURRENT_DATE, LIMIT."
        }.get(db_type.lower(), "")
        
        prompt = f"""You are an expert {db_type.upper()} SQL developer. Convert natural language queries into {db_type.upper()} SQL statements.

CRITICAL RULES:
1. Return ONLY the SQL query. No explanations, no markdown, no markdown code blocks formatting.
2. The query MUST be raw text. Don't wrap it in ```sql ... ```.

Schema Context:
{schema_context}

{db_rules}

User Query: Convert to {db_type.upper()} SQL: {natural_query}

Your Response (ONLY raw SQL):"""
        
        try:
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            
            sql_query = response.json().get('response', '').strip()
            
            # Simple cleaning
            cleaned = sql_query.replace('```sql', '').replace('```', '').strip()
            return cleaned
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
