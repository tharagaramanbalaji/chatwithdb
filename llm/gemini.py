import google.generativeai as genai
from typing import Dict, List
import re

class GeminiManager:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name=model_name)
            # Test the connection
            self.model.generate_content("Hello, test connection")
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini: {str(e)}")
            
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
        return schema_context
        
    def generate_sql(self, natural_query: str, selected_tables: List[str], full_schema_info: Dict, db_manager, db_type: str, include_sample_data: bool = True) -> str:
        if not selected_tables:
            raise Exception("No tables selected for context")
            
        schema_context = self._build_context(selected_tables, full_schema_info, db_manager, include_sample_data)
        
        db_rules = {
            "oracle": "Use Oracle-specific functions: SYSDATE, TO_DATE, TO_CHAR, NVL, DECODE. Use ROWNUM for row limiting.",
            "mysql": "Use MySQL-specific functions: NOW(), DATE_FORMAT, IFNULL. Use LIMIT for row limiting.",
            "postgresql": "Use PostgreSQL-specific functions: CURRENT_DATE, TO_CHAR, COALESCE. Use LIMIT for row limiting."
        }.get(db_type.lower(), "")
        
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
- The Sample Data section is provided ONLY to help you understand the kind of data in each column. DO NOT use sample values for filtering, limiting, or selection in your SQL.

Specific {db_type.upper()} Guidelines:
{db_rules}

Database Schema and Sample Data:
{schema_context}

Instructions:
- Generate ONLY the {db_type.upper()} SQL query
- Use exact table and column names from the schema above
- For multiple tables, use appropriate JOINs
- Return ONLY the raw SQL query
</s>

<|user|>
Convert to {db_type.upper()} SQL: {natural_query}
</s>

<|assistant|>"""
        
        try:
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip()
            
            # Simple Strict cleaning
            sql_keywords = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "MERGE")
            lines = sql_query.split('\n')
            cleaned_lines = []
            in_sql = False
            for line in lines:
                line = line.strip()
                if not line: continue
                if any(skip in line.lower() for skip in ['think:', 'thinking:', 'let me', 'i need to', 'okay,', "here's", 'the sql query is', '```sql', '```', 'sql:', 'answer:', 'solution:', 'query:', 'result:']):
                    continue
                if any(line.upper().startswith(kw) for kw in sql_keywords):
                    in_sql = True
                if in_sql:
                    if any(skip in line.lower() for skip in ['think:', 'explanation:', 'answer:', 'solution:', 'query:', 'result:', '```']):
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
