from flask import Blueprint, render_template, request, jsonify, session, send_file
import tempfile
import os
from datetime import datetime
import requests

from database import get_db_manager
from llm import get_llm_manager

api_bp = Blueprint('api', __name__)

# Global variables (matching original implementation)
db_manager = None
llm_manager = None

@api_bp.route('/')
def index():
    return render_template('index.html')

@api_bp.route('/favicon.ico')
def favicon():
    return '', 204

@api_bp.route('/connect_db', methods=['POST'])
def connect_db():
    global db_manager
    try:
        data = request.json
        db_type = data.get('db_type', 'oracle')
        
        # Create appropriate database manager
        db_manager = get_db_manager(db_type)
        
        if db_type in ['mysql', 'postgresql']:
            success = db_manager.connect(
                host=data['host'],
                port=data['port'],
                database=data['database'],
                username=data['username'],
                password=data['password']
            )
        else:  # oracle
            success = db_manager.connect(
                data['username'], 
                data['password'], 
                data['host'], 
                data['port'], 
                data.get('service_name', data.get('database'))
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

@api_bp.route('/init_llm', methods=['POST'])
def init_llm():
    global llm_manager
    try:
        data = request.json
        llm_type = data.get('llm_type', 'gemini')
        db_type = session.get('db_type', 'oracle')
        
        if llm_type == 'ollama':
            model_name = data.get('model', 'llama2')
            base_url = data.get('base_url', 'http://localhost:11434')
            llm_manager = get_llm_manager("ollama", model_name=model_name, base_url=base_url)
        else:  # gemini
            api_key = data.get('api_key')
            if not api_key:
                return jsonify({'success': False, 'message': 'No Gemini API key provided. Please enter your API key in the configuration panel.'})
            
            model_name = data.get('model', 'gemini-1.5-pro')
            llm_manager = get_llm_manager("gemini", model_name=model_name, api_key=api_key)
        
        session['llm_initialized'] = True
        session['llm_type'] = llm_type
        
        return jsonify({'success': True, 'message': f'{llm_type.capitalize()} initialized successfully for {db_type.upper()}!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@api_bp.route('/get_tables')
def get_tables():
    if 'all_tables' in session:
        return jsonify({'tables': session['all_tables']})
    return jsonify({'tables': []})

@api_bp.route('/get_table_info/<table_name>')
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

@api_bp.route('/generate_sql', methods=['POST'])
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
        db_type = session.get('db_type', 'oracle')
        
        sql_query = llm_manager.generate_sql(
            natural_query, 
            selected_tables, 
            schema_info, 
            db_manager,
            db_type=db_type,
            include_sample_data=include_sample
        )
        
        return jsonify({'success': True, 'sql': sql_query})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@api_bp.route('/execute_sql', methods=['POST'])
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

@api_bp.route('/download_results')
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

@api_bp.route('/test_ollama_connection')
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
