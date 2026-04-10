import axios from 'axios';

const api = axios.create({
  baseURL: '', // Using relative paths for proxy/Vercel
});

export const connectDb = async (config) => {
  const response = await api.post('/api/connect_db', config);
  return response.data;
};

export const initLlm = async (config) => {
  const response = await api.post('/api/init_llm', config);
  return response.data;
};

export const getTables = async () => {
  const response = await api.get('/api/get_tables');
  return response.data;
};

export const getTableInfo = async (tableName) => {
  const response = await api.get(`/api/get_table_info/${tableName}`);
  return response.data;
};

export const generateSql = async (query, tables, includeSample = true) => {
  const response = await api.post('/api/generate_sql', {
    query,
    tables,
    include_sample: includeSample,
  });
  return response.data;
};

export const executeSql = async (sql) => {
  const response = await api.post('/api/execute_sql', { sql });
  return response.data;
};

export const downloadResults = () => {
  window.open('/api/download_results', '_blank');
};

export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
