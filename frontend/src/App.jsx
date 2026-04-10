import React, { useState, useEffect, useRef } from 'react';
import { connectDb, initLlm, getTables, generateSql, executeSql, downloadResults } from './api';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [dbConnected, setDbConnected] = useState(false);
  const [llmInitialized, setLlmInitialized] = useState(false);
  const [tables, setTables] = useState([]);
  const [selectedTables, setSelectedTables] = useState([]);
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  
  // Config state with localStorage persistence
  const [dbConfig, setDbConfig] = useState(() => {
    const saved = localStorage.getItem('dbConfig');
    return saved ? JSON.parse(saved) : {
      db_type: 'postgresql',
      host: 'localhost',
      port: '5432',
      database: '',
      username: '',
      password: ''
    };
  });
  
  const [llmConfig, setLlmConfig] = useState(() => {
    const saved = localStorage.getItem('llmConfig');
    return saved ? JSON.parse(saved) : {
      llm_type: 'gemini',
      api_key: '',
      model: 'gemini-1.5-flash-lite'
    };
  });

  // Save configs to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('dbConfig', JSON.stringify(dbConfig));
  }, [dbConfig]);

  useEffect(() => {
    localStorage.setItem('llmConfig', JSON.stringify(llmConfig));
  }, [llmConfig]);

  const chatAreaRef = useRef(null);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleConnectDb = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const res = await connectDb(dbConfig);
      if (res.success) {
        setDbConnected(true);
        const tableRes = await getTables();
        setTables(tableRes.tables || []);
        addAlert('success', `Connected! Found ${tableRes.tables?.length || 0} tables.`);
      } else {
        addAlert('danger', res.message || 'Failed to connect.');
      }
    } catch (err) {
      addAlert('danger', 'Error connecting to database.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleInitLlm = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const res = await initLlm(llmConfig);
      if (res.success) {
        setLlmInitialized(true);
        addAlert('success', 'AI Engine active.');
      } else {
        addAlert('danger', res.message || 'Failed to initialize AI.');
      }
    } catch (err) {
      addAlert('danger', 'Error initializing AI.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    if (!input.trim() || isLoading) return;

    if (!dbConnected) {
      addAlert('warning', 'Connect a database first.');
      return;
    }
    if (!llmInitialized) {
      addAlert('warning', 'Initialize AI first.');
      return;
    }

    const userQuery = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userQuery }]);
    setIsLoading(true);

    try {
      const res = await generateSql(userQuery, selectedTables.length > 0 ? selectedTables : tables);
      if (res.success) {
        setMessages(prev => [...prev, { 
          role: 'bot', 
          type: 'sql', 
          content: res.sql
        }]);
      } else {
        addAlert('danger', res.message || 'SQL generation failed.');
      }
    } catch (err) {
      addAlert('danger', 'Error generating SQL.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunSql = async (sql) => {
    setIsLoading(true);
    try {
      const res = await executeSql(sql);
      if (res.success) {
        setMessages(prev => [...prev, { 
          role: 'bot', 
          type: 'data', 
          result: res.result 
        }]);
      } else {
        addAlert('danger', res.message || 'Execution failed.');
      }
    } catch (err) {
      addAlert('danger', 'SQL Execution error.');
    } finally {
      setIsLoading(false);
    }
  };

  const [alerts, setAlerts] = useState([]);
  const addAlert = (type, message) => {
    const id = Date.now();
    setAlerts(prev => [...prev, { id, type, message }]);
    setTimeout(() => {
      setAlerts(prev => prev.filter(a => a.id !== id));
    }, 5000);
  };

  const fillQuery = (text) => {
    setInput(text);
  };

  return (
    <div className="app-layout">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="logo-icon"><i className="fas fa-database"></i></div>
            <div>
              <h2>SQL Chat</h2>
              <small>by ChatWithDB</small>
            </div>
          </div>
          <button className="new-chat-btn" onClick={() => setMessages([])}>
            <i className="fas fa-plus"></i> New Chat
          </button>
        </div>

        <nav className="sidebar-nav">
          <a className={`sidebar-nav-item ${!isConfigOpen ? 'active' : ''}`} href="#" onClick={(e) => { e.preventDefault(); setIsConfigOpen(false); }}>
            <i className="fas fa-comments"></i> Chat
          </a>
          <a className="sidebar-nav-item" href="#" onClick={(e) => { e.preventDefault(); setIsConfigOpen(true); }}>
            <i className="fas fa-layer-group"></i> Tables
            <span className="ms-auto" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{tables.length}</span>
          </a>
        </nav>

        <div className="sidebar-footer">
          <a className="sidebar-nav-item" href="#" onClick={(e) => { e.preventDefault(); setIsConfigOpen(true); }}>
            <i className="fas fa-cog"></i> Settings
          </a>
        </div>
      </aside>

      {/* MAIN AREA */}
      <div className="main-area">
        {/* TOP BAR */}
        <div className="top-bar">
          <div className="top-bar-left">
            <div className="session-tab">
              <i className="fas fa-clock" style={{ fontSize: '0.75rem' }}></i>
              <span>{currentTime}</span>
            </div>
          </div>
          <div className="top-bar-right">
            <div className="top-bar-badge">
              <span className={`status-dot ${dbConnected ? 'connected' : 'disconnected'}`}></span>
              Database
            </div>
            <div className="top-bar-badge">
              <span className={`status-dot ${llmInitialized ? 'connected' : 'disconnected'}`}></span>
              AI
            </div>
          </div>
        </div>

        {/* CHAT AREA */}
        <div className="chat-area" ref={chatAreaRef}>
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-logo">
                <div className="logo-big"><i className="fas fa-robot"></i></div>
                <h1>SQL Chat</h1>
              </div>

              <div className="welcome-columns">
                <div>
                  <div className="welcome-col-header">
                    <i className="fas fa-sun"></i>
                    <span>Examples</span>
                  </div>
                  <div className="welcome-card" onClick={() => fillQuery('Show all users table schema')}>
                    "Show all users table schema" <span className="arrow">→</span>
                  </div>
                  <div className="welcome-card" onClick={() => fillQuery('Give me top 10 users')}>
                    "Give me top 10 users" <span className="arrow">→</span>
                  </div>
                </div>
                <div>
                  <div className="welcome-col-header">
                    <i className="fas fa-bolt"></i>
                    <span>Capabilities</span>
                  </div>
                  <div className="welcome-card">Remembers table context</div>
                  <div className="welcome-card">Supports follow-up corrections</div>
                </div>
                <div>
                  <div className="welcome-col-header">
                    <i className="fas fa-exclamation-circle"></i>
                    <span>Limitations</span>
                  </div>
                  <div className="welcome-card">May generate invalid SQL</div>
                  <div className="welcome-card">Requires table schema context</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="chat-messages">
              {messages.map((msg, i) => (
                <MessageItem key={i} msg={msg} onRun={handleRunSql} />
              ))}
              {isLoading && (
                <div className="chat-msg bot">
                  <div className="avatar"><i className="fas fa-robot"></i></div>
                  <div className="msg-body">
                    <div className="msg-sender">SQL Chat</div>
                    <div className="msg-content"><div className="typing-dots"><span></span><span></span><span></span></div></div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ALERT AREA */}
          <div id="alert-area" style={{ maxWidth: '800px', width: '100%' }}>
            {alerts.map(alert => (
              <div key={alert.id} className={`chat-alert ${alert.type}`}>
                <i className={`fas fa-${alert.type === 'success' ? 'check-circle' : 'exclamation-circle'}`}></i>
                {alert.message}
              </div>
            ))}
          </div>
        </div>

        {/* BOTTOM INPUT BAR */}
        <div className="input-bar-wrap">
          <form onSubmit={handleSendMessage} autoComplete="off">
            <div className="input-bar">
              <i className="fas fa-database" style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}></i>
              <input 
                type="text" 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Enter your question here..." 
              />
              <button type="submit" className="send-btn" disabled={isLoading}>
                <i className={`fas ${isLoading ? 'fa-spinner spinner' : 'fa-arrow-right'}`}></i>
              </button>
            </div>
          </form>
          <div className="input-bar-meta">
            <div>
              <label style={{ cursor: 'pointer' }}>
                <input type="checkbox" defaultChecked style={{ marginRight: '4px' }} />
                Include sample data
              </label>
            </div>
            <a className="prompt-link" onClick={() => setIsConfigOpen(true)}>
              <i className="fas fa-circle-info"></i> Configure
            </a>
          </div>
        </div>
      </div>

      {/* CONFIG PANEL */}
      {isConfigOpen && (
        <div className="config-overlay" onClick={() => setIsConfigOpen(false)}></div>
      )}
      <div className={`config-panel ${isConfigOpen ? 'show' : ''}`}>
        <h4>
          <span><i className="fas fa-cog me-2"></i>Configuration</span>
          <button className="close-config" onClick={() => setIsConfigOpen(false)}>
            <i className="fas fa-times"></i>
          </button>
        </h4>

        <div className="config-section">
          <h6><i className="fas fa-database me-2"></i>Database Connection</h6>
          <form onSubmit={handleConnectDb}>
            <select 
              className="config-input" 
              value={dbConfig.db_type}
              onChange={e => setDbConfig({...dbConfig, db_type: e.target.value})}
            >
              <option value="postgresql">PostgreSQL</option>
              <option value="mysql">MySQL</option>
              <option value="oracle">Oracle</option>
            </select>
            
            <input 
              className="config-input" 
              type="text" 
              placeholder="Host" 
              value={dbConfig.host}
              onChange={e => setDbConfig({...dbConfig, host: e.target.value})}
            />
            <input 
              className="config-input" 
              type="text" 
              placeholder="Port" 
              value={dbConfig.port}
              onChange={e => setDbConfig({...dbConfig, port: e.target.value})}
            />
            <input 
              className="config-input" 
              type="text" 
              placeholder="Database/Service Name" 
              value={dbConfig.db_type === 'oracle' ? dbConfig.service_name : dbConfig.database}
              onChange={e => {
                if(dbConfig.db_type === 'oracle') setDbConfig({...dbConfig, service_name: e.target.value});
                else setDbConfig({...dbConfig, database: e.target.value});
              }}
            />
            <input 
              className="config-input" 
              type="text" 
              placeholder="Username" 
              value={dbConfig.username}
              onChange={e => setDbConfig({...dbConfig, username: e.target.value})}
            />
            <input 
              className="config-input" 
              type="password" 
              placeholder="Password" 
              value={dbConfig.password}
              onChange={e => setDbConfig({...dbConfig, password: e.target.value})}
            />

            <button type="submit" className="config-btn config-btn-primary" disabled={isLoading}>
              <i className={`fas ${isLoading ? 'fa-spinner spinner' : 'fa-plug'}`}></i> Connect Database
            </button>
          </form>
        </div>

        <div className="config-section">
          <h6><i className="fas fa-robot me-2"></i>AI Configuration</h6>
          <form onSubmit={handleInitLlm}>
            <select className="config-input" value={llmConfig.llm_type} onChange={e => setLlmConfig({...llmConfig, llm_type: e.target.value})}>
              <option value="gemini">Google Gemini</option>
            </select>
            <select 
              className="config-input" 
              value={['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-exp', 'gemini-2.0-pro-exp-02-05', 'gemini-2.0-flash-lite-preview-02-05'].includes(llmConfig.model) ? llmConfig.model : 'custom'} 
              onChange={e => {
                const val = e.target.value;
                if (val !== 'custom') {
                  setLlmConfig({...llmConfig, model: val});
                }
              }}
            >
              <option value="gemini-1.5-flash">Gemini 1.5 Flash</option>
              <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
              <option value="gemini-2.0-flash-exp">Gemini 2.0 Flash</option>
              <option value="gemini-2.0-pro-exp-02-05">Gemini 2.0 Pro (Experimental)</option>
              <option value="gemini-2.0-flash-lite-preview-02-05">Gemini 2.0 Flash Lite (Preview)</option>
              <option value="custom">-- Enter Custom Model --</option>
            </select>
            {(!['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-exp', 'gemini-2.0-pro-exp-02-05', 'gemini-2.0-flash-lite-preview-02-05'].includes(llmConfig.model) || llmConfig.model === 'custom') && (
              <input 
                className="config-input" 
                type="text" 
                placeholder="Enter model name (e.g. gemini-1.5-flash)" 
                value={llmConfig.model === 'custom' ? '' : llmConfig.model}
                onChange={e => setLlmConfig({...llmConfig, model: e.target.value})}
              />
            )}
            <input 
              className="config-input" 
              type="password" 
              placeholder="Gemini API Key" 
              value={llmConfig.api_key}
              onChange={e => setLlmConfig({...llmConfig, api_key: e.target.value})}
            />
            <button type="submit" className="config-btn config-btn-primary" disabled={isLoading}>
              <i className={`fas ${isLoading ? 'fa-spinner spinner' : 'fa-rocket'}`}></i> Initialize AI
            </button>
          </form>
        </div>

        <div className="config-section">
          <h6><i className="fas fa-layer-group me-2"></i>Table Context</h6>
          <div style={{ marginTop: '10px' }}>
            <span className="table-tag" style={{ background: 'var(--accent-blue)', color: '#fff', border: 'none' }}>
              {tables.length} Tables Found
            </span>
          </div>
          <div style={{ marginTop: '10px', maxHeight: '200px', overflowY: 'auto' }}>
            {tables.map(table => (
              <div 
                key={table}
                onClick={() => {
                  if (selectedTables.includes(table)) {
                    setSelectedTables(selectedTables.filter(t => t !== table));
                  } else {
                    setSelectedTables([...selectedTables, table]);
                  }
                }}
                className="table-tag"
                style={{ cursor: 'pointer', border: selectedTables.includes(table) ? '1px solid var(--accent-blue)' : '1px solid var(--border-color)' }}
              >
                {table} {selectedTables.includes(table) && <i className="fas fa-check ms-1" style={{ fontSize: '0.6rem' }}></i>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function MessageItem({ msg, onRun }) {
  const isUser = msg.role === 'user';
  
  if (msg.type === 'data') {
    const { columns, data, row_count } = msg.result;
    return (
      <div className="chat-msg bot">
        <div className="avatar"><i className="fas fa-database"></i></div>
        <div className="msg-body">
          <div className="msg-sender">Query Results</div>
          <div className="msg-content">
            <p>Query executed successfully! <strong>{row_count || data.length} rows</strong> returned.</p>
            {data && data.length > 0 && (
              <>
                <div className="results-table-wrap">
                  <table>
                    <thead>
                      <tr>{columns.map(c => <th key={c}>{c}</th>)}</tr>
                    </thead>
                    <tbody>
                      {data.map((row, i) => (
                        <tr key={i}>
                          {row.map((cell, j) => <td key={j}>{cell === null ? 'NULL' : String(cell)}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="result-meta">
                  <a onClick={() => downloadResults()}><i className="fas fa-download me-1"></i>Download CSV</a>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`chat-msg ${isUser ? 'user' : 'bot'}`}>
      <div className="avatar"><i className={`fas fa-${isUser ? 'user' : 'robot'}`}></i></div>
      <div className="msg-body">
        <div className="msg-sender">{isUser ? 'You' : 'SQL Chat'}</div>
        <div className="msg-content">
          {msg.type === 'sql' ? (
            <>
              <p>Here's the generated SQL for your query:</p>
              <div className="sql-block">{msg.content}</div>
              <div className="sql-block-actions">
                <button onClick={() => navigator.clipboard.writeText(msg.content)}><i className="fas fa-copy"></i> Copy</button>
                <button className="run-btn" onClick={() => onRun(msg.content)}><i className="fas fa-play"></i> Run Query</button>
              </div>
            </>
          ) : (
            msg.content
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
