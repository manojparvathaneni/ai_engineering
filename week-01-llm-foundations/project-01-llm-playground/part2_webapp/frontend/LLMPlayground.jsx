import React, { useState, useEffect, useCallback, useRef } from 'react';

// =============================================================================
// Configuration - Update this to point to your backend
// =============================================================================
const API_BASE_URL = 'http://localhost:8000';

// =============================================================================
// API Client
// =============================================================================
const api = {
  headers: (apiKey) => ({
    'Content-Type': 'application/json',
    ...(apiKey && { 'X-API-Key': apiKey })
  }),

  async generate(prompt, options, apiKey) {
    const res = await fetch(`${API_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: this.headers(apiKey),
      body: JSON.stringify({ prompt, ...options })
    });
    if (!res.ok) throw new Error((await res.json()).detail || 'Generation failed');
    return res.json();
  },

  async getSessions(apiKey) {
    const res = await fetch(`${API_BASE_URL}/api/sessions`, { headers: this.headers(apiKey) });
    if (!res.ok) throw new Error('Failed to fetch sessions');
    return res.json();
  },

  async getSession(sessionId, apiKey) {
    const res = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`, { headers: this.headers(apiKey) });
    if (!res.ok) throw new Error('Failed to fetch session');
    return res.json();
  },

  async createSession(name, systemPrompt, apiKey) {
    const res = await fetch(`${API_BASE_URL}/api/sessions`, {
      method: 'POST',
      headers: this.headers(apiKey),
      body: JSON.stringify({ name, system_prompt: systemPrompt })
    });
    if (!res.ok) throw new Error('Failed to create session');
    return res.json();
  },

  async deleteSession(sessionId, apiKey) {
    const res = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`, {
      method: 'DELETE',
      headers: this.headers(apiKey)
    });
    if (!res.ok) throw new Error('Failed to delete session');
    return res.json();
  },

  async getConfig(apiKey) {
    const res = await fetch(`${API_BASE_URL}/api/config`, { headers: this.headers(apiKey) });
    if (!res.ok) throw new Error('Failed to fetch config');
    return res.json();
  },

  async updateConfig(config, apiKey) {
    const res = await fetch(`${API_BASE_URL}/api/config`, {
      method: 'PUT',
      headers: this.headers(apiKey),
      body: JSON.stringify(config)
    });
    if (!res.ok) throw new Error('Failed to update config');
    return res.json();
  },

  async getTemplates() {
    const res = await fetch(`${API_BASE_URL}/api/templates`);
    if (!res.ok) throw new Error('Failed to fetch templates');
    return res.json();
  }
};

// =============================================================================
// Icons (inline SVG for simplicity)
// =============================================================================
const Icons = {
  Send: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
    </svg>
  ),
  Settings: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
  Plus: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
  ),
  Chat: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  ),
  Trash: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
  ),
  Key: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
    </svg>
  ),
  Info: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  ChevronDown: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  )
};

// =============================================================================
// Tooltip Component
// =============================================================================
const Tooltip = ({ content, children }) => {
  const [show, setShow] = useState(false);
  return (
    <div className="relative inline-flex items-center">
      <div onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
        {children}
      </div>
      {show && (
        <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 text-xs bg-gray-900 text-white rounded-lg shadow-lg whitespace-nowrap">
          {content}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
        </div>
      )}
    </div>
  );
};

// =============================================================================
// Parameter Slider Component
// =============================================================================
const ParamSlider = ({ label, value, onChange, min, max, step, tooltip, color = 'indigo' }) => {
  const colors = {
    indigo: 'bg-indigo-500',
    emerald: 'bg-emerald-500',
    amber: 'bg-amber-500',
    rose: 'bg-rose-500'
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">{label}</span>
          {tooltip && (
            <Tooltip content={tooltip}>
              <Icons.Info />
            </Tooltip>
          )}
        </div>
        <span className={`text-sm font-mono font-semibold text-${color}-600`}>
          {typeof value === 'number' ? (Number.isInteger(step) ? value : value.toFixed(2)) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className={`w-full h-2 rounded-full appearance-none cursor-pointer bg-gray-200 
          [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:${colors[color]} 
          [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:cursor-pointer
          [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:hover:scale-110`}
      />
    </div>
  );
};

// =============================================================================
// Message Bubble Component
// =============================================================================
const MessageBubble = ({ role, content }) => {
  const isUser = role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[80%] px-4 py-3 rounded-2xl ${
        isUser 
          ? 'bg-indigo-500 text-white rounded-br-md' 
          : 'bg-white border border-gray-200 text-gray-800 rounded-bl-md shadow-sm'
      }`}>
        <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
      </div>
    </div>
  );
};

// =============================================================================
// Settings Panel Component
// =============================================================================
const SettingsPanel = ({ config, onUpdate, templates, onClose }) => {
  const [systemPrompt, setSystemPrompt] = useState(config?.system_prompt || '');
  const [temperature, setTemperature] = useState(config?.default_temperature || 0.7);
  const [topP, setTopP] = useState(config?.default_top_p || 0.9);
  const [topK, setTopK] = useState(config?.default_top_k || 50);
  const [maxTokens, setMaxTokens] = useState(config?.default_max_tokens || 150);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await onUpdate({
        system_prompt: systemPrompt,
        default_temperature: temperature,
        default_top_p: topP,
        default_top_k: topK,
        default_max_tokens: maxTokens
      });
      onClose();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg max-h-[90vh] overflow-y-auto">
        <div className="p-6 border-b border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900">Configuration</h2>
          <p className="text-sm text-gray-500 mt-1">Customize your playground defaults</p>
        </div>

        <div className="p-6 space-y-6">
          {/* System Prompt */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">System Prompt</label>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={4}
              className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 resize-none"
              placeholder="You are a helpful assistant..."
            />
            {/* Templates */}
            {templates && templates.length > 0 && (
              <div className="mt-3">
                <p className="text-xs text-gray-500 mb-2">Quick templates:</p>
                <div className="flex flex-wrap gap-2">
                  {templates.map((t) => (
                    <button
                      key={t.id}
                      onClick={() => setSystemPrompt(t.prompt)}
                      className="px-3 py-1 text-xs bg-gray-100 hover:bg-indigo-100 text-gray-700 hover:text-indigo-700 rounded-full transition-colors"
                    >
                      {t.name}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Parameters */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-900">Default Parameters</h3>
            <ParamSlider
              label="Temperature"
              value={temperature}
              onChange={setTemperature}
              min={0}
              max={2}
              step={0.05}
              tooltip="Controls randomness (0=focused, 2=creative)"
              color="indigo"
            />
            <ParamSlider
              label="Top P"
              value={topP}
              onChange={setTopP}
              min={0.1}
              max={1}
              step={0.05}
              tooltip="Nucleus sampling threshold"
              color="emerald"
            />
            <ParamSlider
              label="Top K"
              value={topK}
              onChange={setTopK}
              min={1}
              max={100}
              step={1}
              tooltip="Number of top tokens to consider"
              color="amber"
            />
            <ParamSlider
              label="Max Tokens"
              value={maxTokens}
              onChange={setMaxTokens}
              min={50}
              max={1000}
              step={10}
              tooltip="Maximum response length"
              color="rose"
            />
          </div>
        </div>

        <div className="p-6 border-t border-gray-100 flex gap-3 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-6 py-2 text-sm bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// Setup Panel (First-time configuration)
// =============================================================================
const SetupPanel = ({ onComplete }) => {
  const [apiKey, setApiKey] = useState('');
  const [backendUrl, setBackendUrl] = useState('http://localhost:8000');
  const [useDemo, setUseDemo] = useState(true);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-8">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-indigo-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Icons.Key />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">Welcome to LLM Playground</h1>
          <p className="text-gray-500 mt-2">Configure your connection to get started</p>
        </div>

        <div className="space-y-6">
          {/* Backend URL */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Backend URL</label>
            <input
              type="text"
              value={backendUrl}
              onChange={(e) => setBackendUrl(e.target.value)}
              className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500"
              placeholder="http://localhost:8000"
            />
            <p className="text-xs text-gray-400 mt-1">Where your FastAPI backend is running</p>
          </div>

          {/* Demo Mode Toggle */}
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
            <div>
              <p className="text-sm font-medium text-gray-700">Use Demo API Key</p>
              <p className="text-xs text-gray-500">For testing without authentication</p>
            </div>
            <button
              onClick={() => setUseDemo(!useDemo)}
              className={`w-12 h-6 rounded-full transition-colors ${useDemo ? 'bg-indigo-500' : 'bg-gray-300'}`}
            >
              <div className={`w-5 h-5 bg-white rounded-full shadow-sm transform transition-transform ${useDemo ? 'translate-x-6' : 'translate-x-0.5'}`} />
            </button>
          </div>

          {/* API Key Input (when not using demo) */}
          {!useDemo && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">API Key</label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500"
                placeholder="your-api-key"
              />
            </div>
          )}

          <button
            onClick={() => onComplete({ apiKey: useDemo ? 'demo-key-12345' : apiKey, backendUrl })}
            className="w-full py-3 bg-indigo-500 hover:bg-indigo-600 text-white rounded-xl font-medium transition-colors"
          >
            Get Started
          </button>

          {/* Instructions */}
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
            <p className="text-sm font-medium text-amber-800 mb-2">🚀 Quick Start</p>
            <ol className="text-xs text-amber-700 space-y-1">
              <li>1. Set <code className="bg-amber-100 px-1 rounded">ANTHROPIC_API_KEY</code> env var</li>
              <li>2. Run: <code className="bg-amber-100 px-1 rounded">uvicorn main:app --reload</code></li>
              <li>3. Click "Get Started" above</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// Main App Component
// =============================================================================
export default function LLMPlayground() {
  // Setup state
  const [isSetup, setIsSetup] = useState(false);
  const [apiKey, setApiKey] = useState('');
  
  // UI state
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  // Data state
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [config, setConfig] = useState(null);
  const [templates, setTemplates] = useState([]);
  
  // Generation state
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  
  // Refs
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Load data when setup completes
  useEffect(() => {
    if (isSetup) {
      loadSessions();
      loadConfig();
      loadTemplates();
    }
  }, [isSetup]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadSessions = async () => {
    try {
      const data = await api.getSessions(apiKey);
      setSessions(data);
    } catch (e) {
      console.error('Failed to load sessions:', e);
    }
  };

  const loadConfig = async () => {
    try {
      const data = await api.getConfig(apiKey);
      setConfig(data.config);
    } catch (e) {
      console.error('Failed to load config:', e);
    }
  };

  const loadTemplates = async () => {
    try {
      const data = await api.getTemplates();
      setTemplates(data.templates);
    } catch (e) {
      console.error('Failed to load templates:', e);
    }
  };

  const selectSession = async (sessionId) => {
    try {
      const data = await api.getSession(sessionId, apiKey);
      setCurrentSession(data);
      setMessages(data.messages || []);
    } catch (e) {
      setError('Failed to load session');
    }
  };

  const createNewSession = async () => {
    try {
      const data = await api.createSession('New Chat', config?.system_prompt, apiKey);
      await loadSessions();
      selectSession(data.session_id);
    } catch (e) {
      setError('Failed to create session');
    }
  };

  const deleteSession = async (sessionId, e) => {
    e.stopPropagation();
    try {
      await api.deleteSession(sessionId, apiKey);
      if (currentSession?.session_id === sessionId) {
        setCurrentSession(null);
        setMessages([]);
      }
      await loadSessions();
    } catch (e) {
      setError('Failed to delete session');
    }
  };

  const handleSubmit = async (e) => {
    e?.preventDefault();
    if (!prompt.trim() || loading) return;

    const userMessage = prompt.trim();
    setPrompt('');
    setError(null);
    setLoading(true);

    // Optimistically add user message
    setMessages(prev => [...prev, { role: 'user', content: userMessage, timestamp: new Date().toISOString() }]);

    try {
      const response = await api.generate(userMessage, {
        session_id: currentSession?.session_id,
        temperature: config?.default_temperature,
        top_p: config?.default_top_p,
        top_k: config?.default_top_k,
        max_tokens: config?.default_max_tokens
      }, apiKey);

      // Add assistant response
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: response.text, 
        timestamp: new Date().toISOString() 
      }]);

      setStats({
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
        timeMs: response.generation_time_ms
      });

      // Update current session if new
      if (!currentSession) {
        setCurrentSession({ session_id: response.session_id });
        loadSessions();
      }

    } catch (e) {
      setError(e.message);
      // Remove optimistic message on error
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleConfigUpdate = async (newConfig) => {
    try {
      const result = await api.updateConfig(newConfig, apiKey);
      setConfig(result.config);
    } catch (e) {
      setError('Failed to update configuration');
    }
  };

  // Setup screen
  if (!isSetup) {
    return (
      <SetupPanel 
        onComplete={({ apiKey: key, backendUrl }) => {
          setApiKey(key);
          setIsSetup(true);
        }} 
      />
    );
  }

  return (
    <div className="h-screen flex bg-gray-50">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-72' : 'w-0'} flex-shrink-0 bg-white border-r border-gray-200 transition-all duration-300 overflow-hidden`}>
        <div className="h-full flex flex-col w-72">
          {/* Header */}
          <div className="p-4 border-b border-gray-100">
            <h1 className="text-lg font-bold text-gray-900">LLM Playground</h1>
            <p className="text-xs text-gray-500">Week 1 Project</p>
          </div>

          {/* New Chat Button */}
          <div className="p-3">
            <button
              onClick={createNewSession}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-500 hover:bg-indigo-600 text-white rounded-xl font-medium text-sm transition-colors"
            >
              <Icons.Plus />
              New Chat
            </button>
          </div>

          {/* Sessions List */}
          <div className="flex-1 overflow-y-auto p-3 space-y-1">
            {sessions.map((session) => (
              <div
                key={session.session_id}
                onClick={() => selectSession(session.session_id)}
                className={`group flex items-center gap-3 px-3 py-2.5 rounded-xl cursor-pointer transition-colors ${
                  currentSession?.session_id === session.session_id
                    ? 'bg-indigo-50 text-indigo-700'
                    : 'hover:bg-gray-100 text-gray-700'
                }`}
              >
                <Icons.Chat />
                <span className="flex-1 text-sm truncate">{session.name}</span>
                <button
                  onClick={(e) => deleteSession(session.session_id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded text-red-500 transition-all"
                >
                  <Icons.Trash />
                </button>
              </div>
            ))}
          </div>

          {/* Settings Button */}
          <div className="p-3 border-t border-gray-100">
            <button
              onClick={() => setShowSettings(true)}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-gray-100 text-gray-700 transition-colors"
            >
              <Icons.Settings />
              <span className="text-sm">Settings</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <div className="h-14 border-b border-gray-200 bg-white flex items-center px-4 gap-4">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <span className="text-sm font-medium text-gray-700">
            {currentSession?.name || 'Select or create a chat'}
          </span>
          {stats && (
            <div className="ml-auto flex items-center gap-4 text-xs text-gray-500">
              <span>↑ {stats.inputTokens} tokens</span>
              <span>↓ {stats.outputTokens} tokens</span>
              <span>{(stats.timeMs / 1000).toFixed(2)}s</span>
            </div>
          )}
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 bg-gradient-to-b from-gray-50 to-white">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="w-20 h-20 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-3xl flex items-center justify-center mx-auto mb-6">
                  <svg className="w-10 h-10 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <h2 className="text-xl font-semibold text-gray-900 mb-2">Start a Conversation</h2>
                <p className="text-gray-500 text-sm">
                  Type a message below to begin exploring LLM generation parameters.
                </p>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto">
              {messages.map((msg, idx) => (
                <MessageBubble key={idx} role={msg.role} content={msg.content} />
              ))}
              {loading && (
                <div className="flex justify-start mb-4">
                  <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Error Banner */}
        {error && (
          <div className="px-4 py-2 bg-red-50 border-t border-red-200">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Input Area */}
        <div className="p-4 border-t border-gray-200 bg-white">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="flex items-end gap-3">
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit();
                    }
                  }}
                  placeholder="Type your message..."
                  rows={1}
                  className="w-full px-4 py-3 pr-12 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 resize-none"
                  style={{ minHeight: '48px', maxHeight: '200px' }}
                />
              </div>
              <button
                type="submit"
                disabled={loading || !prompt.trim()}
                className="p-3 bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-300 text-white rounded-xl transition-colors"
              >
                <Icons.Send />
              </button>
            </div>
            <p className="text-xs text-gray-400 mt-2 text-center">
              Press Enter to send • Shift+Enter for new line
            </p>
          </form>
        </div>
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <SettingsPanel
          config={config}
          templates={templates}
          onUpdate={handleConfigUpdate}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}
