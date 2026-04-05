const { contextBridge, ipcRenderer } = require('electron');

// Expose any Node.js APIs or IPC communication through contextBridge
contextBridge.exposeInMainWorld('electronAPI', {
  // Add any APIs you want to expose to window.electronAPI
  // For now, keep it simple
  platform: process.platform,
  version: process.versions.electron
});
