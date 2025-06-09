const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  loadParameters: () => ipcRenderer.invoke('load-parameters'),
  runPython: (args) => ipcRenderer.send('run-python', args),
  stopPython: () => ipcRenderer.send('stop-python'),
  onOutput: (callback) => ipcRenderer.on('python-output', (_, data) => callback(data)),
  onStop: (callback) => ipcRenderer.on('python-stopped', () => callback()),
  selectFile: () => ipcRenderer.invoke('select-file')
});
