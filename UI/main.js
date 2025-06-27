const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { dialog } = require('electron');

function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 650,
    minWidth: 600,
    minHeight: 400,
    webPreferences: {
      contextIsolation: true, // true = safe
      nodeIntegration: false, // false = safe
      preload: path.join(__dirname, 'assets/preload.js')
    }, 
    icon: path.join(__dirname, 'assets/Logo_icon_FINAL.jpg')
  });

  // hide menu bar
  win.setMenuBarVisibility(false);
  win.removeMenu();
  
  win.loadFile('assets/index.html');
}

// create the UI window when ready
app.whenReady().then(createWindow);

// handle parameter loading
ipcMain.handle('load-parameters', () => {
  const filePath = path.join(__dirname, 'assets/parameters.json');
  try {
    const raw = fs.readFileSync(filePath);
    return JSON.parse(raw);
  } catch (err) {
    console.error('Error loading assets/parameters.json:', err);
    return [];
  }
});

/* 
handle python console
*/
let scriptName = './src/main.py'

let runningProcess = null;

ipcMain.on('run-python', (event, args) => {
  // check if the script is already running and if so, don't do anything
  if (runningProcess !== null) {
    event.sender.send('python-output', 'Script is already running');
    return;
  }
  // create a new process for the python script
  event.sender.send('python-output', `> python -u ${scriptName}\n\n`);
  const py = spawn('python', ['-u', `${scriptName}`, '--ui', ...args]);
  runningProcess = py;
  // handle the output from the script
  py.stdout.on('data', (data) => {
    event.sender.send('python-output', data.toString());
  });
  py.stderr.on('data', (data) => {
    event.sender.send('python-output', `ERROR: ${data}\n`);
  });
  py.on('close', (code) => {
    event.sender.send('python-output', `Process exited with code ${code}\n\n`);
    runningProcess = null;
    event.sender.send('python-stopped');
  });
});

ipcMain.on('stop-python', (event) => {
  if (runningProcess !== null) {
    runningProcess.kill('SIGTERM');
    event.sender.send('python-output', 'Process terminated by user.\n');
    runningProcess = null;
  }
});

// handle file selection
ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile']
  });
  return result.canceled ? null : result.filePaths[0];
});
