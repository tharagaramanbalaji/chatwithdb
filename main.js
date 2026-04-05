const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

let mainWindow;
let flaskProcess;

function startFlask() {
  console.log("Starting Flask server...");
  
  // Use the virtual environment python
  const pythonPath = path.join(__dirname, 'vebnv', 'Scripts', 'python.exe');
  
  // Spawn the flask process
  // Adjust the script name if you have a different entry point
  flaskProcess = spawn(pythonPath, ['app.py'], {
    cwd: __dirname,
    env: { ...process.env, PORT: '5000' }
  });

  flaskProcess.stdout.on('data', (data) => {
    console.log(`Flask: ${data}`);
  });

  flaskProcess.stderr.on('data', (data) => {
    console.error(`Flask Error: ${data}`);
  });

  flaskProcess.on('close', (code) => {
    console.log(`Flask process exited with code ${code}`);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    },
    icon: path.join(__dirname, 'static', 'favicon.ico') // Optional
  });

  // Function to poll the Flask server until it's ready
  const pollServer = () => {
    http.get('http://127.0.0.1:5000', (res) => {
      console.log("Flask server is ready. Loading app...");
      mainWindow.loadURL('http://127.0.0.1:5000');
    }).on('error', (err) => {
      console.log("Waiting for Flask server...");
      setTimeout(pollServer, 1000); // Retry every second
    });
  };

  pollServer();

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.on('ready', () => {
  startFlask();
  createWindow();
});

app.on('window-all-closed', function () {
  // On OS X it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    if (flaskProcess) {
      flaskProcess.kill();
    }
    app.quit();
  }
});

app.on('activate', function () {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (mainWindow === null) {
    createWindow();
  }
});

// Ensure Flask is killed when app quits
app.on('will-quit', () => {
  if (flaskProcess) {
    flaskProcess.kill();
  }
});
