/*
Parameter fields
*/

function createInputField(param) {
  const container = document.getElementById('parameters');
  const label = document.createElement('label');
  label.textContent = param.name;

  let input;

  switch (param.type) {
    case 'text':
      input = document.createElement('input');
      input.type = 'text';
      break;
    case 'integer':
    case 'float':
      input = document.createElement('input');
      input.type = 'number';
      input.step = param.type === 'integer' ? '1' : 'any';
      break;
    case 'flag':
      input = document.createElement('input');
      input.type = 'checkbox';
      break;
    case 'select':
      input = document.createElement('select');
      param.options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = opt;
        input.appendChild(option);
      });
      break;
  }

  input.name = param.arg;
  input.dataset.type = param.type;

  const wrapper = document.createElement('div');
  wrapper.appendChild(label);
  wrapper.appendChild(input);
  container.appendChild(wrapper);
}

function gatherArgs() {
  const inputs = document.querySelectorAll('#parameters input, #parameters select');
  const args = [];

  inputs.forEach(input => {
    const arg = input.name;
    const type = input.dataset.type;

    if ((type === 'text' || type === 'integer' || type === 'float' || type === 'select') && input.value !== '') {
      args.push(arg, input.value);
    } else if (type === 'flag' && input.checked) {
      args.push(arg);
    }
  });

  return args;
}

/*
Console related
*/

// update console window with python output
let currentLine = '';
let visibleLines = [];
window.api.onOutput((data) => {
  if (data.includes('Script is already running')) {
    setStatus('⚠️ Already running');
    return
  }
  for (const char of data) {
    if (char === '\r') {
      // Simulate carriage return: reset position to start of last line
      // but do NOT clear it
      if (visibleLines.length === 0) {
        visibleLines.push('');
      }
      const lastLine = visibleLines[visibleLines.length - 1] || '';
      visibleLines[visibleLines.length - 1] = currentLine + lastLine.slice(currentLine.length);
      currentLine = '';
    } else if (char === '\n') {
      visibleLines.push(currentLine);
      currentLine = '';
    } else {
      currentLine += char;
    }
  }
  if (currentLine) {
    if (visibleLines.length === 0) {
      visibleLines.push(currentLine);
    } else {
      visibleLines[visibleLines.length - 1] = currentLine +
        visibleLines[visibleLines.length - 1].slice(currentLine.length);
    }
  }
  const output = document.getElementById('output');
  output.textContent = visibleLines.join('\n');
  output.scrollTop = output.scrollHeight;
});

function setStatus(msg) {
  document.getElementById('statusBar').textContent = msg;
}

function run() {
  setStatus('▶️ Running...');
  document.getElementById('runButton').disabled = true;
  document.getElementById('stopButton').disabled = false;
  document.getElementById('spinner').style.visibility = 'visible';
  const args = gatherArgs();
  window.api.runPython(args);
}

function stop() {
  // stop current python process
  setStatus('⏹️ Stopping...');
  window.api.stopPython();
}

window.api.onStop(() => {
  setStatus('✔️ Finished');
  document.getElementById('runButton').disabled = false;
  document.getElementById('stopButton').disabled = true;
  document.getElementById('spinner').style.visibility = 'hidden';
});

function clearOutput() {
  const output = document.getElementById('output');
  output.textContent = '';
  currentLine = '';
  visibleLines = [];
}

/*
Initialization
*/

// Initialize parameters
window.addEventListener('DOMContentLoaded', async () => {
  const params = await window.api.loadParameters();
  params.forEach(createInputField);
});
