/*
Parameter fields
*/

function createInputField(param) {
  const container = document.getElementById('parameters');
  let wrapper = document.createElement('div');

  const label = document.createElement((param.type === 'boolean') ? 'span' : 'label');
  label.textContent = param.name;
  if (['--general_input', '--measure_effects', '--pressure_state'].includes(param.arg)) { label.classList.add('input-data-legacy'); } 
  else if (param.arg === '--input_data') { label.classList.add('input-data'); }
  else if (param.arg === '--scenario') { label.classList.add('scenario'); }
  else if (param.arg === '--random_seed') { label.classList.add('random-seed'); }
  else if (['--mpa_layer_path', '--mpa_layer_id_attribute', '--mpa_layer_name_attribute', 
            '--mpa_layer_measure_attribute', '--mpa_layer_measure_delimiter', 
            '--subbasin_layer_path', '--subbasin_layer_id_attribute'].includes(param.arg)) { label.classList.add('mpas-subbasins'); }

  let input;

  switch (param.type) {
    case 'text':
    case 'integer':
    case 'float':
      input = document.createElement('input');
      input.type = 'number';
      if (param.type === 'text') input.type = 'text';
      if (param.type === 'float') input.step = 'any';
      if (param.type === 'integer') input.step = '1';
      input.name = param.arg;
      input.dataset.type = param.type;
      if (param.arg === '--scenario') { input.classList.add('scenario'); }
      else if (param.arg === '--random_seed') { input.classList.add('random-seed'); }
      else if (['--mpa_layer_id_attribute', '--mpa_layer_name_attribute', 
                '--mpa_layer_measure_attribute', '--mpa_layer_measure_delimiter', 
                '--subbasin_layer_id_attribute'].includes(param.arg)) { input.classList.add('mpas-subbasins'); }
      wrapper.appendChild(label);
      wrapper.appendChild(input);
      break;

    case 'boolean':
      wrapper.classList.add('toggle-wrapper');
      const toggleLabel = document.createElement('label');
      toggleLabel.className = 'toggle';
      const slider = document.createElement('span');
      slider.className = 'slider';
      input = document.createElement('input');
      input.type = 'checkbox';
      input.name = param.arg;
      input.dataset.type = param.type;
      if (param.arg === '--legacy') {
        input.addEventListener('change', function() {
          if (this.checked) {
            document.querySelectorAll('.input-data').forEach(function (element) { element.style.display = 'none'; });
            document.querySelectorAll('.input-data-legacy').forEach(function (element) { element.style.display = 'flex'; });
          } else {
            document.querySelectorAll('.input-data').forEach(function (element) { element.style.display = 'flex'; });
            document.querySelectorAll('.input-data-legacy').forEach(function (element) { element.style.display = 'none'; });
          }
        });
        input.checked = true; // legacy input data checked by default
      } else if (param.arg === '--use_scenario') {
        input.addEventListener('change', function() {
          if (this.checked) { document.querySelectorAll('.scenario').forEach(function (element) { element.style.display = 'flex'; }); }
          else { document.querySelectorAll('.scenario').forEach(function (element) { element.style.display = 'none'; }); }
        });
        input.checked = false; // scenario unchecked by default
      } else if (param.arg === '--use_random_seed') {
        input.addEventListener('change', function() {
          if (this.checked) { document.querySelectorAll('.random-seed').forEach(function (element) { element.style.display = 'flex'; }); }
          else { document.querySelectorAll('.random-seed').forEach(function (element) { element.style.display = 'none'; }); }
        });
        input.checked = false; // random seed unchecked by default
      } else if (param.arg === '--link_mpas_to_subbasins') {
        input.addEventListener('change', function() {
          if (this.checked) { document.querySelectorAll('.mpas-subbasins').forEach(function (element) { element.style.display = 'flex'; }); }
          else { document.querySelectorAll('.mpas-subbasins').forEach(function (element) { element.style.display = 'none'; }); }
        });
        input.checked = false; // linking unchecked by default
      }
      toggleLabel.appendChild(input);
      toggleLabel.appendChild(slider);
      wrapper.appendChild(label);
      wrapper.appendChild(toggleLabel);
      break;

    case 'select':
      input = document.createElement('select');
      param.options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = opt;
        input.appendChild(option);
      });
      input.name = param.arg;
      input.dataset.type = param.type;
      wrapper.appendChild(label);
      wrapper.appendChild(input);
      break;

      case 'filepath':
      input = document.createElement('input');
      input.type = 'text';
      // input.readOnly = true;
      input.placeholder = 'Select a file...';
      input.name = param.arg;
      input.dataset.type = param.type;
      const fileButton = document.createElement('button');
      fileButton.textContent = 'Browse';
      fileButton.className = 'file-select-button';
      fileButton.onclick = async () => {
        const path = await window.api.selectFile();
        if (path) input.value = path;
      };
      const fileWrapper = document.createElement('div');
      fileWrapper.className = 'file-input-wrapper';
      if (['--general_input', '--measure_effects', '--pressure_state'].includes(param.arg)) { fileWrapper.classList.add('input-data-legacy'); }
      else if (param.arg === '--input_data') { fileWrapper.classList.add('input-data'); }
      else if (['--mpa_layer_path', '--subbasin_layer_path'].includes(param.arg)) { fileWrapper.classList.add('mpas-subbasins'); }
      fileWrapper.appendChild(input);
      fileWrapper.appendChild(fileButton);
      wrapper.appendChild(label);
      wrapper.appendChild(fileWrapper);
      break;
    default:
      console.warn('Unknown input type:', param.type);
      return;
  }
  // add wrapper to parameter div
  container.appendChild(wrapper);
}


/*
Select arguments to send to python
*/

function gatherArgs() {
  const inputs = document.querySelectorAll('#parameters input, #parameters select');
  const args = [];

  inputs.forEach(input => {
    const arg = input.name;
    const type = input.dataset.type;

    if ((type === 'text' || type === 'integer' || type === 'float' || type === 'select' || type === 'filepath') && input.value !== '') {
      args.push(arg, input.value);
    } else if (type === 'boolean' && input.checked) {
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


// tell backend to run python with given arguments
function run() {
  setStatus('▶️ Running...');
  document.getElementById('runButton').disabled = true;
  document.getElementById('stopButton').disabled = false;
  document.getElementById('spinner').style.visibility = 'visible';
  const args = gatherArgs();
  window.api.runPython(args);
}


// tell backend to stop python
function stop() {
  // stop current python process
  setStatus('⏹️ Stopping...');
  window.api.stopPython();
}


// what to do on stop
window.api.onStop(() => {
  setStatus('✔️ Finished');
  document.getElementById('runButton').disabled = false;
  document.getElementById('stopButton').disabled = true;
  document.getElementById('spinner').style.visibility = 'hidden';
});


// clear terminal
function clearOutput() {
  const output = document.getElementById('output');
  output.textContent = '';
  currentLine = '';
  visibleLines = [];
}


/*
Resizer
*/

const dragbar = document.getElementById('dragbar');
const leftPanel = document.querySelector('.left-panel');
const rightPanel = document.querySelector('.right-panel');

let isDragging = false;

dragbar.addEventListener('mousedown', function (e) {
  e.preventDefault();
  isDragging = true;
  document.body.style.cursor = 'col-resize';
});

document.addEventListener('mousemove', function (e) {
  if (!isDragging) return;

  const minWidth = 200;
  const maxWidth = window.innerWidth * 0.7;
  let newLeftWidth = e.clientX;

  newLeftWidth = Math.max(minWidth, Math.min(newLeftWidth, maxWidth));
  leftPanel.style.width = newLeftWidth + 'px';
});

document.addEventListener('mouseup', function () {
  isDragging = false;
  document.body.style.cursor = 'default';
});


/*
Initialization
*/

let som_logo = String.raw`
   _____  ____  __  __ 
  / ____|/ __ \|  \/  |
 | (___ | |  | | \  / |
  \___ \| |  | | |\/| |
  ____) | |__| | |  | |
 |_____/ \____/|_|  |_|
                       
Copyright (c) 2025 HELCOM

Source: https://github.com/helcomsecretariat/SOM
Documentation: https://helcomsecretariat.github.io/SOM/
`;

// Initialize parameters
window.addEventListener('DOMContentLoaded', async () => {
  // create parameter fields
  const params = await window.api.loadParameters();
  params.forEach(createInputField);
  // hide standard input data, scenario and random seed initially
  document.querySelectorAll('.input-data').forEach(function (element) { element.style.display = 'none'; });
  document.querySelectorAll('.scenario').forEach(function (element) { element.style.display = 'none'; });
  document.querySelectorAll('.random-seed').forEach(function (element) { element.style.display = 'none'; });
  document.querySelectorAll('.mpas-subbasins').forEach(function (element) { element.style.display = 'none'; });
  // start console with logo printed
  document.getElementById('output').textContent = som_logo;
});
