The user interface code is separated from the main tool and located in the `UI` directory.

The code uses [electron](https://www.electronjs.org/) to run a standalone app, running Node.js as its backend. This allows for both the frontend and backend to be written in JavaScript, and makes for quick development and updates. 

Most of the files UI files are located in the sub-directory `assets`.

## Server configuration

The Node.js server configuration is defined in the `package.json` file, with additional electron-forge configuration defined in `forge.config.js`. 

## main.js

Running the server with `npm run start` will launch the main.js file. This file creates the application window, as well as handles the server-side logic. The window will start by displaying `assets/index.html`. 

## index.html

This file acts as the main page displayed in the application, which holds the main hierarchy of elements on the page.

## renderer.js

Handles the user-side logic, and creates the dynamic interface as defined in `parameters.json`. 

## parameters.json

Defines the parameter input options to be included in the interface and their order. 

## preload.js

Acts as a bridge between main.js and renderer.js, preventing exposure of server-side methods and control to the user. 

## style.css

Style sheet for the interface elements. 
