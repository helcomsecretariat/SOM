These steps go through the installation of the SOM package for developers on a Windows system. 
For normal users, see [Installation](installation.md).

## 1. Installing Python

Running the SOM tool requires Python version 3.12 or above, which can be downloaded [here](https://www.python.org/downloads/). Follow the installer instructions to set it up. 
NOTE! The GUI requires Python to be set up as an environment variable.

## 2. Installing Node.js (optional)

To run the script using the GUI from the terminal, Node.js is required, which can be downloaded [here](https://nodejs.org/en). Follow the installer instructions to set it up. 

## 3. Download the model

The code can be downloaded from the [repository](https://github.com/helcomsecretariat/SOM).

Once downloaded, unzip the archive into your project directory.

## 4. Setting up the environment

Open up a terminal and install the python dependencies:

1. Navigate to your directory

    ```
    cd "/path/to/SOM"
    ```

2. Create a new python environment (optional):

    ```
    python -m venv .
    source bin/activate
    ```

3. Install dependencies:

    ```
    python -m pip install .
    ```

To run the GUI from the terminal with Node.js, additionally install the required node modules:

4. Navigate to UI directory

    ```
    cd "/path/to/SOM/ui"
    ```

5. Install dependencies:

    ```
    npm install
    ```

## 5. Running the tool

To run the tool from the terminal:

```
python "/path/to/SOM/src/main.py"
```

To run the UI from the terminal:

```
cd "/path/to/SOM/ui"
npm run start
```

Alternatively, see [Using the tool (CLI)](using-the-tool.md).
