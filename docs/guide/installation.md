## Download the model

The code can be downloaded from the [repository](https://github.com/helcomsecretariat/SOM) (steps highlighted in red). If you are using the graphical user interface (GUI), instead download the compiled tool from [releases](https://github.com/helcomsecretariat/SOM/releases).

![image](images/install_github_1.png)
![image](images/install_github_2.png)

Once downloaded, unzip the archive into your project directory.

## Installing requirements

### Python

Running the SOM tool requires Python version 3.12 or above, which can be downloaded [here](https://www.python.org/downloads/). Follow the installer instructions to set it up. 

### Node.js

To run the script using the GUI from the terminal, Node.js is required, which can be downloaded [here](https://nodejs.org/en). Follow the installer instructions to set it up. Alternatively, download the standalone GUI from [here](https://github.com/helcomsecretariat/SOM/releases) to skip this step.

### Setting up the environment

Open up a terminal and enter the following:

1. Navigate to your directory

    ```
    cd "/path/to/SOM"
    ```

2. Create a new python environment (optional, not if using the GUI):

    ```
    python -m venv .
    source bin/activate
    ```

3. Install dependencies:

    ```
    python -m pip install .
    ```

If Node.js was installed to run the GUI from the terminal, additionally install the required node modules:

4. Navigate to UI directory

    ```
    cd "/path/to/SOM/ui"
    ```

5. Install dependencies:

    ```
    npm install
    ```

## Running the tool

To run the tool from the terminal:

```
python "/path/to/SOM/src/main.py"
```

To run the UI from the terminal:

```
cd "/path/to/SOM/ui"
npm run start
```

Alternatively, see [Using the tool](using-the-tool.md).
