This page details the use of the Command Line Interface tool, for the Graphical User Interface, see [Using the tool (GUI)](using-the-gui.md)

## Preparation

Before running the tool, make sure you have all dependencies installed (see [Installation (developers)](installation-developers.md)).

If you are using the GUI, see the instructions [here instead](using-the-gui.md).

Your input data should be inside the ```data``` directory (or alternatively, edit the ```config.toml``` file).

Set the settings in the ```config.toml``` file to your preferences, see [Configuration](configuration.md).

## Running the tool

Run the ```main.py``` from the terminal:

```
cd "/path/to/SOM/src"
python main.py
```

## Results

The results matrices are saved to the excel file set in ```config.toml```. The individual simulation round results are saved to the ```sim_res``` folder. Plots are saved to the ```output``` folder. 
