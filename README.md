# Overview

This repository contains the code for a project that utilizes the Hugging Face `transformers` library to generate embeddings, process them, and perform various analyses on their geometrical properties. The code is split into two main files: `main.py` and `utils.py`.

- `main.py`: The main script where all the primary functions are executed, including loading data, generating embeddings, and plotting results.
- `utils.py`: Contains all helper functions used by the main script, including data processing, geometry computations, and plotting utilities.

## Prerequisites

Before running the scripts, ensure you have the following prerequisites set up:

1. **Python Virtual Environment**: It is recommended to use a virtual environment to manage dependencies.

2. **Dependencies**: Install all required Python modules using pip: pip install numpy pandas matplotlib seaborn torch transformers dotenv

3. **Environment Variables**: The Hugging Face access token is required to fetch models and is stored in a `.env` file. Ensure you have this file configured with the following content: ACCESS_TOKEN='your_hugging_face_access_token_here'

## Repository Structure

- `main.py`: Executes the core functions, from data loading to generating and processing embeddings.
- `utils.py`: Provides various helper functions that support data manipulation and result visualization.
- `long_inputs.json`: Input data file used by `main.py`
- `plots`: Folder that stores output plots of dimensionality, mean square radius, and signal strength vs. threshold

## Running the Code

To run the main script, navigate to the repository directory in your terminal and execute: python main.py

Ensure that you are within the virtual environment where all dependencies have been installed.

## Expected Outputs

The script will output several PNG files representing the processed data and visualizations, respectively. These include plots of dimensional participation ratios, mean squared radii, and signal strength across different thresholds.


