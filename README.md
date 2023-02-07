# htde


## Description

TODO ;)


## Installation

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager);

2. Create a virtual environment:
    ```bash
    conda create --name htde python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate htde
    ```

4. Install dependencies:
    ```bash
    pip install torch matplotlib pandas seaborn scikit-learn numpy jupyterlab tqdm opt_einsum
    ```


5. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name htde --all -y
    ```


## Usage

Please, see the jupyter notebook `launch.ipynb` (usage of `ht.py` class) or run the demo script `python demo.py` (demonstration of usage of the `node.py` class).
