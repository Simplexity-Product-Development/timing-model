# Overview

This repo contains supporting material for the [Simplexity Timing Model Whitepaper](https://www.simplexitypd.com/maximizing-robotics-performance/).

`timing_model_scheduler.py` contains a basic implementation that allows you to schedule and plot various timing studies. Run command `python timing_model_scheduler.py` to see an example schedule (mashed potatoes example used in whitepaper).

`timing_model_workspace.ipynb` is a Jupyter Notebook that contains examples on how to use `timing_model_scheduler.py` and shows how some of the examples for the whitepaper were generated.

`environment.yml` is an anaconda environment that you may use to run the scripts. Anaconda is a python package manager that allows you to more easily manage dependencies when running python scripts. Alternatively, you may install all dependencies for these scripts manually using `pip install`. To activate the environment in the `environment.yml` file:

1. Install [Anaconda](https://docs.anaconda.com/) or [Miniconda](https://docs.anaconda.com/miniconda/)
2. Run Anaconda Terminal
3. Navigate to where this repo is cloned or downloaded on your local PC using `cd` command.
4. Run command `conda env create --file environment.yml --solver libmamba`
   1. The `--solver` flag specifies the solver used by anaconda to set up the dependencies, and libmamba tends to be faster.
   2. The command may take several minutes to install all dependencies and configure the environment.
5. Run command `conda activate timing-model` to activate the environment.
6. Run command `python timing_model_scheduler.py` to see an example schedule.

If you would like any input on how to adapt the scheduler to benefit your product, feel free to contact us at [Simplexity Product Development](https://www.simplexitypd.com/contact/).