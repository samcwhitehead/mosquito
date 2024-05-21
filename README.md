# Mosquito data analysis code

Code used to process ephys recordings and high-speed video. Still in progress.

## Installation

This code is currently packaged as a [Pixi](https://pixi.sh/latest/) project. To get it running:

* Install Pixi (follow instructrions [here](https://pixi.sh/latest/))
* In the terminal or power shell, navigate to the mosquito directory and run `pixi install`
* (Optional) If you want the code to read Photron mraw video files, also run `pixi run postinstall`. But it's easier to just skip this.

After this, you should be ready to run code in the new Pixi project environment. See the [Pixi Python tutorial](https://pixi.sh/latest/tutorials/python/) for extra info.

Some potentially useful notes:
* To use Jupyter notebooks, you can run `pixi run nb` in the terminal/power shell while in the mosquito directory
* TO use this project's Python interpreter in PyCharm, go to the *Add Python Interpreter* dialog (bottom right corner of the PyCharm window) and select *Conda Environment*. Set *Conda Executable* to the full path of the conda file (on Windows: conda.bat) which is located in .pixi/envs/default/libexec. See [here](https://pixi.sh/latest/ide_integration/pycharm/#how-to-use) for more details.
