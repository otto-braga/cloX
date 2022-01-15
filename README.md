# set up virtual environment
create in directory **/path**
> python3 -m venv /path

activate it
> source /path/bin/activate

# install dependencies
basic
> python -m pip install mediapipe python-osc

for drawn gesture classification
> python -m pip install keras tensorflow

# clone repository
clone
> git clone https://github.com/otto-braga/cloX

change into its directory
> cd ./cloX

# run
> python cloX.py /path/to/project.json

JSON project examples can be found in **./examples**