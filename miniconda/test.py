# Script to test conda package installation
import numpy
import plotly
import scipy

if __name__ == '__main__':
    print(f'Numpy version: {numpy.__version__}')
    print(f'Plotly version: {plotly.__version__}')
    print(f'Scipy version: {scipy.__version__}')
