import os
import matplotlib.pyplot as plt
from pathlib import Path

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
        
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def draw_pic(flight_data):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size
    plt.title('Month vs Passenger')
    plt.ylabel('Total Passengers')
    plt.xlabel('Months')
    plt.grid(True)
    plt.autoscale(axis='x',tight=True)
    plt.plot(flight_data['passengers'])
    plt.savefig('original.jpg')