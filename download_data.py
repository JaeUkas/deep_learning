import os
from six.moves import urllib

def download_data(url, dir_name, file_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_path = os.path.join(dir_name, file_name)
    urllib.request.urlretrieve(url, file_path)

dir_name = "datasets/iris"
file_name = "iris.csv"
download_data("https://www.openml.org/data/get_csv/61/datasets_61_iris.arft", dir_name, file_name)