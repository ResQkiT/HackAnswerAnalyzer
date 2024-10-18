import pandas as pd
from dao.ResponseObject import ResponseObject

def first_analizer(file_path):
    df = pd.read_excel(file_path)
    height_of_file = df.shape[0];
    width_of_file = df.shape[1];
    mp = {"height" : height_of_file , "width" : width_of_file}
    print("")
    return ResponseObject(mp)

def week_analizer(file_path):
    return ResponseObject({});