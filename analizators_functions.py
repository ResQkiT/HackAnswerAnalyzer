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
 
def deep_learn_analizer(file_path):
    df = pd.read_excel(file_path)
    mp = {}
    # Проверка на количество столбцов
    if df.shape[1] != 4:
        raise Exception(f'Ошибка: Ожидается 5 столбцов, найдено {df.shape[1]}')

    col1 = df.iloc[:, 0].tolist()  # Значения первого столбца
    col2 = df.iloc[:, 1].tolist()  # Значения второго столбца
    col3 = df.iloc[:, 2].tolist()  # Значения третьего столбца
    col4 = df.iloc[:, 3].tolist()  # Значения четвертого столбца
    
    mp["col1"] = len(col1)
    mp["col2"] = len(col2)
    mp["col3"] = len(col3)
    mp["col4"] = len(col4)

    return ResponseObject(mp);