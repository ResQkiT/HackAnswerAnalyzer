from flask import Flask, render_template, request, redirect, flash
import os
import pandas as pd
from dao.AnalizerObject import AnalizerObject
from analizators_functions import *

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Для отображения сообщений через flash

# Папка для сохранения загруженных файлов
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Разрешённые расширения для файлов
ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

#Используемые анализаторы
analizers_list = [AnalizerObject("Стандартный анализатор", "Считает число строк и столбцов в файлу", first_analizer),
                  AnalizerObject("Нестандартный анализатор", "Считает ворон", week_analizer),
                  AnalizerObject("Гойда анализатор", "Смотрит сколько в каждом столбце строк", deep_learn_analizer)
                  ]

# Проверяем, является ли файл разрешённым типом
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    global  analizers_list
    return render_template("index.html", analizers_list=analizers_list)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Файл не найден')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('Файл не выбран')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(file_path)  # Сохраняем файл

        # Обрабатываем Excel файл с помощью pandas
        try:
            # Запускаем анализаторы
            for analizer in analizers_list:
                analizer.analize(file_path)

        except Exception as e:
            flash(f'Ошибка при обработке файла: {e}')
            return redirect(request.url)

        return redirect('/')
    else:
        flash('Разрешены только Excel файлы (xls, xlsx)')
        return redirect(request.url)


if __name__ == '__main__':
    # Создаём папку для загрузок, если её нет
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
