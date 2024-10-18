import matplotlib.pyplot as plt
import seaborn as sns

class GistogramGenerator:
    def __init__(self) -> None:
        pass

    def generate_pie_chart(self, data: dict, output_path: str):
        """
        Генерирует красивую разноцветную круговую диаграмму с использованием seaborn
        и сохраняет её как изображение.
        
        :param data: Хеш-таблица (словарь), где ключи — категории, а значения — величины.
        :param output_path: Путь для сохранения изображения.
        """
        # Получаем ключи и значения из хеш-таблицы
        labels = list(data.keys())
        sizes = list(data.values())
        
        # Используем палитру из Seaborn для ярких и разноцветных сегментов
        colors = sns.color_palette('bright', len(labels))

        # Генерация круговой диаграммы с разделением между сегментами
        plt.figure(figsize=(7, 7))  # Устанавливаем размер, чтобы диаграмма точно помещалась в квадрат
        wedges, texts, autotexts = plt.pie(
            sizes, 
            labels=labels, 
            colors=colors, 
            autopct='%1.1f%%', 
            startangle=90, 
            pctdistance=0.85, 
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},  # Разделение между сегментами
        )
        
        # Настройка внешнего вида текста
        for text in texts:
            text.set_color('black')  # Цвет меток

        for autotext in autotexts:
            autotext.set_color('white')  # Цвет процентов внутри диаграммы
            autotext.set_weight('bold')  # Жирный шрифт

        # Добавим круг в центре для стиля "пончика"
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        plt.gca().add_artist(centre_circle)

        # Убедимся, что диаграмма остается круговой
        plt.axis('equal')  

        # Сохранение диаграммы как изображение
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
