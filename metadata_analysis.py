from pymediainfo import MediaInfo
import os
import csv

# Шлях до папки проекту та CSV-файлу
project_folder = "/Users/marynalarchenko/Desktop/Документи/Папка_проєкту"
csv_filename = "metadata_log.csv"
csv_path = os.path.join(project_folder, csv_filename)

def write_metadata_to_csv(metadata, csv_path):
    """Запис метаданих у CSV-файл."""
    # Створюємо файл, якщо його немає, і додаємо заголовки
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metadata.keys())
        if not file_exists:
            writer.writeheader()  # Записуємо заголовки тільки якщо файл новий
        writer.writerow(metadata)  # Дописуємо нові дані

def extract_metadata(file_path):
    """Витягування метаданих з файлу."""
    try:
        media_info = MediaInfo.parse(file_path)
        metadata = {}
        for track in media_info.tracks:
            for key, value in track.to_data().items():
                if value is not None:  # Фільтруємо значення None
                    metadata[key] = value
        return metadata
    except Exception as e:
        print(f"Помилка під час витягування метаданих: {e}")
        return {}

def main():
    """Основна функція для витягування метаданих."""
    file_path = input("Введіть повний шлях до файлу: ").strip()
    
    # Перевірка, чи файл існує
    if not os.path.isfile(file_path):
        print(f"Файл за шляхом '{file_path}' не знайдено.")
        return
    
    # Витягування метаданих
    metadata = extract_metadata(file_path)
    
    if metadata:
        print("Витягнуті метадані:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        # Запис метаданих у CSV
        write_metadata_to_csv(metadata, csv_path)
        print(f"Метадані збережено в файл: {csv_path}")
    else:
        print("Не вдалося отримати метадані або файл порожній.")

if __name__ == "__main__":
    main()

