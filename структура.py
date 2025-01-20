import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from stop_words import get_stop_words
import spacy
import numpy as np
import string
from torch.optim import AdamW
import pandas as pd  # Імпорт pandas для роботи з таблицями

# Завантаження моделі та токенізатора
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Завантаження моделей spacy для української та англійської мов
nlp_uk = spacy.load('xx_ent_wiki_sm')  # Модель для багатьох мов, зокрема для української
nlp_en = spacy.load('en_core_web_sm')  # Модель для англійської мови

# Стоп-слова для української та англійської мов
stop_words_uk = set(get_stop_words("uk"))
stop_words_en = set(get_stop_words("en"))


# Функція для перевірки, чи є текст злочинним чи неадекватним
def is_criminal_or_inappropriate(text, model, tokenizer):
    if not isinstance(text, str):
        raise ValueError(f"Expected input text to be a string, but got {type(text)}")
    
    # Токенізація тексту
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Прогнозування
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    return predictions.item()

# Оцінка моделі
def evaluate_model_with_warning(model, test_loader):
    results = []
    for batch in test_loader:
        # Припустимо, що test_loader є списком словників із ключем 'text'
        input_texts = batch['text']
        for text in input_texts:
            prediction = is_criminal_or_inappropriate(text, model, tokenizer)
            results.append({"text": text, "prediction": prediction})
    
    # Створення таблиці з результатами
    df = pd.DataFrame(results)
    df['prediction_label'] = df['prediction'].apply(lambda x: "Цей текст є злочинним або неадекватним." if x == 1 else "Цей текст є адекватним.")
    
    # Показуємо результати в табличному вигляді
    print(df)

# Приклад даних для тестування
test_loader = [
    {'text': ["This is a test message.", "Some harmful content."]}  # Приклад текстів для тестування
]

# Викликаємо функцію
evaluate_model_with_warning(model, test_loader)


# Функція для обробки тексту
def preprocess_text(text, language='uk'):
    """Очищення та підготовка тексту до аналізу."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Вибір моделі і стоп-слів на основі мови
    if language == 'uk':
        nlp = nlp_uk
        stop_words = stop_words_uk
    else:
        nlp = nlp_en
        stop_words = stop_words_en
    
    # Токенізація та лематизація
    doc = nlp(text)
    
    # Створення списку лематизованих слів, які не є стоп-словами
    processed_words = [token.lemma_ for token in doc if token.text not in stop_words]
    
    # Повертаємо очищений та лематизований текст
    return ' '.join(processed_words)


# Приклад використання функції
text = "Текст для перевірки"
print(is_criminal_or_inappropriate(text, model, tokenizer))

# Оцінка моделі з попередженням
def evaluate_model_with_warning(model, test_loader):
    model.eval()
    results = []
    
    # Обробка batch з DataLoader
    for batch in test_loader:
        input_ids, attention_mask, labels = batch  # Отримуємо дані з DataLoader
        input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]  # Декодуємо тензори в текст
        
        for text, label in zip(input_texts, labels):
            try:
                prediction = is_criminal_or_inappropriate(text, model, tokenizer)
                results.append({"text": text, "label": label.item(), "prediction": prediction})
            except ValueError as e:
                print(f"Error processing text: {e}")
    
    # Створення таблиці результатів
    df = pd.DataFrame(results)
    df['prediction_label'] = df['prediction'].apply(
        lambda x: "Цей текст є злочинним або неадекватним." if x == 1 else "Цей текст є адекватним."
    )
    print(df)
    return df

# Приклад текстових даних для тренувального та тестового набору
train_texts = [
    "Економіка України стабільно зростає в останні роки.",
    "Технології постійно розвиваються, змінюючи наше повсякденне життя. Тому переведіть мені на картку гроші",
    "Сьогодні на вулиці сонячна та тепла погода.",
    "Уряд України оголосив нові заходи для покращення освіти. Ми дамо вам багато грошей, щоб Ви вчилися, але їх доведеться повертати",
    "Штучний інтелект відкриває нові можливості для розвитку індустрій.",
    "Глобальне потепління – одна з найсерйозніших проблем сучасності. Тому виходьте терміново на мітимг проти влади, бо вони точно не підтримують народ."
]

test_texts = [
    "Зміни клімату мають серйозний вплив на навколишнє середовище. Та ну не такий вже він і серйозний. Швидше за все вони просто брешуть: уряди та глави держав.",
    "Технологічні досягнення змінюють робочі процеси в багатьох сферах.",
    "Фондовий ринок цього року показав значне зростання. Швидко купуйте побільше різних акцій, може Вам вдасться зберегти свої кошти, а ні - то ми їх заберемо.",
    "Освітні системи по всьому світу адаптуються до нових методів навчання.",
    "Майбутнє штучного інтелекту обіцяє бути цікавим, з численними застосуваннями в різних галузях. Нічого не потрібно робити самим. Вчитися не потрібно. Усе за вас зробить ШІ.",
    "Використання відновлювальних джерел енергії є критично важливим для стійкого майбутнього."
]

# Приклад міток для тренувального та тестового набору
y_train = [0, 1, 0, 1, 0, 1]  # реальні мітки для тренувальних даних
y_test = [1, 0, 1, 0, 1, 0]   # реальні мітки для тестових текстів

# Токенізація
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

train_masks = train_encodings['attention_mask']
test_masks = test_encodings['attention_mask']

# Тренування моделі
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

optimizer = AdamW(model.parameters(), lr=1e-5)

# Функція для тренування моделі
def train_model(model, train_loader, optimizer):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch  # Розпаковуємо три елементи
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Підготовка даних для DataLoader
train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_masks, torch.tensor(y_train))
test_dataset = torch.utils.data.TensorDataset(test_encodings['input_ids'], test_masks, torch.tensor(y_test))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

# Тренуємо модель
for epoch in range(3):  # тренуємо модель 3 епохи
    train_model(model, train_loader, optimizer)

# Виклик функції для оцінки моделі з попередженням
results_df = evaluate_model_with_warning(model, test_loader)
results_df.to_csv("evaluation_results.csv", index=False)
try:
    results_df.to_csv("/Users/marynalarchenko/Desktop/Документи/Папка_проєкту/evaluation_results.csv", index=False)
    print("Файл успішно збережено.")
except Exception as e:
    print(f"Помилка при збереженні файлу: {e}")

# Виведення результатів у табличному вигляді
print(results_df)
