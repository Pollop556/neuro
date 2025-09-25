# train.py
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch
import re
import os
import numpy as np
from sklearn.metrics import accuracy_score
import logging

# ==================== КОНФИГУРАЦИЯ ====================
MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
CSV_FILE_PATH = "data.csv"  # Используем оригинальный файл
OUTPUT_DIR = "./my_rugpt3_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверка устройства
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# ==================== ОЧИСТКА ТЕКСТА ====================
def clean_text(text):
    """Очистка текста от мусора"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)           # HTML-теги
    text = re.sub(r'http\S+', '', text)           # Ссылки
    text = re.sub(r'[“”«»]', '"', text)           # Нормализация кавычек
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\+\"\'\(\)\—]', ' ', text)  # Только базовая пунктуация
    text = re.sub(r'\s+', ' ', text)              # Множественные пробелы
    text = re.sub(r'\?+', '?', text)              # ?? -> ?
    text = re.sub(r'\!+', '!', text)              # !! -> !
    text = re.sub(r':\s*$', '', text)             # Убрать двоеточие в конце
    text = re.sub(r'^[А-Я][а-я]+\s+[А-Я]\.[А-Я]\.:', 'Оператор:', text)  # Персональные данные
    return text.strip()

def contains_russian(text):
    """Проверяет, содержит ли текст кириллицу"""
    return bool(re.search(r'[а-яА-Я]', text))

def calculate_average_length(dataset):
    """Рассчитывает среднюю длину текстов для настройки max_length"""
    lengths = []
    for example in dataset:
        text = example['text'] if 'text' in example else example
        lengths.append(len(text.split()))
    return np.mean(lengths)

# ==================== ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ====================
def load_and_format_data(csv_path):
    print("🔍 Чтение CSV файла...")
    try:
        df = pd.read_csv(
            csv_path,
            header=None,
            names=['question', 'answer'],
            quoting=1,
            encoding='utf-8',
            on_bad_lines='skip'  # Пропускаем проблемные строки
        )
        print(f"✅ Прочитано {len(df)} строк")
        
        # Проверяем первые несколько строк
        print("\n📋 Примеры данных:")
        for i in range(min(3, len(df))):
            print(f"  {i+1}. В: {df.iloc[i]['question'][:60]}...")
            print(f"     О: {df.iloc[i]['answer'][:60]}...")
            print()
            
    except Exception as e:
        print(f"❌ Ошибка при чтении CSV: {e}")
        return None

    dialog_examples = []
    skipped_no_text = 0
    skipped_english = 0
    length_stats = []

    for _, row in df.iterrows():
        q = clean_text(row['question'])
        a = clean_text(row['answer'])

        # Пропуск пустых
        if not q or not a:
            skipped_no_text += 1
            continue

        # Фильтр: только русские вопросы
        if not contains_russian(q):
            skipped_english += 1
            continue

        # Формат диалога
        dialog = f"Пользователь: {q}\nСистема: {a}"
        dialog_examples.append(dialog)
        length_stats.append(len(dialog.split()))

    print(f"✅ Создано {len(dialog_examples)} диалогов для обучения")
    print(f"ℹ️ Пропущено пустых: {skipped_no_text}, английских: {skipped_english}")
    
    # Статистика по длине текстов
    if length_stats:
        avg_length = np.mean(length_stats)
        max_length = np.max(length_stats)
        print(f"📊 Статистика длины: средняя = {avg_length:.1f}, максимальная = {max_length}")
        print(f"💡 Рекомендуемый max_length: {int(max_length * 1.2)}")

    dataset = Dataset.from_dict({"text": dialog_examples})
    return dataset.train_test_split(test_size=0.15, seed=42, shuffle=True)

# ==================== ОСНОВНОЙ ПРОЦЕСС ====================
def main():
    print("=" * 60)
    print("🚀 ЗАПУЩЕН ПРОЦЕСС ДООБУЧЕНИЯ RuGPT3")
    print(f"Модель: {MODEL_NAME}")
    print(f"Данные: {CSV_FILE_PATH}")
    print("Формат: Пользователь → Система")
    print("=" * 60)

    # Загрузка данных
    dataset = load_and_format_data(CSV_FILE_PATH)
    if dataset is None:
        return

    print(f"📊 Обучающих примеров: {len(dataset['train'])}")
    print(f"📊 Валидационных: {len(dataset['test'])}")

    # Анализ средней длины для настройки max_length
    avg_length = calculate_average_length(dataset['train'])
    max_length = min(256, int(avg_length * 1.5))  # Увеличили лимит
    print(f"🔤 Автоматически установлен max_length: {max_length}")

    # Загрузка модели и токенизатора
    print("\n⏬ Загрузка модели и токенизатора...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Правильное добавление pad_token для GPT
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Загружаем модель в правильном формате
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32  # Используем float32 для стабильности
        )
        
        print("✅ Модель и токенизатор загружены")
        print(f"💡 Размер словаря: {len(tokenizer)}")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    # Токенизация с динамическим max_length
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        return tokenized

    print("🔤 Токенизация данных...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=2,  # Уменьшили для стабильности
        remove_columns=["text"]
    )

    # Коллатор
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    # ОПТИМИЗИРОВАННЫЕ параметры обучения
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,         # Уменьшили batch_size
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,         # Увеличили для компенсации
        learning_rate=2e-5,                    # Вернули оригинальный lr
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        fp16=False,  # ВЫКЛЮЧИЛИ FP16 для стабильности
        dataloader_num_workers=1,              # Уменьшили для Colab
        dataloader_pin_memory=False,           # Выключили для стабильности
        remove_unused_columns=True,
        disable_tqdm=False,
        seed=42,
        prediction_loss_only=True,
        gradient_checkpointing=True,           # Включили для экономии памяти
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Предварительная оценка до обучения
    print("\n📈 ПРЕДВАРИТЕЛЬНАЯ ОЦЕНКА...")
    try:
        eval_results = trainer.evaluate()
        print(f"📊 Начальный loss: {eval_results['eval_loss']:.4f}")
    except Exception as e:
        print(f"⚠️ Не удалось выполнить предварительную оценку: {e}")

    # Обучение
    print("\n🔥 НАЧИНАЕМ ОБУЧЕНИЕ...")
    try:
        train_result = trainer.train()
        
        # Сохранение финальной модели
        print("\n💾 Сохранение модели...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Логирование результатов
        metrics = train_result.metrics
        print(f"🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"📊 Final train loss: {metrics.get('train_loss', 'N/A')}")
        print(f"📊 Final eval loss: {metrics.get('eval_loss', 'N/A')}")
        print(f"💾 Модель сохранена в '{OUTPUT_DIR}'")
        
        # Вывод примера генерации
        print("\n🧪 ТЕСТ ГЕНЕРАЦИИ...")
        test_question = "Как получить дубликат диплома?"
        inputs = tokenizer(f"Пользователь: {test_question}\nСистема:", return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Вопрос: {test_question}")
        print(f"Ответ модели: {generated_text}")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()