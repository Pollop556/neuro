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

# ==================== КОНФИГУРАЦИЯ ====================
MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
CSV_FILE_PATH = "data.csv"           # Убедись, что файл в той же папке
OUTPUT_DIR = "./my_rugpt3_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./logs", exist_ok=True)

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

# ==================== ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ====================
def load_and_format_data(csv_path):
    print("🔍 Чтение CSV файла...")
    try:
        df = pd.read_csv(
            csv_path,
            header=None,
            names=['question', 'answer'],
            quoting=1,               # Обработка кавычек
            escapechar='\\',
            on_bad_lines='warn',     # Покажет ошибки, но не сломается
            encoding='utf-8'
        )
        print(f"✅ Прочитано {len(df)} строк")
    except Exception as e:
        print(f"❌ Ошибка при чтении CSV: {e}")
        return None

    dialog_examples = []
    skipped_no_text = 0
    skipped_english = 0

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

    print(f"✅ Создано {len(dialog_examples)} диалогов для обучения")
    print(f"ℹ️ Пропущено пустых: {skipped_no_text}, английских: {skipped_english}")

    dataset = Dataset.from_dict({"text": dialog_examples})
    return dataset.train_test_split(test_size=0.1, seed=42)

# ==================== ОСНОВНОЙ ПРОЦЕСС ====================
def main():
    print("=" * 60)
    print("🚀 ЗАПУЩЕН ПРОЦЕСС ДООБУЧЕНИЯ RuGPT3")
    print("Модель: ai-forever/rugpt3small_based_on_gpt2")
    print("Формат: Пользователь → Система")
    print("=" * 60)

    # Загрузка данных
    dataset = load_and_format_data(CSV_FILE_PATH)
    if dataset is None:
        return

    print(f"📊 Обучающих примеров: {len(dataset['train'])}")
    print(f"📊 Валидационных: {len(dataset['test'])}")

    # Загрузка модели и токенизатора
    print("\n⏬ Загрузка модели и токенизатора...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Безопасное добавление pad_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.resize_token_embeddings(len(tokenizer))  # Важно после добавления токенов!
        
        print("✅ Модель и токенизатор загружены")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    # Токенизация
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,          # Оптимизировано под Q&A
            padding="max_length",
            return_tensors=None,
        )

    print("🔤 Токенизация данных...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )

    # Коллатор
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    # Параметры обучения — оптимизированы под 817 строк
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=4,                    # Оптимально для малого датасета
        per_device_train_batch_size=8,         # Увеличено, если GPU позволяет
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,         # Эффективный batch = 16
        learning_rate=2e-5,                    # Более консервативный lr
        warmup_steps=100,                      # Меньше, т.к. данных мало
        weight_decay=0.01,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=50,                      # Частое логирование
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,                    # Ограничение чекпоинтов
        report_to="none",
        fp16=torch.cuda.is_available(),        # Авто-включение для GPU
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        disable_tqdm=False,
        seed=42,
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

    # Обучение
    print("\n🔥 НАЧИНАЕМ ОБУЧЕНИЕ...")
    try:
        trainer.train()
        print("\n💾 Сохранение модели...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"🎉 УСПЕШНО: модель сохранена в '{OUTPUT_DIR}'")
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()