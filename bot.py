# bot.py
import telebot
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import logging
from datetime import datetime

# ==================== КОНФИГУРАЦИЯ ====================
BOT_TOKEN = "8450391232:AAGtconwAu_Lig4gre6k05NJgXWukh6NIHU"  # Замените на токен от @BotFather
MODEL_PATH = "./my_rugpt3_finetuned"  # Путь к вашей дообученной модели

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== ИНИЦИАЛИЗАЦИЯ ====================
try:
    # Загрузка модели и токенизатора
    print("🔄 Загрузка модели...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    
    # Проверка устройства
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # Переводим модель в режим оценки
    
    print(f"✅ Модель загружена на устройство: {device}")
    
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")
    exit(1)

# Инициализация бота
try:
    bot = telebot.TeleBot(BOT_TOKEN)
    print("✅ Бот инициализирован")
except Exception as e:
    logger.error(f"❌ Ошибка инициализации бота: {e}")
    exit(1)

# ==================== ФУНКЦИИ ОБРАБОТКИ ====================
def clean_text(text):
    """Очистка текста от мусора"""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'http\S+', '', text)           # Ссылки
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\+\"\'\(\)\—]', ' ', text)
    text = re.sub(r'\s+', ' ', text)              # Множественные пробелы
    return text.strip()

def generate_response(question, max_length=150, temperature=0.7):
    """
    Генерация ответа на вопрос пользователя
    """
    try:
        # Очищаем вопрос
        cleaned_question = clean_text(question)
        
        # Форматируем промпт как в обучении
        prompt = f"Пользователь: {cleaned_question}\nСистема:"
        
        # Токенизация
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(device)
        
        # Генерация ответа
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Штраф за повторения
                top_p=0.9,              # Nucleus sampling
                top_k=50,               # Top-k sampling
            )
        
        # Декодирование ответа
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлекаем только ответ системы (после "Система:")
        if "Система:" in generated_text:
            response = generated_text.split("Система:")[1].strip()
        else:
            # Если формат не сохранился, возвращаем все после промпта
            response = generated_text.replace(prompt, "").strip()
        
        # Очищаем ответ от возможных артефактов
        response = re.sub(r'[^\.\!\?]$', '.', response)  # Добавляем точку если нет пунктуации
        response = re.sub(r'\s+', ' ', response).strip()
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса."

# ==================== ОБРАБОТЧИКИ КОМАНД ====================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Обработчик команд /start и /help"""
    welcome_text = """
🤖 Добро пожаловать в чат-бот техникума!

Я могу ответить на вопросы о:
• Расписании и графике работы
• Поступлении и переводе
• Учебном процессе
• Библиотеке и документах
• И многом другом

Просто задайте ваш вопрос в свободной форме!

📋 Примеры вопросов:
- "Когда работает юрист?"
- "Как перевестись на другую специальность?"
- "Какие языки программирования изучают?"

Команды:
/start - начать работу
/help - помощь
/status - статус бота
    """
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    """Статус бота и модели"""
    status_text = f"""
📊 Статус системы:
• Модель: RuGPT3 (дообученная)
• Устройство: {device}
• Время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
• Память GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB (если доступно)

✅ Бот работает нормально
    """
    bot.reply_to(message, status_text)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Обработка всех текстовых сообщений"""
    try:
        user_id = message.from_user.id
        username = message.from_user.username or "Unknown"
        question = message.text
        
        logger.info(f"👤 Вопрос от {username} ({user_id}): {question}")
        
        # Показываем что бот печатает
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Генерируем ответ
        response = generate_response(question)
        
        # Логируем ответ
        logger.info(f"🤖 Ответ для {username}: {response[:100]}...")
        
        # Отправляем ответ
        bot.reply_to(message, response)
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки сообщения: {e}")
        bot.reply_to(message, "⚠️ Произошла ошибка. Попробуйте задать вопрос еще раз.")

# ==================== ОБРАБОТКА ОШИБОК ====================
def error_handler(update, context):
    """Глобальный обработчик ошибок"""
    logger.error(f"Ошибка в update {update}: {context.error}")

# ==================== ЗАПУСК БОТА ====================
if __name__ == "__main__":
    print("=" * 50)
    print("🤖 ЗАПУСК ТЕЛЕГРАМ БОТА")
    print(f"📍 Модель: {MODEL_PATH}")
    print(f"⚙️ Устройство: {device}")
    print("📱 Бот готов к работе...")
    print("=" * 50)
    
    try:
        # Запуск бота
        bot.infinity_polling(timeout=60, long_pending=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка бота: {e}")
