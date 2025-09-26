# bot.py
import telebot
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# ==================== КОНФИГУРАЦИЯ ====================
BOT_TOKEN = "8450391232:AAGtconwAu_Lig4gre6k05NJgXWukh6NIHU"  # Замените на токен от @BotFather
MODEL_PATH = "./my_rugpt3_finetuned"

# ==================== ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ====================
print("🔄 Загрузка модели...")

# Убираем лишние предупреждения
import warnings
warnings.filterwarnings("ignore")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✅ Модель загружена на устройство: {device}")
    
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    exit(1)

# ==================== ФУНКЦИИ ГЕНЕРАЦИИ ====================
def generate_response(question, max_length=150):
    """Генерация ответа на вопрос"""
    try:
        # Форматируем промпт как в обучении
        prompt = f"Пользователь: {question}\nСистема:"
        
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
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9,
            )
        
        # Декодирование ответа
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлекаем только ответ системы
        if "Система:" in generated_text:
            response = generated_text.split("Система:")[1].strip()
        else:
            response = generated_text.replace(prompt, "").strip()
        
        # Очищаем ответ
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
        
    except Exception as e:
        return "Извините, произошла ошибка при обработке вашего запроса."

# ==================== ТЕЛЕГРАМ БОТ ====================
print("🤖 Инициализация бота...")
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Приветственное сообщение"""
    welcome_text = """
🤖 Добро пожаловать в чат-бот техникума!

Я могу ответить на вопросы о:
• Расписании и графике работы
• Поступлении и переводе
• Учебном процессе
• Библиотеке и документах

Просто задайте ваш вопрос!

Примеры:
- "Когда работает юрист?"
- "Как перевестись на другую специальность?"
- "Какие языки программирования изучают?"
    """
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    """Статус бота"""
    status_text = f"""
📊 Статус системы:
• Устройство: {device}
• Модель: RuGPT3 (дообученная)
• Бот активен ✅
    """
    bot.reply_to(message, status_text)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Обработка всех сообщений"""
    try:
        # Показываем что бот печатает
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Генерируем ответ
        response = generate_response(message.text)
        
        # Отправляем ответ
        bot.reply_to(message, response)
        
        # Логируем только краткую информацию
        user = message.from_user.username or message.from_user.first_name
        print(f"👤 {user}: {message.text[:50]}...")
        print(f"🤖 Бот: {response[:50]}...")
        print("-" * 50)
        
    except Exception as e:
        bot.reply_to(message, "⚠️ Произошла ошибка. Попробуйте еще раз.")
        print(f"❌ Ошибка: {e}")

# ==================== ЗАПУСК БОТА ====================
if __name__ == "__main__":
    print("=" * 40)
    print("🚀 ТЕЛЕГРАМ БОТ ЗАПУЩЕН")
    print(f"📍 Модель: {MODEL_PATH}")
    print(f"⚙️ Устройство: {device}")
    print("📱 Ожидаем сообщения...")
    print("=" * 40)
    
    try:
        # ИСПРАВЛЕННАЯ СТРОКА: убрал long_pending
        bot.infinity_polling(timeout=60, long_polling_timeout=30)
        
    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен")
    except Exception as e:
        print(f"❌ Ошибка бота: {e}")
