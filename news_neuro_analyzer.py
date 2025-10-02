"""
Нейроассистент для аналитики новостных потоков:
Анализ на основе имеющихся CSV-файлов (реакции, сообщения, описания каналов).
"""

import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Инициализация NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('russian'))

# Файлы данных
ART_DESC_CSV = 'materials/art_channel_descriptions.csv'
IT_DESC_CSV = 'materials/IT_channel_descriptions.csv'
ART_ENHANCED_CSV = 'materials/art_enhanced_messages_with_stats_0825.csv'
IT_ENHANCED_CSV = 'materials/it_enhanced_messages_with_stats_0825_fixed.csv'
ART_REACTIONS_CSV = 'materials/art_reactions_detailed_0825.csv'
IT_REACTIONS_CSV = 'materials/IT_reactions_detailed_0825.csv'


def load_channels_from_descriptions():
    """Загружает список каналов из файлов описаний, используя ID."""
    channels_set = set()
    for csv_path in [ART_DESC_CSV, IT_DESC_CSV]:
        try:
            df = pd.read_csv(csv_path)
            if 'ID' in df.columns:
                channels_set.update(df['ID'].astype(str).unique())
            else:
                print(f"Колонка 'ID' не найдена в {csv_path}")
        except FileNotFoundError:
            print(f"Файл {csv_path} не найден. Пропускаем.")
        except Exception as e:
            print(f"Ошибка при загрузке {csv_path}: {e}")
    return list(channels_set)


def load_channels_from_reactions():
    """Загружает список уникальных каналов из файлов реакций, используя Channel_ID."""
    channels_set = set()
    for csv_path in [ART_REACTIONS_CSV, IT_REACTIONS_CSV]:
        try:
            df = pd.read_csv(csv_path)
            if 'Channel_ID' in df.columns:
                channels_set.update(df['Channel_ID'].astype(str).unique())
            else:
                print(f"Колонка 'Channel_ID' не найдена в {csv_path}")
        except FileNotFoundError:
            print(f"Файл {csv_path} не найден. Пропускаем.")
        except Exception as e:
            print(f"Ошибка при загрузке {csv_path}: {e}")
    return list(channels_set)


def load_enhanced_messages():
    """Загружает сообщения с дополнительными метриками"""
    try:
        art_enhanced = pd.read_csv(ART_ENHANCED_CSV)
    except FileNotFoundError:
        print(f"Файл {ART_ENHANCED_CSV} не найден. Пропускаем.")
        art_enhanced = pd.DataFrame()

    try:
        it_enhanced = pd.read_csv(IT_ENHANCED_CSV)
    except FileNotFoundError:
        print(f"Файл {IT_ENHANCED_CSV} не найден. Пропускаем.")
        it_enhanced = pd.DataFrame()

    if art_enhanced.empty and it_enhanced.empty:
        return pd.DataFrame()
    elif art_enhanced.empty:
        return it_enhanced
    elif it_enhanced.empty:
        return art_enhanced
    else:
        df = pd.concat([art_enhanced, it_enhanced], ignore_index=True)
        # Переименуем колонки для единообразия
        # 'ID' -> 'channel_id'
        # 'Message_ID' -> 'message_id'
        # 'Reactions_Count' -> 'reactions_count'
        # 'Comments_Count' -> 'comments_count'
        df.rename(columns={
            'ID': 'channel_id',
            'Message_ID': 'message_id',
            'Reactions_Count': 'reactions_count',
            'Comments_Count': 'comments_count'
        }, inplace=True)
        return df


def load_reactions():
    """Загружает детальные реакции"""
    try:
        art_reactions = pd.read_csv(ART_REACTIONS_CSV)
    except FileNotFoundError:
        print(f"Файл {ART_REACTIONS_CSV} не найден. Пропускаем.")
        art_reactions = pd.DataFrame()

    try:
        it_reactions = pd.read_csv(IT_REACTIONS_CSV)
    except FileNotFoundError:
        print(f"Файл {IT_REACTIONS_CSV} не найден. Пропускаем.")
        it_reactions = pd.DataFrame()

    if art_reactions.empty and it_reactions.empty:
        return pd.DataFrame()
    elif art_reactions.empty:
        return it_reactions
    elif it_reactions.empty:
        return art_reactions
    else:
        df = pd.concat([art_reactions, it_reactions], ignore_index=True)
        # Переименуем колонки для единообразия
        # 'Channel_ID' -> 'channel_id'
        # 'Message_ID' -> 'message_id'
        # 'Reaction_Count' -> 'count'
        # 'Reaction_Emoji' -> 'reaction_type'
        df.rename(columns={
            'Channel_ID':     'channel_id',
            'Message_ID':     'message_id',
            'Reaction_Count': 'count',
            'Reaction_Emoji': 'reaction_type'
        }, inplace=True)
        return df


def compute_trend_with_reactions(enhanced_df, reactions_df):
    """Вычисляет тренды с учётом реакций и комментариев"""
    if enhanced_df.empty or reactions_df.empty:
        print("Один из DataFrame (enhanced или reactions) пуст. Пропускаем объединение.")
        return pd.DataFrame()

    # enhanced_df уже имеет колонки 'channel_id' и 'message_id' благодаря load_enhanced_messages
    # reactions_df уже имеет колонки 'channel_id' и 'message_id' благодаря load_reactions
    # Объединяем по этим колонкам
    merged = enhanced_df.merge(reactions_df, on=['channel_id', 'message_id'], how='left')
    # Суммируем реакции и комментарии для метрики вовлечённости
    merged['engagement'] = merged.get('reactions_count', 0) + merged.get('comments_count', 0)
    return merged



# ... (весь остальной код без изменений, включая функции load_... и compute_trend_with_reactions) ...

def main():
    print("--- Запуск анализа на основе CSV-файлов ---")

    # --- НОВОЕ: Выбор категории ---
    while True:
        category_choice = input("Выберите категорию для анализа (введите 'ART' или 'IT'): ").strip().upper()
        if category_choice in ['ART', 'IT']:
            break
        else:
            print("Неверный ввод. Пожалуйста, введите 'ART' или 'IT'.")

    # Определяем файлы в зависимости от выбора
    if category_choice == 'ART':
        desc_csv = ART_DESC_CSV
        enhanced_csv = ART_ENHANCED_CSV
        reactions_csv = ART_REACTIONS_CSV
        print(f"\n--- Анализ для категории: {category_choice} ---")
    else: # category_choice == 'IT'
        desc_csv = IT_DESC_CSV
        enhanced_csv = IT_ENHANCED_CSV
        reactions_csv = IT_REACTIONS_CSV
        print(f"\n--- Анализ для категории: {category_choice} ---")

    # Загрузка каналов (только для выбранной категории)
    print(f"\nЗагрузка списка каналов из {desc_csv}...")
    channels_from_desc = load_channels_from_descriptions_for_category(desc_csv)
    print(f"Найдено {len(channels_from_desc)} каналов из описаний ({category_choice}).")

    # Загрузка каналов из реакций (только для выбранной категории)
    print(f"\nЗагрузка списка каналов из {reactions_csv}...")
    channels_from_react = load_channels_from_reactions_for_category(reactions_csv)
    print(f"Найдено {len(channels_from_react)} каналов из реакций ({category_choice}).")

    all_channels = list(set(channels_from_desc + channels_from_react))
    print(f"Всего уникальных каналов ({category_choice}): {len(all_channels)}")

    # Загрузка и анализ (только для выбранной категории)
    print(f"\nЗагрузка реакций из {reactions_csv}...")
    reactions_df = load_reactions_for_category(reactions_csv) # Загружаем только для выбранной категории
    print(f"Загружено {len(reactions_df)} записей реакций ({category_choice}).")

    print(f"\nЗагрузка сообщений (enhanced) из {enhanced_csv}...")
    enhanced_df = load_enhanced_messages_for_category(enhanced_csv) # Загружаем только для выбранной категории
    print(f"Загружено {len(enhanced_df)} записей сообщений ({category_choice}).")

    # --- НОВОЕ: Анализ важности и формирование дайджеста ---
    if not reactions_df.empty:
        print(f"\n--- Анализ важности новостей на основе реакций ({category_choice}) ---")
        reaction_weights = {
            '👍': 1, '❤': 3, '🔥': 5, '👏': 2,
            '😍': 3, '😂': 1, '😮': 2, '😢': -1,
            '😡': -2, '🤯': 4
        }

        # Применим веса
        reactions_df['weight'] = reactions_df['reaction_type'].map(reaction_weights).fillna(1)
        reactions_df['weighted_score'] = reactions_df['count'] * reactions_df['weight']

        # Сгруппируем по message_id и просуммируем взвешенные баллы
        importance_scores = reactions_df.groupby(['channel_id', 'message_id'])['weighted_score'].sum().reset_index()
        importance_scores.rename(columns={'weighted_score': 'importance'}, inplace=True)

        print(f"Топ-10 сообщений по важности ({category_choice}):")
        top_10_importance = importance_scores.nlargest(10, 'importance')
        print(top_10_importance)

        # --- Формирование дайджеста ---
        print(f"\n--- Формирование дайджеста для топ-10 важных новостей ({category_choice}) ---")
        # Объединим top_10_importance с enhanced_df, чтобы получить текст и метаданные
        digest_df = top_10_importance.merge(enhanced_df, on=['channel_id', 'message_id'], how='left')

        # Выберем нужные колонки для дайджеста
        available_cols = enhanced_df.columns.tolist()
        # print(f"Доступные колонки в enhanced_df ({category_choice}): {available_cols}") # Опционально для отладки
        desired_digest_cols = ['channel_id', 'message_id', 'importance', 'Name', 'Date', 'Original', 'Content_Type', 'Views', 'Forwards', 'Comments']
        cols_to_select = [col for col in desired_digest_cols if col in available_cols]
        if 'importance' not in cols_to_select:
            cols_to_select = ['importance'] + cols_to_select

        if cols_to_select:
            digest_final = digest_df[cols_to_select]
            print(f"\nДайджест топ-10 важных новостей ({category_choice}):")
            print(digest_final)

            # --- НОВОЕ: Сохранение дайджеста с именем файла, зависящим от категории ---
            digest_filename = f'digest_top_10_important_news_{category_choice.lower()}.csv'
            digest_final.to_csv(digest_filename, index=False)
            print(f"\nДайджест сохранен в '{digest_filename}'")

        else:
            print(f"Не найдено подходящих колонок в enhanced_df ({category_choice}) для формирования дайджеста.")
            print("Выводим только таблицу важности:")
            print(top_10_importance)

            # --- НОВОЕ: Сохранение базового дайджеста (важность) с именем файла, зависящим от категории ---
            basic_digest_filename = f'digest_top_10_important_news_basic_{category_choice.lower()}.csv'
            top_10_importance.to_csv(basic_digest_filename, index=False)
            print(f"Базовый дайджест (важность) сохранен в '{basic_digest_filename}'")

    else:
        print(f"Файл реакций для {category_choice} пуст или не содержит данных. Пропускаем анализ важности и дайджест.")

    # --- (остальной код анализа трендов и описаний без изменений, но только для выбранной категории) ---
    # Анализ трендов (объединение данных) - только если оба файла (enhanced и reactions) для выбранной категории не пусты
    if not enhanced_df.empty and not reactions_df.empty:
        print(f"\n--- Объединение данных для анализа трендов ({category_choice}) ---")
        trend_df = compute_trend_with_reactions(enhanced_df, reactions_df)
        if not trend_df.empty:
            print(f"Объединено {len(trend_df)} записей для тренд-анализа ({category_choice}).")
            # Проверим, есть ли колонка 'Name' (название канала) в объединённом DataFrame
            if 'Name' in trend_df.columns:
                # Группируем по 'Name' из trend_df, которое является названием канала
                top_engaged_channels = trend_df.groupby('Name')['engagement'].sum().nlargest(10)
                print(f"Топ-10 каналов по общему вовлечению (реакции + комментарии) ({category_choice}):")
                print(top_engaged_channels)
            else:
                print(f"Колонка 'Name' не найдена в объединённом DataFrame ({category_choice}). Невозможно сгруппировать по названию канала.")
        else:
            print(f"Не удалось объединить данные для анализа трендов ({category_choice}).")

    # Загрузка описаний каналов (только для выбранной категории)
    print(f"\n--- Загрузка описаний каналов ({category_choice}) ---")
    try:
        desc_df = pd.read_csv(desc_csv).set_index('ID')
        print(f"Описания каналов ({category_choice}) загружены.")
    except FileNotFoundError as e:
        print(f"Файл описания {desc_csv} не найден: {e}")
    except KeyError as e:
        print(f"Колонка 'ID' не найдена при загрузке описаний ({category_choice}): {e}")

    print(f"\n--- Завершение анализа для {category_choice} ---")


# --- НОВЫЕ: Функции загрузки для конкретной категории ---
# Эти функции повторяют логику основных функций, но работают с одним CSV-файлом
def load_channels_from_descriptions_for_category(csv_path):
    """Загружает список каналов из файла описаний, используя ID."""
    channels_set = set()
    try:
        df = pd.read_csv(csv_path)
        if 'ID' in df.columns:
            channels_set.update(df['ID'].astype(str).unique())
        else:
            print(f"Колонка 'ID' не найдена в {csv_path}")
    except FileNotFoundError:
        print(f"Файл описания {csv_path} не найден. Пропускаем.")
    except Exception as e:
        print(f"Ошибка при загрузке {csv_path}: {e}")
    return list(channels_set)

def load_channels_from_reactions_for_category(csv_path):
    """Загружает список уникальных каналов из файла реакций, используя Channel_ID."""
    channels_set = set()
    try:
        df = pd.read_csv(csv_path)
        if 'Channel_ID' in df.columns:
            channels_set.update(df['Channel_ID'].astype(str).unique())
        else:
            print(f"Колонка 'Channel_ID' не найдена в {csv_path}")
    except FileNotFoundError:
        print(f"Файл {csv_path} не найден. Пропускаем.")
    except Exception as e:
        print(f"Ошибка при загрузке {csv_path}: {e}")
    return list(channels_set)

def load_enhanced_messages_for_category(csv_path):
    """Загружает сообщения с дополнительными метриками из одного файла"""
    try:
        df = pd.read_csv(csv_path)
        # Переименуем колонки для единообразия
        df.rename(columns={
            'ID': 'channel_id',
            'Message_ID': 'message_id',
            'Reactions_Count': 'reactions_count',
            'Comments_Count': 'comments_count'
        }, inplace=True)
        return df
    except FileNotFoundError:
        print(f"Файл {csv_path} не найден. Пропускаем.")
        return pd.DataFrame()

def load_reactions_for_category(csv_path):
    """Загружает детальные реакции из одного файла"""
    try:
        df = pd.read_csv(csv_path)
        df.rename(columns={
            'Channel_ID':     'channel_id',
            'Message_ID':     'message_id',
            'Reaction_Count': 'count',
            'Reaction_Emoji': 'reaction_type'
        }, inplace=True)
        return df
    except FileNotFoundError:
        print(f"Файл {csv_path} не найден. Пропускаем.")
        return pd.DataFrame()


if __name__ == '__main__':
    main()
