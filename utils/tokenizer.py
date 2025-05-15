import re

def tokenize(text: str) -> list[str]:
    text = text.lower()

    # Заменяем спецструктуры на временные плейсхолдеры
    text = re.sub(r"https?://\S+", " __URL__ ", text)
    text = re.sub(r"pic\.twitter\.com/\S+", " __PIC__ ", text)
    text = re.sub(r"@\w+", " __MENTION__ ", text)
    text = re.sub(r"#\w+", " __HASHTAG__ ", text)

    # Удаляем всю пунктуацию (кроме подстановок)
    text = re.sub(r"[^\w\s]", "", text)

    # Возвращаем подстановки в виде токенов
    text = text.replace(" __URL__ ", " <URL> ")
    text = text.replace(" __PIC__ ", " <PIC> ")
    text = text.replace(" __MENTION__ ", " <MENTION> ")
    text = text.replace(" __HASHTAG__ ", " <HASHTAG> ")

    # Финальная токенизация
    tokens = text.strip().split()
    return tokens




