"""
Сжатие текста методом Хаффмана с расчётом основных метрик
по предмету «Основы теории информации».

Запуск: streamlit run app.py
"""

import gzip
import json
import math

import pandas as pd
import streamlit as st


# =====================================================================
#  АЛГОРИТМ ХАФФМАНА (собственная реализация)
# =====================================================================

def count_symbols(text):
    """Шаг 1. Считаем, сколько раз встретился каждый символ."""
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return freq


def build_codes(freq):
    """Шаг 2. Строим коды Хаффмана.

    Идея простая: берём две самые редкие группы символов и объединяем их.
    Одной дописываем в начало кода «0», другой — «1». Повторяем,
    пока не останется одна группа.
    Значит частые символы получают короткие коды, редкие — длинные.
    """
    if not freq:
        return {}

    # если в тексте всего один разный символ — даём ему код "0"
    if len(freq) == 1:
        only_symbol = list(freq.keys())[0]
        return {only_symbol: "0"}

    codes = {symbol: "" for symbol in freq}

    # группа = [суммарная частота, список символов]
    groups = [[count, [symbol]] for symbol, count in freq.items()]

    while len(groups) > 1:
        groups.sort(key=lambda g: g[0])       # сортируем по частоте
        first = groups.pop(0)                # самая редкая
        second = groups.pop(0)               # вторая по редкости

        for symbol in first[1]:
            codes[symbol] = "0" + codes[symbol]
        for symbol in second[1]:
            codes[symbol] = "1" + codes[symbol]

        groups.append([first[0] + second[0], first[1] + second[1]])

    return codes


def encode(text, codes):
    """Шаг 3. Заменяем каждый символ его кодом."""
    return "".join(codes[ch] for ch in text)


def decode(bits, codes):
    """Обратная операция: из нулей и единиц получаем текст.

    Идём по битам и копим их в буфер. Как только буфер совпал с каким-то
    кодом — записываем символ и очищаем буфер.
    Ошибки нет, потому что ни один код не является началом другого.
    """
    table = {code: symbol for symbol, code in codes.items()}
    result = ""
    buffer = ""
    for bit in bits:
        buffer += bit
        if buffer in table:
            result += table[buffer]
            buffer = ""
    return result


# =====================================================================
#  МЕТРИКИ (три формулы)
# =====================================================================

def entropy(freq):
    """Энтропия Шеннона: H = -∑ p * log2(p)."""
    total = sum(freq.values())
    h = 0.0
    for count in freq.values():
        p = count / total
        h -= p * math.log2(p)
    return h


def average_length(freq, codes):
    """Средняя длина кода: L = ∑ p * длина кода."""
    total = sum(freq.values())
    length = 0.0
    for symbol, count in freq.items():
        p = count / total
        length += p * len(codes[symbol])
    return length


# =====================================================================
#  СОХРАНЕНИЕ И ЗАГРУЗКА СЖАТОГО ФАЙЛА
#  Файл — обычный текст в формате JSON: коды + строка из 0 и 1.
# =====================================================================

def make_file(codes, bits):
    data = {"codes": codes, "bits": bits}
    return json.dumps(data, ensure_ascii=False, indent=1)


def read_file(content):
    data = json.loads(content)
    return data["codes"], data["bits"]


# =====================================================================
#  ИНТЕРФЕЙС
# =====================================================================

DEMO_TEXT = "абракадабра абракадабра abracadabra"

st.set_page_config(page_title="Сжатие Хаффмана", page_icon="🗜️")

st.title("🗜️ Сжатие текста методом Хаффмана")
st.write("Лабораторная работа по предмету «Основы теории информации»")

tab1, tab2 = st.tabs(["Сжатие и анализ", "Распаковка"])

# ---------------------- ВКЛАДКА 1 ----------------------
with tab1:
    st.subheader("1. Исходные данные")

    text = st.text_area("Введите текст", value=DEMO_TEXT, height=120)

    uploaded = st.file_uploader("Или загрузите файл .txt", type=["txt"])
    if uploaded is not None:
        text = uploaded.read().decode("utf-8")
        st.success("Файл загружен, текст из файла будет использован для сжатия")

    if st.button("Сжать текст", type="primary"):
        if not text:
            st.error("Сначала введите текст или загрузите файл.")
        else:
            freq = count_symbols(text)
            codes = build_codes(freq)
            bits = encode(text, codes)

            h = entropy(freq)
            l_avg = average_length(freq, codes)

            size_before = len(text.encode("utf-8"))       # байты до сжатия
            size_after = math.ceil(len(bits) / 8)          # байты после сжатия
            size_gzip = len(gzip.compress(text.encode("utf-8")))
            ratio = (1 - size_after / size_before) * 100

            # ---------- метрики ----------
            st.subheader("2. Результаты")

            c1, c2, c3 = st.columns(3)
            c1.metric("Было", f"{size_before} байт")
            c2.metric("Стало", f"{size_after} байт")
            c3.metric("Сжатие", f"{ratio:.1f} %")

            c4, c5, c6 = st.columns(3)
            c4.metric("Энтропия H", f"{h:.2f} бит")
            c5.metric("Средняя длина L", f"{l_avg:.2f} бит")
            c6.metric("Избыточность L − H", f"{l_avg - h:.2f} бит")

            with st.expander("Формулы, по которым считали"):
                st.write("**Энтропия Шеннона** — сколько бит информации в одном символе:")
                st.latex(r"H = -\sum p_i \log_2 p_i")
                st.write("**Средняя длина кода** — сколько бит в среднем тратим мы:")
                st.latex(r"L = \sum p_i \cdot l_i")
                st.write("**Избыточность** — насколько наш код хуже идеального:")
                st.latex(r"R = L - H")
                st.write("**Коэффициент сжатия:**")
                st.latex(r"K = \left(1 - \frac{V_{после}}{V_{до}}\right) \cdot 100\%")
                st.write(
                    "Здесь p — вероятность символа (частота делённая на длину текста), "
                    "l — длина его кода в битах."
                )

            # ---------- кодовая таблица ----------
            st.subheader("3. Таблица кодов")

            total = len(text)
            rows = []
            for symbol, count in freq.items():
                shown = symbol
                if symbol == " ":
                    shown = "(пробел)"
                elif symbol == "\n":
                    shown = "(перенос строки)"
                rows.append(
                    {
                        "Символ": shown,
                        "Частота": count,
                        "Вероятность": round(count / total, 3),
                        "Код": codes[symbol],
                    }
                )

            table = pd.DataFrame(rows).sort_values("Частота", ascending=False)
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption("Видно, что частые символы получили короткие коды, а редкие — длинные.")

            # ---------- сравнение ----------
            st.subheader("4. Сравнение размеров")
            compare = pd.DataFrame(
                {
                    "Способ": ["Обычный текст", "Наш код Хаффмана", "Архиватор Gzip"],
                    "Размер, байт": [size_before, size_after, size_gzip],
                }
            )
            st.dataframe(compare, hide_index=True, use_container_width=True)
            st.bar_chart(compare.set_index("Способ"))

            # ---------- скачивание ----------
            st.subheader("5. Сохранить результат")
            st.download_button(
                "Скачать сжатый файл",
                data=make_file(codes, bits).encode("utf-8"),
                file_name="compressed.json",
                mime="application/json",
            )
            st.caption("В файле две вещи: таблица кодов и сам код текста из нулей и единиц.")

# ---------------------- ВКЛАДКА 2 ----------------------
with tab2:
    st.subheader("Восстановление текста")
    st.write("Загрузите файл, который вы скачали на первой вкладке.")

    packed = st.file_uploader("Сжатый файл", type=["json"], key="unpack")

    if packed is not None:
        try:
            codes, bits = read_file(packed.read().decode("utf-8"))
            restored = decode(bits, codes)
        except Exception:
            st.error("Не получилось прочитать файл. Нужен файл с первой вкладки.")
        else:
            st.success("Текст восстановлен полностью, без потерь")
            st.text_area("Результат", value=restored, height=180)
            st.download_button(
                "Скачать текст (.txt)",
                data=restored.encode("utf-8"),
                file_name="restored.txt",
                mime="text/plain",
            )
