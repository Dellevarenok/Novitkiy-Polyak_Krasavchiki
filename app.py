"""
Облачный сервис оптимального сжатия данных (алгоритм Хаффмана)
с анализом метрик теории информации.

Предмет: «Основы теории информации» (ОТИ)
Запуск локально:  streamlit run app.py
Деплой:           Streamlit Community Cloud + GitHub
"""

import gzip
import heapq
import json
import math
from collections import Counter

import pandas as pd
import streamlit as st

# ==========================================================================
#  ЯДРО: собственная реализация алгоритма Хаффмана (чистый Python)
# ==========================================================================

MAGIC = b"HUFF1"  # подпись формата файла


class Node:
    """Узел бинарного кодового дерева Хаффмана."""

    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol  # None => внутренний узел
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


def build_frequencies(text: str) -> dict:
    """Частоты символов (шаг 1 алгоритма)."""
    return dict(Counter(text))


def build_tree(freq: dict):
    """Жадное построение дерева Хаффмана через приоритетную очередь.

    Сложность: O(n log n), где n — размер алфавита.
    """
    if not freq:
        return None

    counter = 0  # для детерминированного разрешения «ничьих» по частоте
    heap = []
    for symbol, f in sorted(freq.items()):
        heap.append((f, counter, Node(f, symbol)))
        counter += 1
    heapq.heapify(heap)

    # Особый случай: во всём тексте один уникальный символ
    if len(heap) == 1:
        f, _, only = heapq.heappop(heap)
        return Node(f, None, only, None)

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)  # два минимальных веса
        f2, _, n2 = heapq.heappop(heap)
        merged = Node(f1 + f2, None, n1, n2)
        heapq.heappush(heap, (merged.freq, counter, merged))
        counter += 1

    return heap[0][2]


def build_codes(root) -> dict:
    """Обход дерева: влево — «0», вправо — «1». Возвращает {символ: код}."""
    if root is None:
        return {}

    codes = {}
    stack = [(root, "")]
    while stack:
        node, prefix = stack.pop()
        if node.is_leaf:
            codes[node.symbol] = prefix or "0"  # защита от пустого кода
            continue
        if node.left is not None:
            stack.append((node.left, prefix + "0"))
        if node.right is not None:
            stack.append((node.right, prefix + "1"))
    return codes


def encode(text: str, codes: dict) -> str:
    """Текст -> строка битов."""
    return "".join(codes[ch] for ch in text)


def decode(bits: str, codes: dict) -> str:
    """Строка битов -> текст (использует префиксность кода)."""
    inverse = {code: symbol for symbol, code in codes.items()}
    out = []
    buffer = ""
    for bit in bits:
        buffer += bit
        symbol = inverse.get(buffer)
        if symbol is not None:
            out.append(symbol)
            buffer = ""
    if buffer:
        raise ValueError("Повреждённый поток битов: остался незавершённый код.")
    return "".join(out)


def pack_bits(bits: str):
    """Упаковка битовой строки в байты. Возвращает (bytes, число_доп_нулей)."""
    if not bits:
        return b"", 0
    padding = (-len(bits)) % 8
    padded = bits + "0" * padding
    value = int(padded, 2)
    return value.to_bytes(len(padded) // 8, "big"), padding


def unpack_bits(data: bytes, padding: int) -> str:
    """Байты -> битовая строка (с удалением выравнивающих нулей)."""
    if not data:
        return ""
    bits = bin(int.from_bytes(data, "big"))[2:].zfill(len(data) * 8)
    return bits[: len(bits) - padding] if padding else bits


# ==========================================================================
#  ФОРМАТ ФАЙЛА .huff  (сжатые биты + таблица восстановления в одном файле)
#  [MAGIC(5)] [len(header) 4 байта BE] [header JSON utf-8] [payload]
# ==========================================================================


def serialize(text: str, codes: dict, freq: dict) -> bytes:
    bits = encode(text, codes)
    payload, padding = pack_bits(bits)
    header = {
        "algorithm": "huffman-static",
        "padding": padding,
        "symbols": len(text),
        "codes": codes,          # таблица для восстановления
        "freq": freq,            # частоты (для повторного анализа)
    }
    raw = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return MAGIC + len(raw).to_bytes(4, "big") + raw + payload


def deserialize(blob: bytes):
    """Разбор .huff файла -> (текст, header)."""
    if not blob.startswith(MAGIC):
        raise ValueError("Это не файл формата HUFF1 (неверная подпись).")
    size = int.from_bytes(blob[5:9], "big")
    header = json.loads(blob[9 : 9 + size].decode("utf-8"))
    payload = blob[9 + size :]
    bits = unpack_bits(payload, int(header.get("padding", 0)))
    text = decode(bits, header["codes"])
    if "symbols" in header and len(text) != header["symbols"]:
        raise ValueError("Число восстановленных символов не совпадает с заголовком.")
    return text, header


# ==========================================================================
#  МЕТРИКИ ТЕОРИИ ИНФОРМАЦИИ
# ==========================================================================


def entropy(freq: dict) -> float:
    """Энтропия Шеннона H = -Σ p_i · log2(p_i), бит/символ."""
    total = sum(freq.values())
    if total == 0:
        return 0.0
    h = 0.0
    for f in freq.values():
        p = f / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def average_code_length(freq: dict, codes: dict) -> float:
    """L_ср = Σ p_i · l_i, бит/символ."""
    total = sum(freq.values())
    if total == 0:
        return 0.0
    return sum((f / total) * len(codes[s]) for s, f in freq.items())


def analyze(text: str) -> dict:
    freq = build_frequencies(text)
    codes = build_codes(build_tree(freq))
    bits = encode(text, codes)

    h = entropy(freq)
    l_avg = average_code_length(freq, codes)

    original_bytes = len(text.encode("utf-8"))
    huff_payload = math.ceil(len(bits) / 8)
    file_bytes = len(serialize(text, codes, freq))
    gzip_bytes = len(gzip.compress(text.encode("utf-8"), 9))

    ratio = (1 - huff_payload / original_bytes) * 100 if original_bytes else 0.0
    gzip_ratio = (1 - gzip_bytes / original_bytes) * 100 if original_bytes else 0.0

    return {
        "text": text,
        "freq": freq,
        "codes": codes,
        "bits": bits,
        "total_bits": len(bits),
        "entropy": h,
        "l_avg": l_avg,
        "redundancy": l_avg - h,
        "efficiency": (h / l_avg * 100) if l_avg else 0.0,
        "original_bytes": original_bytes,
        "huffman_bytes": huff_payload,
        "file_bytes": file_bytes,
        "gzip_bytes": gzip_bytes,
        "ratio": ratio,
        "gzip_ratio": gzip_ratio,
        "alphabet": len(freq),
        "fixed_bits": max(1, math.ceil(math.log2(len(freq)))) if freq else 0,
    }


def codebook_frame(freq: dict, codes: dict) -> pd.DataFrame:
    total = sum(freq.values())
    rows = []
    for symbol, f in freq.items():
        rows.append(
            {
                "Символ": repr(symbol)[1:-1] if symbol.isspace() else symbol,
                "Частота": f,
                "Вероятность p": round(f / total, 5),
                "Код Хаффмана": codes[symbol],
                "Длина кода l": len(codes[symbol]),
                "-log2(p)": round(-math.log2(f / total), 3),
            }
        )
    df = pd.DataFrame(rows).sort_values(
        ["Частота", "Символ"], ascending=[False, True]
    )
    return df.reset_index(drop=True)


# ==========================================================================
#  ИНТЕРФЕЙС STREAMLIT
# ==========================================================================

DEMO_TEXT = (
    "Теория информации изучает количественные законы передачи, хранения "
    "и обработки информации. Энтропия Шеннона задаёт нижнюю границу средней "
    "длины кода. Information theory studies the quantification of information."
)

st.set_page_config(
    page_title="Сжатие Хаффмана · ОТИ",
    page_icon="🗜️",
    layout="wide",
)

st.title("🗜️ Облачный сервис оптимального сжатия данных")
st.caption(
    "Алгоритм Хаффмана + анализ метрик теории информации · "
    "собственная реализация на чистом Python"
)

with st.sidebar:
    st.header("ℹ️ О проекте")
    st.write(
        "Учебное веб-приложение по предмету **«Основы теории информации»**.\n\n"
        "Кодирование и декодирование Хаффмана реализованы вручную: "
        "приоритетная очередь, бинарное кодовое дерево, побитовая упаковка."
    )
    st.divider()
    st.subheader("Формат файла .huff")
    st.code(
        "MAGIC (5 байт)\n"
        "длина заголовка (4 байта)\n"
        "заголовок JSON: коды + частоты\n"
        "полезная нагрузка: сжатые биты",
        language="text",
    )

tab_compress, tab_decompress = st.tabs(["📉 Сжатие и анализ", "📂 Распаковка"])

# --------------------------------------------------------------------------
#  ВКЛАДКА 1 — СЖАТИЕ И АНАЛИЗ
# --------------------------------------------------------------------------
with tab_compress:
    source = st.radio(
        "Источник данных",
        ["✍️ Ввести текст", "📎 Загрузить .txt файл"],
        horizontal=True,
    )

    text = ""
    if source == "✍️ Ввести текст":
        text = st.text_area(
            "Текст для сжатия (русский / английский, UTF-8)",
            value=DEMO_TEXT,
            height=170,
        )
    else:
        uploaded = st.file_uploader(
            "Выберите текстовый файл", type=["txt", "md", "csv", "log"]
        )
        if uploaded is not None:
            raw = uploaded.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp1251", errors="replace")
                st.warning("Файл не в UTF-8 — прочитан как CP1251.")
            st.success(f"Загружен файл **{uploaded.name}** ({len(raw)} байт)")

    run = st.button("🚀 Сжать и проанализировать", type="primary")

    if run and not text:
        st.error("Введите текст или загрузите файл.")

    if run and text:
        res = analyze(text)

        st.subheader("📊 Метрики теории информации")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Исходный размер", f"{res['original_bytes']} Б")
        c2.metric(
            "Сжатый размер",
            f"{res['huffman_bytes']} Б",
            delta=f"-{res['original_bytes'] - res['huffman_bytes']} Б",
        )
        c3.metric("Коэффициент сжатия", f"{res['ratio']:.2f} %")
        c4.metric("Размер алфавита", f"{res['alphabet']} симв.")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Энтропия H", f"{res['entropy']:.4f} бит/симв")
        c6.metric("Средняя длина L_ср", f"{res['l_avg']:.4f} бит/симв")
        c7.metric("Избыточность R = L−H", f"{res['redundancy']:.4f} бит")
        c8.metric("Эффективность η = H/L", f"{res['efficiency']:.2f} %")

        with st.expander("📐 Формулы ОТИ и расшифровка метрик"):
            st.markdown("**Энтропия Шеннона** — средняя информативность символа:")
            st.latex(r"H(X) = -\sum_{i=1}^{n} p_i \log_2 p_i \quad [\text{бит/символ}]")
            st.markdown("**Средняя длина кодового слова:**")
            st.latex(r"L_{ср} = \sum_{i=1}^{n} p_i \, l_i")
            st.markdown("**Избыточность кода и эффективность:**")
            st.latex(r"R = L_{ср} - H(X) \ge 0, \qquad \eta = \frac{H(X)}{L_{ср}}\cdot 100\%")
            st.markdown("**Теорема Шеннона о кодировании источника:**")
            st.latex(r"H(X) \le L_{ср} < H(X) + 1")
            st.markdown("**Коэффициент сжатия:**")
            st.latex(r"K = \left(1 - \frac{V_{сж}}{V_{исх}}\right)\cdot 100\%")
            st.markdown("**Неравенство Крафта** (условие существования префиксного кода):")
            st.latex(r"\sum_{i=1}^{n} 2^{-l_i} \le 1")
            kraft = sum(2 ** -len(c) for c in res["codes"].values())
            st.info(
                f"Проверка для нашего кода: Σ2^(−l) = **{kraft:.6f}** "
                "(для оптимального полного дерева = 1)."
            )

        st.subheader("📖 Кодовая книга")
        df = codebook_frame(res["freq"], res["codes"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Частоты символов (топ-25)**")
            chart = df.head(25).set_index("Символ")[["Частота"]]
            st.bar_chart(chart)
        with right:
            st.markdown("**Сравнение объёмов**")
            cmp = pd.DataFrame(
                {
                    "Метод": [
                        "Оригинал (UTF-8)",
                        "Хаффман (только биты)",
                        "Хаффман (.huff с таблицей)",
                        "Gzip (stdlib, уровень 9)",
                    ],
                    "Размер, байт": [
                        res["original_bytes"],
                        res["huffman_bytes"],
                        res["file_bytes"],
                        res["gzip_bytes"],
                    ],
                    "Экономия, %": [
                        0.0,
                        round(res["ratio"], 2),
                        round(
                            (1 - res["file_bytes"] / res["original_bytes"]) * 100, 2
                        )
                        if res["original_bytes"]
                        else 0.0,
                        round(res["gzip_ratio"], 2),
                    ],
                }
            )
            st.dataframe(cmp, use_container_width=True, hide_index=True)
            st.caption(
                "Gzip (DEFLATE) = LZ77 + Хаффман, поэтому он учитывает ещё и "
                "повторяющиеся подстроки, а не только частоты символов."
            )

        with st.expander("🔬 Битовый поток (первые 512 бит)"):
            st.code(res["bits"][:512] or "—", language="text")
            st.write(
                f"Всего бит: **{res['total_bits']}** · "
                f"при равномерном коде понадобилось бы "
                f"**{res['fixed_bits'] * len(text)}** бит "
                f"({res['fixed_bits']} бит/символ)."
            )

        blob = serialize(text, res["codes"], res["freq"])
        st.download_button(
            "💾 Скачать сжатый файл (.huff)",
            data=blob,
            file_name="compressed.huff",
            mime="application/octet-stream",
            type="primary",
        )
        st.caption(
            "В одном файле: сжатые биты + кодовая таблица для восстановления. "
            "Проверьте его на вкладке «Распаковка»."
        )

        st.download_button(
            "📄 Скачать отчёт по метрикам (CSV кодовой книги)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="codebook.csv",
            mime="text/csv",
        )

# --------------------------------------------------------------------------
#  ВКЛАДКА 2 — РАСПАКОВКА
# --------------------------------------------------------------------------
with tab_decompress:
    st.subheader("📂 Восстановление исходного текста без потерь")
    st.write(
        "Загрузите файл `.huff`, полученный на первой вкладке. "
        "Кодовая таблица хранится внутри файла, поэтому декодирование "
        "выполняется полностью автономно."
    )

    packed = st.file_uploader(
        "Сжатый файл", type=["huff", "bin", "dat"], key="decoder"
    )

    if packed is not None:
        blob = packed.read()
        try:
            restored, header = deserialize(blob)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Не удалось распаковать файл: {exc}")
        else:
            st.success("✅ Файл успешно распакован без потерь")

            d1, d2, d3 = st.columns(3)
            d1.metric("Размер архива", f"{len(blob)} Б")
            d2.metric("Восстановлено", f"{len(restored.encode('utf-8'))} Б")
            d3.metric("Символов", f"{len(restored)}")

            st.text_area("Восстановленный текст", value=restored, height=220)

            st.download_button(
                "💾 Скачать восстановленный текст (.txt)",
                data=restored.encode("utf-8"),
                file_name="restored.txt",
                mime="text/plain",
                type="primary",
            )

            with st.expander("📖 Кодовая таблица из архива"):
                codes = header["codes"]
                freq = header.get("freq") or {s: 1 for s in codes}
                st.dataframe(
                    codebook_frame(freq, codes),
                    use_container_width=True,
                    hide_index=True,
                )

            st.info(
                "Проверка целостности: количество декодированных символов "
                "совпало со значением из заголовка — сжатие **без потерь**."
            )

st.divider()
st.caption(
    "Курсовой проект по дисциплине «Основы теории информации» · "
    "Streamlit + чистый Python · развёрнуто в Streamlit Community Cloud"
)
