import math
import numpy as np
from PIL import Image
import os
import logging.config
import yaml
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class JpegCompressor:
    """Класс JPEG-компрессора.
    
    Attributes:
        _original_pixels (np.ndarray): Исходное изображение в виде матрицы пикселей
        _pixels (np.ndarray): Текущее состояние изображения в процессе обработки
        original_image_path (str): Путь к исходному файлу изображения
        quality (int): Качество сжатия от 1 до 100
    """
    
    
# private 
    
    def __init__(self):
        """Первоначальная инициализация состояния объекта"""
        
        self.logger = self._setup_logger(os.getenv("DISABLE_FILE_LOGGING") == "1")
        
        self.logger.info("__init__: <start>")
        
        self._original_pixels = None
        self._compressed_pixels = None
        self.quality = None
        self.DCT_MATRIX = self._create_dctII_matrix(8)
        self.STANDARD_LUMINANCE_QUANT_TABLE = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68,109,103, 77],
            [24, 35, 55, 64, 81,104,113, 92],
            [49, 64, 78, 87,103,121,120,101],
            [72, 92, 95, 98,112,100,103, 99]], dtype=np.uint8)

        self.STANDARD_CHROMINANCE_QUANT_TABLE = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.uint8)
        
        # для категорий DC: 0..11
        self.dc_freq = {
                'Y' : {},
                'C' : {}
            }
        # 162 возможных кода AC (по JPEG)
        self.ac_freq = {
                'Y' : {},
                'C' : {}
            }
        
        self.logger.info("__init__: <end>")
        
    def _setup_logger(self, disable_file=False):
        
            # Загружаем конфигурацию YAML
            with open("logging.yaml", "r") as f:
                config = yaml.safe_load(f)

            if disable_file:
                # Удаляем file-хендлер и ссылки на него
                config['handlers'].pop('file', None)
                for logger in config.get('loggers', {}).values():
                    if 'file' in logger.get('handlers', []):
                        logger['handlers'].remove('file')

            else:
                # Создаём папку logs, если её нет
                os.makedirs("logs", exist_ok=True)

                # Формируем поддиректорию по дате, например logs_2025-10-04
                today = datetime.now().strftime("%Y-%m-%d")
                dated_log_dir = os.path.join("logs", f"logs_{today}")
                os.makedirs(dated_log_dir, exist_ok=True)

                # Базовое имя файла
                base_name = f"debug_log_{today}"
                ext = ".txt"

                # Ищем свободное имя: debug_log_2025-10-04.txt, _1.txt, _2.txt и т.д.
                i = 0
                while True:
                    suffix = f"_{i}"
                    filename = os.path.join(dated_log_dir, f"{base_name}{suffix}{ext}")
                    if not os.path.exists(filename):
                        break
                    i += 1

                # Подставляем имя файла в конфиг
                config['handlers']['file']['filename'] = filename

            # Применяем конфигурацию
            logging.config.dictConfig(config)
            return logging.getLogger("JPEGCompressor")

    
    def log_step(func):
        def wrapper(self, *args, **kwargs):
            name = func.__name__
            self.logger.info(f"{name}: <start>")
            try:
                result = func(self, *args, **kwargs)
                self.logger.info(f"{name}: <end>")
                return result
            except Exception as e:
                self.logger.error(f"{name}: FAILED — {e}")
                raise
        return wrapper
        
# protected

    @log_step
    def _reset_state(self):
        self._original_pixels = None
        self._compressed_pixels = None
        self.quality = None

    @log_step
    def _create_dctII_matrix(self, N):
        """Создание 2D DCT-матрицы"""
        dct_mat = np.zeros((N, N))
        for k in range(N):
            for n in range(N):
                coef = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
                dct_mat[k, n] = coef * np.cos(np.pi * (2*n + 1) * k / (2 * N))
        return dct_mat.astype('float32')
    
    @log_step
    def _build_huffman_table(self, freq_dict):
        """Строит Хаффман-таблицу из словаря частот"""
        import heapq

        heap = [[weight, [symbol, ""]] for symbol, weight in freq_dict.items() if weight > 0]
        heapq.heapify(heap)

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        return {symbol: code for symbol, code in heap[0][1:]}

    @log_step
    def _generate_all_huffman_tables(self):
        """Генерирует 4 таблицы Хаффмана: DC_Y, DC_C, AC_Y, AC_C"""

        # DC таблицы: категории 0–11
        dc_y_table = self._build_huffman_table(self.dc_freq['Y'])
        dc_c_table = self._build_huffman_table(self.dc_freq['C'])

        # AC таблицы: пары (run, size)
        ac_y_table = self._build_huffman_table(self.ac_freq['Y'])
        ac_c_table = self._build_huffman_table(self.ac_freq['C'])

        huffman_tables = {
            "DC_Y": dc_y_table,
            "DC_C": dc_c_table,
            "AC_Y": ac_y_table,
            "AC_C": ac_c_table
        }

        self.logger.debug(f"""
    Generation Huffman tables: Success
    DC_Y_table size: {len(dc_y_table)}
    AC_Y_table size: {len(ac_y_table)}
    DC_Y_table: {dict(list(dc_y_table.items()))}
    AC_Y_table: {dict(list(ac_y_table.items()))}
    """)
        
        return huffman_tables
    
    def _scale_quant_table(self, table, quality):
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        # ВАЖНО: привести к float32, чтобы избежать переполнения
        table = table.astype(np.float32)
        scaled = np.floor((table * scale + 50) / 100)
        return np.clip(scaled, 1, 255).astype(np.uint8)
    
    @log_step
    def _rgb_to_ycbcr(self, rgb_pixels: np.array):
        """Конвертация RGB в YCbCr"""
        
        self.logger.debug(f"""
    Size of rgb_pixels: {rgb_pixels.shape}
    2x2 block:
        {rgb_pixels[0][0]} {rgb_pixels[0][1]}
        {rgb_pixels[1][0]} {rgb_pixels[1][1]}\n""")
        
        # Разделяем каналы RGB
        R = rgb_pixels[:, :, 0].copy()
        G = rgb_pixels[:, :, 1].copy()
        B = rgb_pixels[:, :, 2].copy()
        
        # Преобразование в YCbCr согласно стандарту JPEG
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
        Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
        
        # Ограничение по стандарту JPEG
        Y = np.clip(Y, 16, 235)
        Cb = np.clip(Cb, 16, 240)
        Cr = np.clip(Cr, 16, 240)
        
        self.logger.debug(f"""
    Transform RGB → YCbCr: Success
    Size of Y: {Y.shape}
    2x2 Y-block:
        {Y[0][0]} {Y[0][1]}
        {Y[1][0]} {Y[1][1]}\n""")
    
        return {
            'Y' : Y,
            'Cb' : Cb,
            'Cr' : Cr
        }
        
    @log_step
    def _chroma_subsampling(self, ycbcr: dict):
        """Хроматическое прореживание 4:2:0 с паддингом и усреднением"""
        
        original_height, original_width = self._original_pixels.shape[:2]
        
        # Вычисляем размеры с паддингом
        padded_height = original_height + (original_height % 2)
        padded_width = original_width + (original_width % 2)
        
        # Создаем массивы с паддингом
        Y_padded = np.zeros((padded_height, padded_width), dtype=np.float32)
        Cb_padded = np.zeros((padded_height, padded_width), dtype=np.float32)
        Cr_padded = np.zeros((padded_height, padded_width), dtype=np.float32)
        
        # Копируем оригинальные данные
        Y_padded[:original_height, :original_width] = ycbcr['Y']
        Cb_padded[:original_height, :original_width] = ycbcr['Cb']
        Cr_padded[:original_height, :original_width] = ycbcr['Cr']
        
        # Дублируем граничные пиксели для паддинга
        if original_height < padded_height:
            Y_padded[original_height:, :] = Y_padded[original_height-1:original_height, :]
            Cb_padded[original_height:, :] = Cb_padded[original_height-1:original_height, :]
            Cr_padded[original_height:, :] = Cr_padded[original_height-1:original_height, :]
        
        if original_width < padded_width:
            Y_padded[:, original_width:] = Y_padded[:, original_width-1:original_width]
            Cb_padded[:, original_width:] = Cb_padded[:, original_width-1:original_width]
            Cr_padded[:, original_width:] = Cr_padded[:, original_width-1:original_width]
        
        # Теперь работаем с четными размерами
        Cb_blocks = Cb_padded.reshape(padded_height//2, 2, padded_width//2, 2)
        Cb_subsampled = np.mean(Cb_blocks, axis=(1, 3))
        
        Cr_blocks = Cr_padded.reshape(padded_height//2, 2, padded_width//2, 2)
        Cr_subsampled = np.mean(Cr_blocks, axis=(1, 3))
        
        # Яркостный канал (не прореживается)
        Y_subsampled = Y_padded[:original_height, :original_width]
        
        self.logger.debug(f"""
    YCbCr-array subsample: Success
    Size of subsampled_pixels: {Y_subsampled.shape}
    2x2 Y-block:
        {Y_subsampled[0][0]} {Y_subsampled[0][1]}
        {Y_subsampled[1][0]} {Y_subsampled[1][1]}\n""")
        
        return {
        'Y_subsampled' : Y_subsampled,
        'Cb_subsampled' : Cb_subsampled,
        'Cr_subsampled' : Cr_subsampled,
    }
        
    @log_step
    def _split_into_blocks(self, subsampled: dict):
        """Разбиение Y, Cb, Cr на блоки 8x8 с индивидуальным паддингом (поддержка 4:2:0)."""

        def pad_channel(channel: np.ndarray) -> np.ndarray:
            """Добавляет паддинг, чтобы размеры были кратны 8, дублируя граничные пиксели."""
            h, w = channel.shape
            padded_h = h + (8 - h % 8) if h % 8 != 0 else h
            padded_w = w + (8 - w % 8) if w % 8 != 0 else w

            padded = np.zeros((padded_h, padded_w), dtype=np.float32)
            padded[:h, :w] = channel

            # дублирование нижней и правой границ
            if h < padded_h:
                padded[h:, :] = padded[h-1:h, :]
            if w < padded_w:
                padded[:, w:] = padded[:, w-1:w]

            return padded

        def split_channel(channel: np.ndarray) -> np.ndarray:
            """Разбивает канал на блоки 8x8."""
            h, w = channel.shape
            return channel.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)

        # Обработка каждого канала отдельно
        Y = subsampled['Y_subsampled']
        Cb = subsampled['Cb_subsampled']
        Cr = subsampled['Cr_subsampled']

        Y_padded = pad_channel(Y)
        Cb_padded = pad_channel(Cb)
        Cr_padded = pad_channel(Cr)

        Y_blocks = split_channel(Y_padded)
        Cb_blocks = split_channel(Cb_padded)
        Cr_blocks = split_channel(Cr_padded)

        self.logger.debug(f"""
    _split_into_blocks: Success
    Y_blocks shape: {Y_blocks.shape} (H_blocks × W_blocks × 8 × 8)
    Cb_blocks shape: {Cb_blocks.shape}
    Cr_blocks shape: {Cr_blocks.shape}
    First Y block sample:
    {np.array2string(Y_blocks[0,0], precision=2, floatmode='fixed')}
    """)

        return {
            'Y_blocks': Y_blocks,
            'Cb_blocks': Cb_blocks,
            'Cr_blocks': Cr_blocks
        }


    @log_step
    def _level_shift(self, blocks_data):
        """Применяет уровень сдвига к данным"""
        
        shifted_blocks = {
            'Y_shift_blocks': blocks_data['Y_blocks'] - 128,
            'Cb_shift_blocks': blocks_data['Cb_blocks'] - 128,
            'Cr_shift_blocks': blocks_data['Cr_blocks'] - 128
        }
        
        shifted_block_str = "\n\n".join(" ".join(f"{val:>8.3f}" for val in row) for row in shifted_blocks["Y_shift_blocks"][0][0])
        self.logger.debug(f"""
    Level Shift: Success
    Number of shifted blocks: {shifted_blocks['Y_shift_blocks'].shape[0]}x{shifted_blocks['Y_shift_blocks'].shape[1]}
    Size of shifted-Y: {shifted_blocks['Y_shift_blocks'].shape}
    DC-component: {shifted_blocks['Y_shift_blocks'][0][0][0][0]}
    First shifted-Y block:\n
{shifted_block_str}\n""")
        
        return shifted_blocks

    def _dct(self, block):
        """Применяет 2D DCT к одному блоку 8x8. Возвращает блок 8x8 DCT коэффициентов"""
        return self.DCT_MATRIX @ block @ self.DCT_MATRIX.T
    
    @log_step
    def _apply_dct(self, blocks_data):
        """Применяет DCT ко всем блокам всех каналов (учитывает разные размеры для Y, Cb, Cr)"""
        
        shifted_blocks_data = self._level_shift(blocks_data)

        Y_blocks = shifted_blocks_data['Y_shift_blocks']
        Cb_blocks = shifted_blocks_data['Cb_shift_blocks']
        Cr_blocks = shifted_blocks_data['Cr_shift_blocks']

        # Функция для применения DCT ко всем блокам одного канала
        def apply_dct_to_channel(blocks: np.ndarray) -> np.ndarray:
            h_blocks, w_blocks = blocks.shape[:2]
            dct_blocks = np.zeros_like(blocks)
            for i in range(h_blocks):
                for j in range(w_blocks):
                    dct_blocks[i, j] = self._dct(blocks[i, j])
            return dct_blocks

        Y_dct = apply_dct_to_channel(Y_blocks)
        Cb_dct = apply_dct_to_channel(Cb_blocks)
        Cr_dct = apply_dct_to_channel(Cr_blocks)

        dct_block_str = "\n\n".join("   ".join(f"{val:>8.3f}" for val in row) for row in Y_dct[0][0])
        self.logger.debug(f"""
        DCT-encode: Success
        Y_dct shape: {Y_dct.shape}
        Cb_dct shape: {Cb_dct.shape}
        Cr_dct shape: {Cr_dct.shape}
        DC component (Y[0,0]): {Y_dct[0][0][0][0]:.3f}
        First Y DCT block:
    {dct_block_str}\n""")

        return {
            'Y_dct': Y_dct,
            'Cb_dct': Cb_dct,
            'Cr_dct': Cr_dct,
        }
   
    @log_step
    def _apply_quantization(self, dct_blocks: dict) -> dict:
        """Применяет квантование ко всем DCT-блокам всех каналов (учитывает разные размеры при 4:2:0)."""

        Y_dct = dct_blocks['Y_dct']
        Cb_dct = dct_blocks['Cb_dct']
        Cr_dct = dct_blocks['Cr_dct']

        # Масштабируем таблицы квантования один раз (а не в каждом цикле)
        qY = self._scale_quant_table(self.STANDARD_LUMINANCE_QUANT_TABLE, self.quality).astype(np.float32)
        qC = self._scale_quant_table(self.STANDARD_CHROMINANCE_QUANT_TABLE, self.quality).astype(np.float32)

        self.logger.debug(f"""
        STANDARD_LUMINANCE_QUANT_TABLE (scaled, quality={self.quality}):
    {qY}

        STANDARD_CHROMINANCE_QUANT_TABLE (scaled):
    {qC}\n""")

        # Функция для квантования одного канала
        def quantize_channel(dct_array: np.ndarray, qtable: np.ndarray) -> np.ndarray:
            h_blocks, w_blocks = dct_array.shape[:2]
            quantized = np.zeros_like(dct_array, dtype=np.int32)
            for i in range(h_blocks):
                for j in range(w_blocks):
                    quantized[i, j] = np.round(dct_array[i, j] / qtable).astype(np.int32)
            return quantized

        # Применяем квантование для каждого канала отдельно
        Y_quant = quantize_channel(Y_dct, qY)
        Cb_quant = quantize_channel(Cb_dct, qC)
        Cr_quant = quantize_channel(Cr_dct, qC)

        quant_block_str = "\n\n".join(
            "   ".join(f"{val:>8d}" for val in row) for row in Y_quant[0][0]
        )
        self.logger.debug(f"""
        Quantization: Success
        Y_quant shape: {Y_quant.shape}
        Cb_quant shape: {Cb_quant.shape}
        Cr_quant shape: {Cr_quant.shape}
        First quantized Y block:
    {quant_block_str}\n""")

        return {
            'Y_quant': Y_quant,
            'Cb_quant': Cb_quant,
            'Cr_quant': Cr_quant
        }

    def _value_to_bits(self, value: int) -> tuple[int, str]:
            """Возвращает (size, bits) по стандарту JPEG для DC-компонент."""
            if value == 0:
                return 0, ""
            size = int(math.floor(math.log2(abs(value)))) + 1
            if value > 0:
                bits = format(value, f"0{size}b")
            else:
                bits = format((1 << size) - 1 + value, f"0{size}b")  # инверсия
            return size, bits

    @log_step
    def _dc_differentiation(self, quantized):
        """
        Выполняет дифференциальное кодирование (DPCM) DC-коэффициентов
        для всех компонент (Y, Cb, Cr) и преобразует их в пары (SIZE, VALUE_BITS),
        а также параллельно собирает информацию о частотах DC-компонент в словарь.

        Args:
            quantized (dict): {
                'Y_quant': np.ndarray,
                'Cb_quant': np.ndarray,
                'Cr_quant': np.ndarray
            }
            
            Size: (m, n, 8, 8), [0, 0] — DC-component.

        Returns:
            dict: {
                'Y_dc_encoded': list,  # [{'size': int, 'value_bits': str}, ...]
                'Cb_dc_encoded': list, # [{'size': int, 'value_bits': str}, ...]
                'Cr_dc_encoded': list  # [{'size': int, 'value_bits': str}, ...]
            }

            - **size (int)**: Количество бит, необходимое для представления
                дифференциала.
            - **value_bits (str)**: Строковое представление битов
                дифференциала (с учетом инверсии для отрицательных чисел)
        """

        def dc_diff(blocks: np.ndarray) -> np.ndarray:
            """Извлекает DC-компоненты и вычисляет дифференциалы."""
            dc_values = blocks[:, :, 0, 0].flatten()
            dc_diffs = np.diff(np.insert(dc_values, 0, 0))  # первый относительно 0
            return dc_diffs

        def encode_dc_channel(blocks: np.ndarray, freq_key: str) -> list[dict]:
            """Кодирует все DC-компоненты канала в список словарей."""
            diffs = dc_diff(blocks)
            encoded = []
            for diff in diffs:
                size, bits = self._value_to_bits(diff)
                # собираем статистику (частоты категорий)
                self.dc_freq.setdefault(freq_key, {})
                self.dc_freq[freq_key][size] = self.dc_freq[freq_key].get(size, 0) + 1
                encoded.append({"size": size, "value_bits": bits})
            return encoded

        # Обработка каждого канала независимо
        Y_dc_encoded = encode_dc_channel(quantized['Y_quant'], 'Y')
        Cb_dc_encoded = encode_dc_channel(quantized['Cb_quant'], 'C')
        Cr_dc_encoded = encode_dc_channel(quantized['Cr_quant'], 'C')

        self.logger.debug(f"""
        DC-differentiation: Success
        Y_dc count: {len(Y_dc_encoded)}
        Cb_dc count: {len(Cb_dc_encoded)}
        Cr_dc count: {len(Cr_dc_encoded)}
        First 8 Y-diffs: {[item for item in np.array([b['size'] for b in Y_dc_encoded[:8]])]}
        First 8 Y value bits: {[b['value_bits'] for b in Y_dc_encoded[:8]]}
        """)

        return {
            'Y_dc_encoded': Y_dc_encoded,
            'Cb_dc_encoded': Cb_dc_encoded,
            'Cr_dc_encoded': Cr_dc_encoded,
        }


    def _zigzag_scanning(self, block):
        """Зигзаг-сканирование одного блока NxN"""
        
        # Для блока 8x8:
        #     zigzag_order = [
        #     (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
        #     (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
        #     (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
        #     (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
        #     (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        #     (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
        #     (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
        #     (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
        # ]
            
        N = block.shape[0]
        result = []
        for s in range(2 * N - 1):
            temp = []
            for i in range(s + 1):
                j = s - i
                if i < N and j < N:
                    temp.append((i, j))
            if s % 2 == 0:
                temp.reverse()  # чётные диагонали — снизу вверх
            for i, j in temp:
                result.append(block[i, j])
                        
    #     self.logger.debug(f"""
    # Zigzag-scanning: Success
    # Size of zigzag-array: {len(result)}
    # Zigzag array:
    #     {[int(el) for el in result]}\n""") 
                    
        return result

    def _run_length_encoding(self, ac_coeffs, freq_key):
        """
        Выполняет JPEG RLE-кодирование AC-коэффициентов: (RUN, SIZE) + битовое значение.
        Также автоматически собирает статистику частот в словаре self.ac_freq_dict.

        Args:
            ac_coeffs (np.ndarray): Массив AC-коэффициентов одного блока (после зигзага, исключая DC)

        Returns:
            dict: {
                'rle': list of (RUN, SIZE) пар,
                'values': list of битовых строк для коэффициентов
            }
        """
        rle = []
        values = []
        zero_run = 0
        self.ac_freq.setdefault(freq_key, {})
        for coeff in ac_coeffs:
            if coeff == 0:
                zero_run += 1
                if zero_run == 16:
                    symbol = (15, 0)
                    rle.append(symbol)
                    values.append("")
                    self.ac_freq[freq_key][symbol] = self.ac_freq[freq_key].get(symbol, 0) + 1
                    zero_run = 0
            else:
                size = int(math.floor(math.log2(abs(coeff)))) + 1
                symbol = (zero_run, size)
                # распакуем (size, bits) — чтобы values содержал строку бит
                size_expected, bits = self._value_to_bits(coeff) if isinstance(self._value_to_bits(coeff), tuple) else (size, self._value_to_bits(coeff))
                rle.append(symbol)
                values.append(bits)
                self.ac_freq[freq_key][symbol] = self.ac_freq[freq_key].get(symbol, 0) + 1
                zero_run = 0

        # Если остались нули в конце блока → EOB
        if zero_run > 0:
            rle.append((0, 0))
            values.append("")
            self.ac_freq[freq_key][(0, 0)] = self.ac_freq[freq_key].get((0, 0), 0) + 1

        return {"rle": rle, "values": values}

    @log_step
    def _encode_rle_ac_components(self, quantized):
        """
        Применяет зигзагообразное сканирование и RLE к AC-коэффициентам.

        Для всех квантованных блоков (Y, Cb, Cr) функция извлекает 
        AC-коэффициенты (исключая DC), выполняет RLE-кодирование (Run-Length
        Encoding) и подготавливает данные для кодирования Хаффмана.

        Args:
            quantized (dict): {
                'Y_quant': np.ndarray,
                'Cb_quant': np.ndarray,
                'Cr_quant': np.ndarray
            }
            
            Size: (m, n, 8, 8), [0, 0] — DC-component (исключается).

        Returns:
            dict: {
                'Y_ac_rle': list,  # [{'rle': int, 'value_bits': str}, ...]
                'Cb_ac_rle': list, # [{'rle': int, 'value_bits': str}, ...]
                'Cr_ac_rle': list  # [{'rle': int, 'value_bits': str}, ...]
            }

            - **rle (list)**: Список пар (RUN, SIZE). RUN — количество нулей, 
              SIZE — количество бит для ненулевого коэффициента.
            - **values (list)**: Список строковых представлений битов
              ненулевых AC-коэффициентов (амплитуд).
        """
        
        def process_channel(channel_blocks, freq_key):
            h, w = channel_blocks.shape[:2]
            flat_rle_list = []

            for i in range(h):
                for j in range(w):
                    block = channel_blocks[i, j]
                    zigzag = self._zigzag_scanning(block)
                    ac_coeffs = zigzag[1:]  # исключаем DC

                    # RLE для одного блока
                    rle_result = self._run_length_encoding(ac_coeffs, freq_key)
                    rle_pairs = rle_result["rle"]
                    values = rle_result["values"]

                    # Преобразуем в список словарей
                    for (run, size), val in zip(rle_pairs, values):
                        flat_rle_list.append({
                            "rle": (run, size),
                            "value_bits": val
                        })

            return flat_rle_list

        # Обрабатываем три канала
        Y_ac_rle = process_channel(quantized['Y_quant'], 'Y')
        Cb_ac_rle = process_channel(quantized['Cb_quant'], 'C')
        Cr_ac_rle = process_channel(quantized['Cr_quant'], 'C')

        # Пример для логов
        sample = Y_ac_rle[:10]
        self.logger.debug(f"""
        RLE-encoding (flat): Success
        Total Y-AC symbols: {len(Y_ac_rle)}
        First 10 Y-AC entries:
            {sample}
        """)

        self.logger.debug(f"""
        AC-Y frequency (top 10): {dict(sorted(self.ac_freq['Y'].items(), key=lambda x: x[1], reverse=True)[:10])}
        AC-Cb frequency (top 10): {dict(sorted(self.ac_freq['C'].items(), key=lambda x: x[1], reverse=True)[:10])}
        """)

        return {
            'Y_ac_rle': Y_ac_rle,
            'Cb_ac_rle': Cb_ac_rle,
            'Cr_ac_rle': Cr_ac_rle,
        }
    
    @log_step
    def _huffman_encoding(self, dc_components, ac_components, huffman_tables):
        """Хаффман-кодирование DC и AC-компонентов по JPEG-таблицам"""

        def encode_channel(dc_list, ac_list, dc_table, ac_table):
            bitstream = ""
            ac_index = 0  # указатель в общем списке AC

            for dc in dc_list:
                # --- DC ---
                size = dc["size"]
                value_bits = dc["value_bits"]
                huff_dc = dc_table.get(size, "")
                bitstream += huff_dc + value_bits

                # --- AC ---
                # читаем AC-коэффициенты для одного блока
                while ac_index < len(ac_list):
                    run, size = ac_list[ac_index]["rle"]
                    value_bits = ac_list[ac_index]["value_bits"]
                    ac_index += 1
                    huff_ac = ac_table.get((run, size), "")
                    bitstream += huff_ac + value_bits

                    # EOB → значит, блок окончен
                    if (run, size) == (0, 0):
                        break

            return bitstream

        # Кодируем каждый канал
        Y_bits = encode_channel(
            dc_components["Y_dc_encoded"], ac_components["Y_ac_rle"],
            huffman_tables["DC_Y"], huffman_tables["AC_Y"]
        )
        Cb_bits = encode_channel(
            dc_components["Cb_dc_encoded"], ac_components["Cb_ac_rle"],
            huffman_tables["DC_C"], huffman_tables["AC_C"]
        )
        Cr_bits = encode_channel(
            dc_components["Cr_dc_encoded"], ac_components["Cr_ac_rle"],
            huffman_tables["DC_C"], huffman_tables["AC_C"]
        )

        self.logger.debug(f"""
        Huffman encoding: Success
        Y bits length: {len(Y_bits)}
        Cb bits length: {len(Cb_bits)}
        Cr bits length: {len(Cr_bits)}
        First 512 bits of Y:
            {Y_bits[:512]}
        """)

        return {
            "Y_bits": Y_bits,
            "Cb_bits": Cb_bits,
            "Cr_bits": Cr_bits,
            "bitstream": Y_bits + Cb_bits + Cr_bits
        }

    
    @log_step
    def _create_app0_segment(self): return bytes([0]*3)
    @log_step
    def _create_dqt_segments(self): return [bytes([0]*3)]
    @log_step
    def _create_sof0_segment(self): return bytes([0]*3)
    @log_step
    def _create_dht_segments(self): return [bytes([0]*3)]
    @log_step
    def _create_sos_segment(self): return bytes([0]*3)
    
    @log_step
    def _create_jpeg(self, encoded_data: bytes, output_path: str):
        """Создание итогового JPEG файла из закодированных данных"""
        print(output_path)
        with open(output_path, 'wb') as f:
            # 1. SOI — Start of Image
            f.write(b'\xFF\xD8')

            # 2. APP0 — JFIF header
            f.write(self._create_app0_segment())

            # 3. DQT — Quantization Tables
            for qt_segment in self._create_dqt_segments():
                f.write(qt_segment)

            # 4. SOF0 — Start of Frame
            f.write(self._create_sof0_segment())

            # 5. DHT — Huffman Tables
            for dht_segment in self._create_dht_segments():
                f.write(dht_segment)

            # 6. SOS — Start of Scan
            f.write(self._create_sos_segment())
            
            def bits_to_bytes(bitstring):
                # Дополняем до кратности 8
                padding = (8 - len(bitstring) % 8) % 8
                bitstring += '0' * padding
                return bytes(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))

            # 7. Image Data — Закодированные данные
            f.write(bits_to_bytes(encoded_data))

            # 8. EOI — End of Image
            f.write(b'\xFF\xD9')

    
    @log_step
    def _load_image(self, image_path, quality):
        """Загрузка изображения (сбрасывает предыдущее состояние)"""
        
        # Сбрасываем все предыдущие данные
        self._reset_state()
        
        try:
            # Загружаем новое изображение
            image = Image.open(image_path)
            
            # Убеждаемся, что исходная цветовая модель = RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Преобразуем изображение в числовую матрицу для дальнейших преобразований
            self._original_pixels = np.array(image, dtype=np.float32)
            self.quality = quality
            
        except Exception as e:
            self._reset_state()
            raise ValueError(f"Image loading Error: {e}")
    
    
# public

    @log_step
    def compress(self, image_path: str, compressed_image_name: str, quality: int = 75):
        """Основной метод сжатия"""

        self._load_image(image_path, quality)

        rgb_pixels = self._original_pixels
        ycbcr_pixels = self._rgb_to_ycbcr(rgb_pixels)
        subsampled = self._chroma_subsampling(ycbcr_pixels)
        dict_blocks = self._split_into_blocks(subsampled)
        dict_dct_blocks = self._apply_dct(dict_blocks)
        dict_quant_blocks = self._apply_quantization(dict_dct_blocks)
        dc_components = self._dc_differentiation(dict_quant_blocks)
        ac_components = self._encode_rle_ac_components(dict_quant_blocks)
        huffman_tables = self._generate_all_huffman_tables()
        bitstream = self._huffman_encoding(dc_components, ac_components, huffman_tables)['bitstream']
        self._create_jpeg(bitstream, "data/"+compressed_image_name)