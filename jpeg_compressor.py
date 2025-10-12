import numpy as np
from PIL import Image
import os
import logging.config
import yaml
from datetime import datetime
import struct
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
        self.origin_height = None
        self.origin_width = None
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
        
        self.STANDARD_LUMINANCE_HUFFMAN_DC_TABLE = None
        self.STANDARD_CHROMINANCE_HUFFMAN_DC_TABLE = None
        self.STANDARD_LUMINANCE_HUFFMAN_AC_TABLE = None
        self.STANDARD_CHROMINANCE_HUFFMAN_AC_TABLE = None
        
        
        self.Y_DC_HUFFMAN_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        self.Y_DC_HUFFMAN_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        
        self.C_DC_HUFFMAN_BITS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        self.C_DC_HUFFMAN_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.Y_AC_HUFFMAN_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
        self.Y_AC_HUFFMAN_VALS = [
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
            0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
            0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
            0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
            0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
            0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
            0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
            0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
            0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
            0xF9, 0xFA
        ]

        self.C_AC_HUFFMAN_BITS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
        self.C_AC_HUFFMAN_VALS = [
            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
            0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
            0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
            0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
            0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
            0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
            0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
            0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
            0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
            0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
            0xF9, 0xFA
        ]

        # --- Сгенерированные таблицы для быстрого доступа ---
        self.STANDARD_LUMINANCE_HUFFMAN_DC_TABLE = self._generate_huffman_table(self.Y_DC_HUFFMAN_BITS, self.Y_DC_HUFFMAN_VALS)
        self.STANDARD_CHROMINANCE_HUFFMAN_DC_TABLE = self._generate_huffman_table(self.C_DC_HUFFMAN_BITS, self.C_DC_HUFFMAN_VALS)
        self.STANDARD_LUMINANCE_HUFFMAN_AC_TABLE = self._generate_huffman_table(self.Y_AC_HUFFMAN_BITS, self.Y_AC_HUFFMAN_VALS)
        self.STANDARD_CHROMINANCE_HUFFMAN_AC_TABLE = self._generate_huffman_table(self.C_AC_HUFFMAN_BITS, self.C_AC_HUFFMAN_VALS)
        
        self.logger.info("__init__: <end>")
        
# protected
        
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

    @log_step
    def _reset_state(self):
        self._original_pixels = None
        self._compressed_pixels = None
        self.origin_height = None
        self.origin_width = None
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
    def _generate_huffman_table(self, bits, huffval):
        """Генерирует таблицу кодов Хаффмана из стандартных BITS и HUFFVAL."""
        codes = {}
        huffsize = {}
        k = 0
        code = 0
        for i in range(16):
            length = i + 1
            for _ in range(bits[i]):
                symbol = huffval[k]
                codes[symbol] = code
                huffsize[symbol] = length
                k += 1
                code += 1
            code <<= 1
            
        return {
            'CODES' : codes,
            'HUFFSIZE' : huffsize
        }
    
    @log_step
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
        
        R = rgb_pixels[:, :, 0].copy()
        G = rgb_pixels[:, :, 1].copy()
        B = rgb_pixels[:, :, 2].copy()
        
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
        Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
        
        self.logger.debug(f"""
    Transform RGB → YCbCr: Success
    Size of ycbcr-Y: {Y.shape}
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
        
        original_height, original_width = self.origin_height, self.origin_width
        
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
    Y_subsampled shape: {Y_subsampled.shape}
    Cb_subsampled shape: {Cb_subsampled.shape}
    Cr_subsampled shape: {Cr_subsampled.shape}
    2x2 Y-block:
        {Y_subsampled[0][0]} {Y_subsampled[0][1]}
        {Y_subsampled[1][0]} {Y_subsampled[1][1]}\n""")
        
        return {
        'Y' : Y_subsampled,
        'Cb' : Cb_subsampled,
        'Cr' : Cr_subsampled,
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
        Y = subsampled['Y']
        Cb = subsampled['Cb']
        Cr = subsampled['Cr']

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
            'Y': Y_blocks,
            'Cb': Cb_blocks,
            'Cr': Cr_blocks
        }

    @log_step
    def _level_shift(self, blocks_data):
        """Применяет уровень сдвига к данным"""
        
        shifted_blocks = {
            'Y': blocks_data['Y'] - 128,
            'Cb': blocks_data['Cb'] - 128,
            'Cr': blocks_data['Cr'] - 128
        }
        
        shifted_block_str = "\n\n".join(" ".join(f"{val:>8.3f}" for val in row) for row in shifted_blocks["Y"][0][0])
        self.logger.debug(f"""
    Level Shift: Success
    Number of shifted blocks: {shifted_blocks['Y'].shape[0]}x{shifted_blocks['Y'].shape[1]}
    Size of shifted-Y: {shifted_blocks['Y'].shape}
    DC-component: {shifted_blocks['Y'][0][0][0][0]}
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

        Y_blocks = shifted_blocks_data['Y']
        Cb_blocks = shifted_blocks_data['Cb']
        Cr_blocks = shifted_blocks_data['Cr']

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
            'Y': Y_dct,
            'Cb': Cb_dct,
            'Cr': Cr_dct,
        }
   
    @log_step
    def _apply_quantization(self, dct_blocks: dict) -> dict:
        """Применяет квантование ко всем DCT-блокам всех каналов (учитывает разные размеры при 4:2:0)."""

        Y_dct = dct_blocks['Y']
        Cb_dct = dct_blocks['Cb']
        Cr_dct = dct_blocks['Cr']

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
            'Y': Y_quant,
            'Cb': Cb_quant,
            'Cr': Cr_quant
        }

    def _value_to_bits(self, value: int) -> tuple[int, str]:
            """Возвращает (size, bits) по стандарту JPEG."""
            # Convert numpy int to python int to use .bit_length()
            py_value = int(value)

            if py_value == 0:
                return 0, ""
            
            size = py_value.bit_length()
            if py_value > 0:
                return size, format(py_value, f'0{size}b')
            else:
                # Для отрицательных чисел: two's complement, но инвертированный
                return size, format((1 << size) + py_value - 1, f'0{size}b')

    def _zigzag_scanning(self, block):
        """Зигзаг-сканирование одного блока NxN"""
            
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
                    
        return result

    @log_step
    def _huffman_encoding(self, quantized_blocks: dict):
        """
        Быстрое Хаффман-кодирование с MCU 4:2:0, минимум аллокаций и вызовов.
        Возвращает bytes.
        """
        import numpy as _np  # локальная ссылка

        out = bytearray()
        bit_buffer = 0
        bit_len = 0

        def write_bits_fast(code: int, size: int):
            nonlocal bit_buffer, bit_len, out
            bit_buffer = (bit_buffer << size) | (code & ((1 << size) - 1))
            bit_len += size
            while bit_len >= 8:
                shift = bit_len - 8
                byte = (bit_buffer >> shift) & 0xFF
                out.append(byte)
                if byte == 0xFF:
                    out.append(0x00)
                bit_len -= 8
                bit_buffer &= (1 << shift) - 1 if shift > 0 else 0

        # локализуем часто используемые таблицы/функции
        prev_dc = {'Y': 0, 'Cb': 0, 'Cr': 0}
        Y_quant = quantized_blocks['Y']
        Cb_quant = quantized_blocks['Cb']
        Cr_quant = quantized_blocks['Cr']

        # Локализация для ускорения
        lum_dc_codes = self.STANDARD_LUMINANCE_HUFFMAN_DC_TABLE['CODES']
        chr_dc_codes = self.STANDARD_CHROMINANCE_HUFFMAN_DC_TABLE['CODES']
        lum_dc_sizes = self.STANDARD_LUMINANCE_HUFFMAN_DC_TABLE['HUFFSIZE']
        chr_dc_sizes = self.STANDARD_CHROMINANCE_HUFFMAN_DC_TABLE['HUFFSIZE']

        lum_ac_codes = self.STANDARD_LUMINANCE_HUFFMAN_AC_TABLE['CODES']
        chr_ac_codes = self.STANDARD_CHROMINANCE_HUFFMAN_AC_TABLE['CODES']
        lum_ac_sizes = self.STANDARD_LUMINANCE_HUFFMAN_AC_TABLE['HUFFSIZE']
        chr_ac_sizes = self.STANDARD_CHROMINANCE_HUFFMAN_AC_TABLE['HUFFSIZE']

        # локальные методы класса
        zigzag = self._zigzag_scanning
        value_to_bits = self._value_to_bits

        h_mcu = Y_quant.shape[0] // 2
        w_mcu = Y_quant.shape[1] // 2

        # ускоренный обход MCU
        for i in range(h_mcu):
            base_i2 = i * 2
            for j in range(w_mcu):
                base_j2 = j * 2
                # собираем 4 блока Y и Cb, Cr
                y0 = Y_quant[base_i2, base_j2]
                y1 = Y_quant[base_i2, base_j2 + 1]
                y2 = Y_quant[base_i2 + 1, base_j2]
                y3 = Y_quant[base_i2 + 1, base_j2 + 1]
                cb = Cb_quant[i, j]
                cr = Cr_quant[i, j]

                # порядок блоков внутри MCU
                blocks = ((y0, 'Y'), (y1, 'Y'), (y2, 'Y'), (y3, 'Y'), (cb, 'Cb'), (cr, 'Cr'))

                for block, ch in blocks:
                    coeffs = zigzag(block)  # ожидается iterable длины 64
                    # DC
                    dc = int(coeffs[0])
                    diff = dc - prev_dc[ch]
                    prev_dc[ch] = dc

                    dc_size, dc_bits_str = value_to_bits(diff)
                    if ch == 'Y':
                        huff_code = lum_dc_codes[dc_size]
                        huff_size = lum_dc_sizes[dc_size]
                    else:
                        huff_code = chr_dc_codes[dc_size]
                        huff_size = chr_dc_sizes[dc_size]

                    write_bits_fast(huff_code, huff_size)
                    if dc_size:
                        write_bits_fast(int(dc_bits_str, 2), dc_size)

                    # AC
                    ac = coeffs[1:]  # последовательность 63
                    zero_run = 0
                    if ch == 'Y':
                        ac_codes = lum_ac_codes
                        ac_sizes = lum_ac_sizes
                    else:
                        ac_codes = chr_ac_codes
                        ac_sizes = chr_ac_sizes

                    # пробегаем коэффициенты
                    # минимизируем Python-обороты внутри цикла
                    for c in ac:
                        if c == 0:
                            zero_run += 1
                            continue
                        while zero_run >= 16:
                            zrl_code = ac_codes[0xF0]
                            zrl_size = ac_sizes[0xF0]
                            write_bits_fast(zrl_code, zrl_size)
                            zero_run -= 16

                        ac_size, ac_bits_str = value_to_bits(c)
                        symbol = (zero_run << 4) | ac_size
                        huff_code = ac_codes[symbol]
                        huff_size = ac_sizes[symbol]
                        write_bits_fast(huff_code, huff_size)
                        write_bits_fast(int(ac_bits_str, 2), ac_size)
                        zero_run = 0

                    # EOB: если после всех коэффициентов остались нули
                    # проверка на все нули AC быстрее как простой zero_run>0, но
                    # если блок был полностью нулевой zero_run==63 - EOB нужен
                    if zero_run > 0:
                        eob_code = ac_codes[0x00]
                        eob_size = ac_sizes[0x00]
                        write_bits_fast(eob_code, eob_size)
                    else:
                        # случая, когда все AC == 0 (zero_run == 63) тоже покрыт выше
                        pass

        # дописать паддинг единицами до полного байта
        if bit_len:
            pad = 8 - bit_len
            pad_bits = (1 << pad) - 1
            final = (bit_buffer << pad) | pad_bits
            out.append(final & 0xFF)
            if (final & 0xFF) == 0xFF:
                out.append(0x00)

        return bytes(out)

    
    @log_step
    def _create_app0_segment(self) -> bytes:
        """APP0 (JFIF) segment."""
        identifier = b"JFIF\x00"
        version = struct.pack(">BB", 1, 1)
        units = 0
        x_density, y_density = 1, 1
        x_thumb, y_thumb = 0, 0
        payload = identifier + version + bytes([units]) + struct.pack(">HHBB", x_density, y_density, x_thumb, y_thumb)
        length = 2 + len(payload)
        return b'\xFF\xE0' + struct.pack(">H", length) + payload

    @log_step
    def _create_dqt_segments(self) -> bytes:
        """Собирает DQT сегменты для Y и C."""
        segments = bytearray()
        
        lum_bytes = self._scale_quant_table(self.STANDARD_LUMINANCE_QUANT_TABLE, self.quality)
        payload = bytes([0x00]) + lum_bytes.flatten().tobytes()
        length = 2 + len(payload)
        segments += b'\xFF\xDB' + struct.pack(">H", length) + payload

        chrom_bytes = self._scale_quant_table(self.STANDARD_CHROMINANCE_QUANT_TABLE, self.quality)
        payload = bytes([0x01]) + chrom_bytes.flatten().tobytes()
        length = 2 + len(payload)
        segments += b'\xFF\xDB' + struct.pack(">H", length) + payload

        return bytes(segments)

    @log_step
    def _create_sof0_segment(self) -> bytes:
        """SOF0 (Baseline DCT)"""
        payload = struct.pack(">BHHB", 8, self.origin_height, self.origin_width, 3)
        # Y: id=1, sampling=(2,2), qt=0
        payload += bytes([1, 0x22, 0])
        # Cb: id=2, sampling=(1,1), qt=1
        payload += bytes([2, 0x11, 1])
        # Cr: id=3, sampling=(1,1), qt=1
        payload += bytes([3, 0x11, 1])
        
        length = 2 + len(payload)
        return b'\xFF\xC0' + struct.pack(">H", length) + payload

    @log_step
    def _build_dht_segment(self, bits, huffval, tc, th) -> bytes:
        """Строит один DHT сегмент из стандартных таблиц."""
        payload = bytearray()
        payload.append((tc << 4) | th) # Tc/Th
        payload.extend(bits)
        payload.extend(huffval)
        
        length = 2 + len(payload)
        return b'\xFF\xC4' + struct.pack(">H", length) + bytes(payload)

    @log_step
    def _create_dht_segments(self) -> bytes:
        """Формирует все 4 DHT сегмента."""
        segments = bytearray()
        segments += self._build_dht_segment(self.Y_DC_HUFFMAN_BITS, self.Y_DC_HUFFMAN_VALS, 0, 0) # DC_Y
        segments += self._build_dht_segment(self.Y_AC_HUFFMAN_BITS, self.Y_AC_HUFFMAN_VALS, 1, 0) # AC_Y
        segments += self._build_dht_segment(self.C_DC_HUFFMAN_BITS, self.C_DC_HUFFMAN_VALS, 0, 1) # DC_C
        segments += self._build_dht_segment(self.C_AC_HUFFMAN_BITS, self.C_AC_HUFFMAN_VALS, 1, 1) # AC_C
        return bytes(segments)

    @log_step
    def _create_sos_segment(self) -> bytes:
        """Start of Scan."""
        payload = bytearray()
        payload.append(3) # 3 компонента в скане
        # Y -> использует таблицы DC 0, AC 0
        payload.extend(bytes([1, (0 << 4) | 0]))
        # Cb -> DC 1, AC 1
        payload.extend(bytes([2, (1 << 4) | 1]))
        # Cr -> DC 1, AC 1
        payload.extend(bytes([3, (1 << 4) | 1]))
        # Ss, Se, Ah, Al
        payload.extend(bytes([0, 63, 0]))

        length = 2 + len(payload)
        return b'\xFF\xDA' + struct.pack(">H", length) + bytes(payload)
    
    @log_step
    def _create_jpeg(self, image_data: bytes, output_path: str):
        """Создание итогового JPEG файла из закодированных данных"""
        
        soi = b'\xFF\xD8'
        app0 = self._create_app0_segment()
        dqt = self._create_dqt_segments()
        sof0 = self._create_sof0_segment()
        dht = self._create_dht_segments()
        sos = self._create_sos_segment()
        eoi = b'\xFF\xD9'
        
        self.logger.debug(f"""
        JPEG segments sizes:
        APP0: {len(app0)}, DQT: {len(dqt)}, SOF0: {len(sof0)},
        DHT: {len(dht)}, SOS: {len(sos)}, Image data: {len(image_data)} bytes
        Total: {len(soi) + len(app0) + len(dqt) + len(sof0) + len(dht) + len(sos) + len(image_data) + len(eoi)} bytes
        """)
        
        try:
            if not output_path.lower().endswith((".jpg", ".jpeg")):
                output_path += ".jpg"
            with open(output_path, "wb") as f:
                f.write(soi)
                f.write(app0)
                f.write(dqt)
                f.write(sof0)
                f.write(dht)
                f.write(sos)
                f.write(image_data)
                f.write(eoi)
            self.logger.info(f"JPEG file created successfully: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to create JPEG file: {e}")
            raise
    
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
            self.origin_height, self.origin_width = self._original_pixels.shape[:2]
            
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
        image_data = self._huffman_encoding(dict_quant_blocks)
        
        # Убедимся, что директория для сохранения существует
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, compressed_image_name)

        self._create_jpeg(image_data, output_path)