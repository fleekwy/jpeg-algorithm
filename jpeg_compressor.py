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
                    
        return result

    @log_step
    def _huffman_encoding(self, quantized_blocks: dict):
        """
        Выполняет кодирование Хаффмана с корректным чередованием MCU для 4:2:0.
        """
        bitstream = ""
        bit_buffer = 0
        bit_buffer_len = 0

        def write_bits(code, size):
            nonlocal bit_buffer, bit_buffer_len, bitstream
            bit_buffer = (bit_buffer << size) | code
            bit_buffer_len += size
            while bit_buffer_len >= 8:
                byte_to_write = (bit_buffer >> (bit_buffer_len - 8)) & 0xFF
                bitstream += chr(byte_to_write)
                if byte_to_write == 0xFF:
                    bitstream += chr(0x00) # Byte stuffing
                bit_buffer_len -= 8
        
        # Предыдущие DC значения для DPCM
        prev_dc = {'Y': 0, 'Cb': 0, 'Cr': 0}

        # Получаем блоки
        Y_quant = quantized_blocks['Y']
        Cb_quant = quantized_blocks['Cb']
        Cr_quant = quantized_blocks['Cr']

        h_mcu = Y_quant.shape[0] // 2
        w_mcu = Y_quant.shape[1] // 2
        
        self.logger.debug(f"Total MCUs to process: {h_mcu * w_mcu} ({h_mcu}x{w_mcu})")

        for i in range(h_mcu):
            for j in range(w_mcu):
                # --- Один MCU (4 Y, 1 Cb, 1 Cr) ---
                y_blocks_mcu = [
                    Y_quant[i*2, j*2], Y_quant[i*2, j*2+1],
                    Y_quant[i*2+1, j*2], Y_quant[i*2+1, j*2+1]
                ]
                mcu_blocks = [
                    (y_blocks_mcu[0], 'Y'), (y_blocks_mcu[1], 'Y'), (y_blocks_mcu[2], 'Y'), (y_blocks_mcu[3], 'Y'),
                    (Cb_quant[i,j], 'Cb'),
                    (Cr_quant[i,j], 'Cr')
                ]

                for block, ch_type in mcu_blocks:
                    # --- Обработка одного блока внутри MCU ---
                    zigzag_coeffs = self._zigzag_scanning(block)
                    
                    # 1. DC коэффициент
                    dc_val = zigzag_coeffs[0]
                    dc_diff = dc_val - prev_dc[ch_type]
                    prev_dc[ch_type] = dc_val
                    
                    dc_size, dc_bits_str = self._value_to_bits(dc_diff)
                    
                    dc_codes = self.STANDARD_LUMINANCE_HUFFMAN_DC_TABLE['CODES'] if ch_type == 'Y' else self.STANDARD_CHROMINANCE_HUFFMAN_DC_TABLE['CODES']
                    dc_huffsize = self.STANDARD_LUMINANCE_HUFFMAN_DC_TABLE['HUFFSIZE'] if ch_type == 'Y' else self.STANDARD_CHROMINANCE_HUFFMAN_DC_TABLE['HUFFSIZE']
                    
                    huff_code = dc_codes[dc_size]
                    huff_size = dc_huffsize[dc_size]
                    
                    write_bits(huff_code, huff_size)
                    if dc_size > 0:
                        write_bits(int(dc_bits_str, 2), dc_size)

                    # 2. AC коэффициенты
                    ac_coeffs = zigzag_coeffs[1:]
                    zero_run = 0
                    
                    ac_codes = self.STANDARD_LUMINANCE_HUFFMAN_AC_TABLE['CODES'] if ch_type == 'Y' else self.STANDARD_CHROMINANCE_HUFFMAN_AC_TABLE['CODES']
                    ac_huffsize = self.STANDARD_LUMINANCE_HUFFMAN_AC_TABLE['HUFFSIZE'] if ch_type == 'Y' else self.STANDARD_CHROMINANCE_HUFFMAN_AC_TABLE['HUFFSIZE']

                    for coeff in ac_coeffs:
                        if coeff == 0:
                            zero_run += 1
                        else:
                            while zero_run >= 16:
                                # ZRL (Zero Run Length)
                                write_bits(ac_codes[0xF0], ac_huffsize[0xF0])
                                zero_run -= 16
                            
                            ac_size, ac_bits_str = self._value_to_bits(coeff)
                            
                            # (run, size)
                            symbol = (zero_run << 4) | ac_size
                            
                            huff_code = ac_codes[symbol]
                            huff_size = ac_huffsize[symbol]
                            
                            write_bits(huff_code, huff_size)
                            write_bits(int(ac_bits_str, 2), ac_size)
                            
                            zero_run = 0
                    
                    # EOB (End of Block)
                    if zero_run > 0 or np.all(ac_coeffs == 0):
                        write_bits(ac_codes[0x00], ac_huffsize[0x00])

        # Завершение битового потока (padding)
        if bit_buffer_len > 0:
            # Паддинг единицами до полного байта
            pad_len = 8 - bit_buffer_len
            code = (bit_buffer << pad_len) | ((1 << pad_len) - 1)
            byte_to_write = code & 0xFF
            bitstream += chr(byte_to_write)
            if byte_to_write == 0xFF:
                bitstream += chr(0x00)
        
        # Конвертация в байты
        return bitstream.encode('latin-1')

    
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