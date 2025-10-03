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

            # Формируем базовое имя по дате
            today = datetime.now().strftime("%Y-%m-%d")
            base_name = f"debug_log_{today}"
            ext = ".txt"
            log_dir = "logs"
            
            # Ищем свободное имя: debug_log_2025-10-03.txt, _1.txt, _2.txt и т.д.
            i = 0
            while True:
                suffix = f"_{i}"
                filename = os.path.join(log_dir, f"{base_name}{suffix}{ext}")
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
    def _rgb_to_ycbcr(self, rgb_pixels):
        """Конвертация RGB в YCbCr"""
        
        self.logger.debug(f"""
    Size of rgb_pixels: {rgb_pixels.shape}
    2x2 block:
        {rgb_pixels[0][0]} {rgb_pixels[0][1]}
        {rgb_pixels[1][0]} {rgb_pixels[1][1]}\n""")
        
        # Создаем пустой массив для YCbCr
        ycbcr_pixels = np.zeros_like(rgb_pixels)
        
        # Разделяем каналы RGB
        R = rgb_pixels[:, :, 0].copy()
        G = rgb_pixels[:, :, 1].copy()
        B = rgb_pixels[:, :, 2].copy()
        
        # Преобразование в YCbCr согласно стандарту JPEG
        # Компонента Y (яркость)
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        
        # Компоненты Cb и Cr (цветность)
        Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
        Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
        
        # Ограничение по стандарту JPEG
        Y = np.clip(Y, 16, 235)
        Cb = np.clip(Cb, 16, 240)
        Cr = np.clip(Cr, 16, 240)
        
        # Сохраняем в поле self._pixels
        ycbcr_pixels[:, :, 0] = Y
        ycbcr_pixels[:, :, 1] = Cb
        ycbcr_pixels[:, :, 2] = Cr
        
        # Протестируем
        # pixels_uint8 = np.clip(self._pixels, 0, 255).astype(np.uint8)
        # Image.fromarray(pixels_uint8).save("data/output_1.jpeg")
        
        self.logger.debug(f"""
    Transform RGB → YCbCr: Success
    Size of ycbcr_pixels: {ycbcr_pixels.shape}
    2x2 block:
        {ycbcr_pixels[0][0]} {ycbcr_pixels[0][1]}
        {ycbcr_pixels[1][0]} {ycbcr_pixels[1][1]}\n""")
    
        return ycbcr_pixels
        
    @log_step
    def _chroma_subsampling(self, ycbcr_pixels):
        """Хроматическое прореживание 4:2:0 с паддингом и усреднением"""
    
        Y = ycbcr_pixels[:, :, 0]
        Cb = ycbcr_pixels[:, :, 1]
        Cr = ycbcr_pixels[:, :, 2]
        
        original_height, original_width = Y.shape
        
        # Вычисляем размеры с паддингом
        padded_height = original_height + (original_height % 2)
        padded_width = original_width + (original_width % 2)
        
        # Создаем массивы с паддингом
        Y_padded = np.zeros((padded_height, padded_width), dtype=np.float32)
        Cb_padded = np.zeros((padded_height, padded_width), dtype=np.float32)
        Cr_padded = np.zeros((padded_height, padded_width), dtype=np.float32)
        
        # Копируем оригинальные данные
        Y_padded[:original_height, :original_width] = Y
        Cb_padded[:original_height, :original_width] = Cb
        Cr_padded[:original_height, :original_width] = Cr
        
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
        
        # Создаем выходной массив с паддингом
        subsampled_padded = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
        subsampled_padded[:, :, 0] = Y_padded
        subsampled_padded[:, :, 1] = np.repeat(np.repeat(Cb_subsampled, 2, axis=0), 2, axis=1)
        subsampled_padded[:, :, 2] = np.repeat(np.repeat(Cr_subsampled, 2, axis=0), 2, axis=1)
        
        # Обрезаем обратно до оригинальных размеров
        subsampled = subsampled_padded[:original_height, :original_width, :]
        
        self.logger.debug(f"""
    YCbCr-array subsample: Success
    Size of subsampled_pixels: {subsampled.shape}
    2x2 block:
        {subsampled[0][0]} {subsampled[0][1]}
        {subsampled[1][0]} {subsampled[1][1]}\n""")
        
        return subsampled
        
    @log_step
    def _split_into_blocks(self, subsampled):
        """Разбиение на блоки 8x8 с паддингом"""
        
        Y = subsampled[:, :, 0]
        Cb = subsampled[:, :, 1]
        Cr = subsampled[:, :, 2]
        
        original_height, original_width = Y.shape
        
        # Вычисляем размеры с паддингом до кратных 8
        padded_height = original_height + (8 - original_height % 8) if original_height % 8 != 0 else original_height
        padded_width = original_width + (8 - original_width % 8) if original_width % 8 != 0 else original_width
        
        # Создаем массивы с паддингом (дублируем граничные пиксели)
        Y_padded = np.zeros((padded_height, padded_width), dtype=np.float32)
        Cb_padded = np.zeros((padded_height, padded_width), dtype=np.float32)
        Cr_padded = np.zeros((padded_height, padded_width), dtype=np.float32)
        
        Y_padded[:original_height, :original_width] = Y
        Cb_padded[:original_height, :original_width] = Cb
        Cr_padded[:original_height, :original_width] = Cr
        
        # Заполняем паддинг дублированием граничных пикселей
        if original_height < padded_height:
            Y_padded[original_height:, :] = Y_padded[original_height-1:original_height, :]
            Cb_padded[original_height:, :] = Cb_padded[original_height-1:original_height, :]
            Cr_padded[original_height:, :] = Cr_padded[original_height-1:original_height, :]
        
        if original_width < padded_width:
            Y_padded[:, original_width:] = Y_padded[:, original_width-1:original_width]
            Cb_padded[:, original_width:] = Cb_padded[:, original_width-1:original_width]
            Cr_padded[:, original_width:] = Cr_padded[:, original_width-1:original_width]
        
        # Разбиваем на блоки 8x8
        def split_channel(channel):
            h, w = channel.shape
            return channel.reshape(h//8, 8, w//8, 8).transpose(0, 2, 1, 3)
        
        Y_blocks = split_channel(Y_padded)
        Cb_blocks = split_channel(Cb_padded)
        Cr_blocks = split_channel(Cr_padded)
        
        block_str = "\n\n".join("   ".join(f"{val:.5f}" for val in row) for row in Y_blocks[0][0])
        self.logger.debug(f"""
    Split into block 8x8: Success
    Number of blocks: {Y_blocks.shape[0]}x{Y_blocks.shape[1]}
    Size of Y: {Y_blocks.shape}
    First Y block:\n
{block_str}\n""")
        
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
        """Применяет DCT ко всем блокам всех каналов"""

        shifted_blocks_data = self._level_shift(blocks_data)

        Y_blocks = shifted_blocks_data['Y_shift_blocks']
        Cb_blocks = shifted_blocks_data['Cb_shift_blocks']
        Cr_blocks = shifted_blocks_data['Cr_shift_blocks']

        num_blocks_h, num_blocks_w = Y_blocks.shape[:2]
        
        # Создаем массивы для DCT коэффициентов
        Y_dct = np.zeros_like(Y_blocks)
        Cb_dct = np.zeros_like(Cb_blocks)
        Cr_dct = np.zeros_like(Cr_blocks)
        
        # Применяем DCT к каждому блоку каждого канала
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                Y_dct[i, j] = self._dct(Y_blocks[i, j])
                Cb_dct[i, j] = self._dct(Cb_blocks[i, j])
                Cr_dct[i, j] = self._dct(Cr_blocks[i, j])
                
        dct_block_str = "\n\n".join("   ".join(f"{val:>8.3f}" for val in row) for row in Y_dct[0][0])
        self.logger.debug(f"""
    DCT-encode: Success
    Number of dct blocks: {Y_dct.shape[0]}x{Y_dct.shape[1]}
    Size of dct-Y: {Y_dct.shape}
    DC-componet: {Y_dct[0][0][0][0]}
    First dct-Y block:\n
{dct_block_str}\n""")
        
        return {
            'Y_dct': Y_dct,
            'Cb_dct': Cb_dct, 
            'Cr_dct': Cr_dct,
        }
   
    @log_step
    def _apply_quantization(self, dct_blocks):
        """Применяет квантование ко всем DCT-блокам всех каналов"""

        Y_dct = dct_blocks['Y_dct']
        Cb_dct = dct_blocks['Cb_dct']
        Cr_dct = dct_blocks['Cr_dct']

        num_blocks_h, num_blocks_w = Y_dct.shape[:2]
        
        self.logger.debug(f"""
    STANDARD_LUMINANCE_QUANT_TABLE before scaling:
{self.STANDARD_LUMINANCE_QUANT_TABLE}\n""")
        
        self.logger.debug(f"""
    STANDARD_LUMINANCE_QUANT_TABLE after scaling (quality = {self.quality}):
{self._scale_quant_table(self.STANDARD_LUMINANCE_QUANT_TABLE, self.quality)}\n""")

        # Создаем массивы для квантованных коэффициентов
        Y_quant = np.zeros_like(Y_dct, dtype=np.int32)
        Cb_quant = np.zeros_like(Cb_dct, dtype=np.int32)
        Cr_quant = np.zeros_like(Cr_dct, dtype=np.int32)

        # Применяем квантование к каждому блоку
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                Y_quant[i, j] = np.round(Y_dct[i, j] / self._scale_quant_table(self.STANDARD_LUMINANCE_QUANT_TABLE, self.quality).astype(np.float32)).astype(np.int32)
                Cb_quant[i, j] = np.round(Cb_dct[i, j] / self._scale_quant_table(self.STANDARD_CHROMINANCE_QUANT_TABLE, self.quality).astype(np.float32)).astype(np.int32)
                Cr_quant[i, j] = np.round(Cr_dct[i, j] / self._scale_quant_table(self.STANDARD_CHROMINANCE_QUANT_TABLE, self.quality).astype(np.float32)).astype(np.int32)

        quant_block_str = "\n\n".join("   ".join(f"{val:>8.3f}" for val in row) for row in Y_quant[0][0])
        self.logger.debug(f"""
    Quantization: Success
    Number of quant blocks: {Y_quant.shape[0]}x{Y_quant.shape[1]}
    Size of quant-Y: {Y_quant.shape}
    First quant-Y block:\n
{quant_block_str}\n""")
        
        return {
            'Y_quant': Y_quant,
            'Cb_quant': Cb_quant,
            'Cr_quant': Cr_quant
        }

    @log_step
    def _dc_differentiation(self, quantized):
        """Дифференциальное кодирование DC-коэффициентов"""

        def diff_dc(blocks):
            # Извлекаем DC-компоненты всех блоков (заранее преобразовав блок 8х8 в одномерный массив с поомщью flatten())
            dc = blocks[:, :, 0, 0].flatten()
            # Дифференциальное кодирование (первый элемент берём как есть)
            return np.diff(np.insert(dc, 0, 0))
        
        self.logger.debug(f"""
    DC-differentiation: Success
    Size of diff-Y: {diff_dc(quantized['Y_quant']).shape[0]}
    First eight diff components:
        {diff_dc(quantized['Y_quant'])[:8]}""")
        
        return {
            'Y_dc_diff': diff_dc(quantized['Y_quant']),
            'Cb_dc_diff': diff_dc(quantized['Cb_quant']),
            'Cr_dc_diff': diff_dc(quantized['Cr_quant']),
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

    def _value_to_bits(self, value: int) -> str:
        """Преобразует значение коэффициента в 'сырые' биты (по JPEG)."""
        if value == 0:
            return ""
        size = int(math.floor(math.log2(abs(value)))) + 1
        if value > 0:
            return format(value, f"0{size}b")
        else:
            max_val = (1 << size) - 1
            return format(value + max_val, f"0{size}b")

    def _run_length_encoding(self, ac_coeffs):
        """Выполняет JPEG RLE-кодирование AC-коэффициентов: (run, size) + битовое значение."""
        rle = []
        values = []
        zero_run = 0

        for coeff in ac_coeffs:
            if coeff == 0:
                zero_run += 1
                if zero_run == 16:
                    rle.append((15, 0))  # ZRL
                    values.append("")
                    zero_run = 0
            else:
                size = int(math.floor(math.log2(abs(coeff)))) + 1
                rle.append((zero_run, size))
                values.append(self._value_to_bits(coeff))
                zero_run = 0

        # Если остались нули в конце блока → EOB
        if zero_run > 0:
            rle.append((0, 0))  # EOB
            values.append("")

    #     self.logger.debug(f"""
    # RLE-encoding: Success
    # Size of AC_coeffs-array: {len(ac_coeffs)}
    # AC_coeffs:
    #     {[int(el) for el in ac_coeffs]}

    # Size of RLE-array: {len(rle)}
    # RLE-array:
    #     {rle}
    # values:
    #     {values}
    # """)

        return {"rle": rle, "values": values}


    @log_step
    def _encode_rle_ac_components(self, quantized):
        """Применяет зигзаг и RLE к AC-компонентам всех блоков"""
    
        def process_channel(channel_blocks):
            h, w = channel_blocks.shape[:2]
            rle_blocks = []
            for i in range(h):
                for j in range(w):
                    block = channel_blocks[i, j]
                    zigzag = self._zigzag_scanning(block)
                    ac = zigzag[1:]  # исключаем DC
                    rle = self._run_length_encoding(ac)
                    rle_blocks.append(rle)
            return rle_blocks
        
        Y_ac_rle = process_channel(quantized['Y_quant'])
        Cb_ac_rle = process_channel(quantized['Cb_quant'])
        Cr_ac_rle = process_channel(quantized['Cr_quant'])
        
        
        self.logger.debug(f"""
    RLE-encoding: Success
    Size of first Y-RLE-array: {len(Y_ac_rle[0]["rle"])}
    RLE-array:
        {Y_ac_rle[0]["rle"]}
    values:
        {Y_ac_rle[0]["values"]}
    """)

        return {
            'Y_ac_rle': Y_ac_rle,
            'Cb_ac_rle': Cb_ac_rle,
            'Cr_ac_rle': Cr_ac_rle,
        }

    @log_step
    def _huffman_encoding(self, rle_data):
        """Хаффман-кодирование"""
        pass
    
    @log_step
    def _create_jpeg(self, encoded_data, output_path):
        """Создание итогового JPEG файла из закодированных данных"""
        pass
    
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
        dict_dc_diff = self._dc_differentiation(dict_quant_blocks)
        #one_zigzag_Y_array = self._zigzag_scanning(dict_quant_blocks['Y_quant'][0][0])
        #one_rle_array = self._run_length_encoding(one_zigzag_Y_array[1:])
        dict_list_dict_rle_values = self._encode_rle_ac_components(dict_quant_blocks)