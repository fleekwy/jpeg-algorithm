import sys
import numpy as np
from PIL import Image
import logging

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
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info("__init__:  <start>")
        
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
            
        self.logger.info("__init__:  <end>")
        
        
# protected

    def _reset_state(self):
        
        self.logger.info("_reset_state:  <start>")
        
        self._original_pixels = None
        self._compressed_pixels = None
        self.quality = None
        
        self.logger.info("_reset_state:  <end>")


    def _create_dctII_matrix(self, N):
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
            
        #self.logger.info(f"{scale=}")

        # ВАЖНО: привести к float32, чтобы избежать переполнения
        table = table.astype(np.float32)

        scaled = np.floor((table * scale + 50) / 100)
        return np.clip(scaled, 1, 255).astype(np.uint8)


    def _rgb_to_ycbcr(self, rgb_pixels):
        """Конвертация RGB в YCbCr"""
        
        self.logger.info("_rgb_to_ycbcr:  <start>")
        
        
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
        
        self.logger.info("_rgb_to_ycbcr:  <end>")
        return ycbcr_pixels
        
    
    def _chroma_subsampling(self, ycbcr_pixels):
        """Хроматическое прореживание 4:2:0 с паддингом и усреднением"""
    
        self.logger.info("_chroma_subsampling:  <start>")
    
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
        
        self.logger.info("_chroma_subsampling:  <end>")
        
        return subsampled
        
    
    def _split_into_blocks(self, subsampled):
        """Разбиение на блоки 8x8 с паддингом"""
        
        self.logger.info(f"_split_into_blocks: <start>")
        
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
        
        self.logger.info(f"_split_into_blocks: <end>")
        
        return {
            'Y_blocks': Y_blocks,
            'Cb_blocks': Cb_blocks,
            'Cr_blocks': Cr_blocks
        }
    
    
    # def _dct(self, block):
    #     """Применяет 2D DCT к одному блоку 8x8. Возвращает блок 8x8 DCT коэффициентов"""
        
    #     # Используем формулу: DCT(u,v) = C(u)C(v) * sum_{x=0}^{7} sum_{y=0}^{7} block(x,y) * cos(...) * cos(...)
        
    #     dct_block = np.zeros((8, 8), dtype=np.float32)
        
    #     for u in range(8):
    #         for v in range(8):
    #             sum_val = 0.0
    #             for x in range(8):
    #                 for y in range(8):
    #                     sum_val += block[x, y] * np.cos((2*x + 1) * u * np.pi / 16) * np.cos((2*y + 1) * v * np.pi / 16)
                
    #             # Нормализующие коэффициенты
    #             cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
    #             cv = 1.0 / np.sqrt(2) if v == 0 else 1.0
                
    #             dct_block[u, v] = 0.25 * cu * cv * sum_val
        
    #     return dct_block
    
    
    def _dct(self, block):
        """Применяет 2D DCT к одному блоку 8x8. Возвращает блок 8x8 DCT коэффициентов"""
        block = block - 128
        return self.DCT_MATRIX @ block @ self.DCT_MATRIX.T
    

    def _apply_dct(self, blocks_data):
        """Применяет DCT ко всем блокам всех каналов"""
        
        self.logger.info(f"_apply_dct: <start>")
        
        Y_blocks = blocks_data['Y_blocks']
        Cb_blocks = blocks_data['Cb_blocks']
        Cr_blocks = blocks_data['Cr_blocks']
        
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
        
        self.logger.info("_apply_dct: <end>")
        
        return {
            'Y_dct': Y_dct,
            'Cb_dct': Cb_dct, 
            'Cr_dct': Cr_dct,
        }
   
    
    def _apply_quantization(self, dct_blocks):
        """Применяет квантование ко всем DCT-блокам всех каналов"""

        self.logger.info(f"_apply_quantization: <start>")

        Y_dct = dct_blocks['Y_dct']
        Cb_dct = dct_blocks['Cb_dct']
        Cr_dct = dct_blocks['Cr_dct']

        num_blocks_h, num_blocks_w = Y_dct.shape[:2]

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

        self.logger.info("_apply_quantization: <end>")

        return {
            'Y_quant': Y_quant,
            'Cb_quant': Cb_quant,
            'Cr_quant': Cr_quant
        }


    def _dc_differentiation(self, quantized):
        """Дифференциальное кодирование DC-коэффициентов"""
        
        self.logger.info("_dc_differentiation: <start>")

        def diff_dc(blocks):
            # Извлекаем DC-компоненты всех блоков (заранее преобразовав блок 8х8 в одномерный массив с поомщью flatten())
            dc = blocks[:, :, 0, 0].flatten()
            # Дифференциальное кодирование (первый элемент берём как есть)
            return np.diff(np.insert(dc, 0, 0))

        self.logger.info("_dc_differentiation: <end>")
        
        return {
            'Y_dc_diff': diff_dc(quantized['Y_quant']),
            'Cb_dc_diff': diff_dc(quantized['Cb_quant']),
            'Cr_dc_diff': diff_dc(quantized['Cr_quant']),
        }


    def _zigzag_scanning(self, block):
        """Зигзаг-сканирование одного блока 8x8"""
        pass


    def _run_length_encoding(self, ac_coeffs):
        """RLE кодирование AC-компонент"""
        pass


    def _huffman_encoding(self, rle_data):
        """Хаффман-кодирование"""
        pass
    
    
    def _create_jpeg(self, encoded_data, output_path):
        """Создание итогового JPEG файла из закодированных данных"""
        pass
    
    
# public

    def load_image(self, image_path, quality=75):
        """Загрузка изображения (сбрасывает предыдущее состояние)"""
        
        self.logger.info("load_image:  <start>")
        
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
        
        self.logger.info("load_image:  <end>")


    def compress(self, compressed_image_name: str):
        """Основной метод сжатия"""
        
        if self._original_pixels is None:
            raise ValueError("Error: _original_pixels is None! First download image with \"load_image()\"")
            
        rgb_pixels = self._original_pixels
        try:
            ycbcr_pixels = self._rgb_to_ycbcr(rgb_pixels)
            self.logger.debug(f"""
    Transform RGB → YCbCr: success
    Size of ycbcr_pixels: {ycbcr_pixels.shape}
    2x2 block:
        {ycbcr_pixels[0][0]} {ycbcr_pixels[0][1]}
        {ycbcr_pixels[1][0]} {ycbcr_pixels[1][1]}\n""")
            
        except Exception as e:
            raise RuntimeError(f"Transforming RGB → YCbCr Error: {e}")
        
        try:
            subsampled = self._chroma_subsampling(ycbcr_pixels)
            self.logger.debug(f"""
    YCbCr-array subsample: success
    Size of subsampled_pixels: {subsampled.shape}
    2x2 block:
        {subsampled[0][0]} {subsampled[0][1]}
        {subsampled[1][0]} {subsampled[1][1]}\n""")
            
        except Exception as e:
            raise RuntimeError(f"YCbCr-array subsampling Error: {e}")
        
        try:
            dict_blocks = self._split_into_blocks(subsampled)
            block_str = "\n\n".join("   ".join(f"{val:.5f}" for val in row) for row in dict_blocks['Y_blocks'][0][0])
            self.logger.debug(f"""
    Split into block 8x8: success
    Number of blocks: {dict_blocks['Y_blocks'].shape[0]}x{dict_blocks['Y_blocks'].shape[1]}
    Size of Y: {dict_blocks['Y_blocks'].shape}
    First Y block:\n
{block_str}\n""")
            
        except Exception as e:
            raise RuntimeError(f"Splittings into block 8x8 Error: {e}")
        
        try:
            dict_dct_blocks = self._apply_dct(dict_blocks)
            dct_block_str = "\n\n".join("   ".join(f"{val:.3f}" for val in row) for row in dict_dct_blocks['Y_dct'][0][0])
            self.logger.debug(f"""
    DCT-encode: success
    Number of dct blocks: {dict_dct_blocks['Y_dct'].shape[0]}x{dict_dct_blocks['Y_dct'].shape[1]}
    Size of dct-Y: {dict_dct_blocks['Y_dct'].shape}
    DC-componet: {dict_dct_blocks['Y_dct'][0][0][0][0]}
    First dct-Y block:\n
{dct_block_str}\n""")
            
        except Exception as e:
            raise RuntimeError(f"DCT-encoding Error: {e}")
        
        try:
            dict_quant_blocks = self._apply_quantization(dict_dct_blocks)
            quant_block_str = "\n\n".join("   ".join(f"{val:.3f}" for val in row) for row in dict_quant_blocks['Y_quant'][0][0])
            self.logger.debug(f"""
    Quantization: success
    Number of quant blocks: {dict_quant_blocks['Y_quant'].shape[0]}x{dict_quant_blocks['Y_quant'].shape[1]}
    Size of quant-Y: {dict_quant_blocks['Y_quant'].shape}
    First quant-Y block:\n
{quant_block_str}\n""")
            
        except Exception as e:
            raise RuntimeError(f"Quantization Error: {e}")
        
        try:
            dict_dc_diff = self._dc_differentiation(dict_quant_blocks)
            self.logger.debug(f"""
    DC-differentiation: success
    Size of diff-Y: {dict_dc_diff['Y_dc_diff'].shape}
    First eight diff components:
        {dict_dc_diff['Y_dc_diff'][:8]}""")
            
        except Exception as e:
            raise RuntimeError(f"DC-differentiation Error: {e}")