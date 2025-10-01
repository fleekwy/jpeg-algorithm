import numpy as np
from PIL import Image

class JpegCompressor:
    """Класс JPEG-компрессора.
    
    Attributes:
        _original_pixels (np.ndarray): Исходное изображение в виде матрицы пикселей
        _pixels (np.ndarray): Текущее состояние изображения в процессе обработки
        original_image_path (str): Путь к исходному файлу изображения
        quality (int): Качество сжатия от 1 до 100
    """
    
    
# private 
    
    def __init__(self, image_path=None):
        """Первоначальная инициализация состояния объекта"""
        
        print("\n__init__:  <start>")
        
        self._original_pixels = None
        self._compressed_pixels = None
        self.quality = None
        
        if image_path:
            self.load_image(image_path)
            
        print("__init__:  <end>\n")
        
        
# protected

    def _rgb_to_ycbcr(self, rgb_pixels):
        """Конвертация RGB в YCbCr"""
        
        print("\n_rgb_to_ycbcr:  <start>")
        
        
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
        
        print("_rgb_to_ycbcr:  <end>\n")
        return ycbcr_pixels
        
    
    def _chroma_subsampling(self, ycbcr_pixels):
        """Хроматическое прореживание 4:2:0 с паддингом и усреднением"""
    
        print("_chroma_subsampling:  <start>")
    
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
        
        return subsampled
        
    
    def _split_into_blocks(self, subsampled):
        """Разбиение на блоки 8x8"""
        pass

    def apply_dct(self, block):
        """Применение дискретного косинусного преобразования"""
        pass

    def _quantization(self, dct_block, channel_type='Y'):
        """Квантование DCT-коэффициентов"""
        pass
    
    def _dc_differentiation(self, quantized):
        """Дифференциальное кодирование DC-коэффициентов"""
        pass

    def _zigzag_scanning(self, matrix):
        """Зигзаг-сканирование"""
        pass

    def _run_length_encoding(self, data):
        """RLE кодирование"""
        pass

    def _huffman_encoding(self, data):
        """Кодирование Хаффмана"""
        pass
    
    
# public

    def load_image(self, image_path, quality=75):
        """Загрузка изображения (сбрасывает предыдущее состояние)"""
        
        print("\nload_image:  <start>")
        
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
            raise ValueError(f"Ошибка загрузки изображения: {e}")
        
        print("load_image:  <end>\n")
        
        
    def _reset_state(self):
        
        print("_reset_state:  <start>")
        
        self._original_pixels = None
        self._compressed_pixels = None
        self.quality = None
        
        print("_reset_state:  <end>")
        
        
    def _create_jpeg(self, encoded_data, output_path):
        """Создание итогового JPEG файла из закодированных данных"""
        pass


    def compress(self, compressed_image_name: str):
        """Основной метод сжатия"""
        
        if self._original_pixels is None:
            raise ValueError("Сначала загрузите изображение с помощью load_image()")
            
        rgb_pixels = self._original_pixels
        ycbcr_pixels = self._rgb_to_ycbcr(rgb_pixels)
        subsampled = self._chroma_subsampling(ycbcr_pixels)