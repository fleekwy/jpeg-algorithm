import numpy as np
from PIL import Image

class JpegCompressor:
    
# private 
    
    def __init__(self, image_path: str):
        
        self.original_image_path = image_path
        
        try:
            self.original_image = Image.open(self.original_image_path)
            
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            raise
        
        self.original_data = np.array(self.original_image)
        self.compressed_data = None
        self.compressed_image = None    
        
        
# protected

    def _rgb_to_ycbcr(self):
        """Конвертация RGB в YCbCr"""
        pass

    def _chroma_subsampling(self, ratio='4:2:0'):
        """Хроматическое прореживание"""
        pass

    def apply_dct(self, block):
        """Применение дискретного косинусного преобразования"""
        pass

    def _quantization(self, dct_block, channel_type='Y'):
        """Квантование DCT-коэффициентов"""
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

    def compress(self, compressed_image_path: str):
        """Основной метод сжатия"""
        pass
