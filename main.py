from jpeg_compressor import JpegCompressor
from PIL import Image, ImageFile
import numpy as np
    
def test_scale():
    compressor2 = JpegCompressor()
    print("Исходная STANDARD_LUMINANCE_QUANT_TABLE:")
    print(compressor2.STANDARD_LUMINANCE_QUANT_TABLE)
    print("Отмасштабировання STANDARD_LUMINANCE_QUANT_TABLE при quality = 75")
    print(compressor2._scale_quant_table(compressor2.STANDARD_LUMINANCE_QUANT_TABLE, 75))
    print("Отмасштабировання STANDARD_LUMINANCE_QUANT_TABLE при quality = 50")
    print(compressor2._scale_quant_table(compressor2.STANDARD_LUMINANCE_QUANT_TABLE, 50))
    print("Отмасштабировання STANDARD_LUMINANCE_QUANT_TABLE при quality = 25")
    print(compressor2._scale_quant_table(compressor2.STANDARD_LUMINANCE_QUANT_TABLE, 25))
    
    
def test():
    compressor = JpegCompressor()
    compressor.compress("data/test_image_3.jpg", "output_test_image_3")

def test_rle():
    compressor = JpegCompressor()
    ac=[0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    res=compressor._run_length_encoding(ac)
    print(res["rle"])
    
def output_test():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open("data/output_test_image_3.jpg")
    img.load()

if __name__ == "__main__":
    test()