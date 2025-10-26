from jpeg_compressor import JpegCompressor
    
def main():
    compressor = JpegCompressor()
    
    compressor.compress("data/test_image.jpg", "compressed_image_q75", 75)
    compressor.compress("data/test_image.jpg", "compressed_image_q20", 20)
    compressor.compress("data/test_image.jpg", "compressed_image_q15", 15)
    compressor.compress("data/test_image.jpg", "compressed_image_q10", 10)
    compressor.compress("data/test_image.jpg", "compressed_image_q8", 8)
    compressor.compress("data/test_image.jpg", "compressed_image_q6", 6)
    compressor.compress("data/test_image.jpg", "compressed_image_q4", 4)
    compressor.compress("data/test_image.jpg", "compressed_image_q1", 1)
    
def ui():
    compressor = JpegCompressor()
    
    quality = int(input("Введите желаемое качество сжатого изображения (от 1 до 100, and press Enter)):\n"))
    show = input("Хотите ли вы чтобы сжатое изображение сразу же всплывало на экране? ([y]/n, and press Enter):\n")
    compressor.compress("data/test_image.jpg", f"compressed_image_q{quality}", quality, show)

if __name__ == "__main__":
    main()