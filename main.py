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

if __name__ == "__main__":
    main()