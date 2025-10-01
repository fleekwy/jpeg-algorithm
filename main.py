from jpeg_compressor import JpegCompressor
        
def test1():
    print("\n=== test1() ===")
    
    compressor = JpegCompressor()
    
    try:
        compressor.load_image("data/test_image_1.jpg", quality=85)
        print("✅ Изображение загружено")
        print(f"   Размер: {compressor._original_pixels.shape}")

    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    try:
        rgb = compressor._original_pixels.copy()
        ycbcr_pixels = compressor._rgb_to_ycbcr(rgb)
        print("✅ Преобразование RGB → YCbCr выполнено")
        print(f"   Размер: {ycbcr_pixels.shape}")
        
    except Exception as e:
        print(f"❌ Ошибка преобразования: {e}")
        

if __name__ == "__main__":
    test1()