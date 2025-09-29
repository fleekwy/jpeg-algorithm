from jpeg_compressor import JpegCompressor

def test():
    """Проверка load_image и _rgb_to_ycbcr"""
    print("=== Тестирование загрузки и преобразования ===")
    
    # 1. Создаем компрессор
    compressor = JpegCompressor()
    print("✅ Компрессор создан")
    
    # 2. Загружаем изображение
    try:
        compressor.load_image("data/test_image_1.jpg", quality=85)
        print("✅ Изображение загружено")
        print(f"   Размер: {compressor._original_pixels.shape}")
        print(f"   Качество: {compressor.quality}")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # 3. Преобразуем в YCbCr
    try:
        compressor._rgb_to_ycbcr()
        print("✅ Преобразование RGB → YCbCr выполнено")
        
        # Проверяем результаты
        y_channel = compressor._pixels[:, :, 0]
        print(f"Y диапазон: [{y_channel.min():.1f}, {y_channel.max():.1f}]")
        print(f"Соответствие стандарту: {y_channel.min() >= 16 and y_channel.max() <= 235}")
        
    except Exception as e:
        print(f"❌ Ошибка преобразования: {e}")

if __name__ == "__main__":
    test()