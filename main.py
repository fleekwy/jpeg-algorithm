from jpeg_compressor import JpegCompressor
        
def test1():
    print("\n=== test1() ===")
    
    compressor = JpegCompressor()
    
    try:
        compressor.load_image("data/test_image_2.jpg")
        print("✅ Изображение загружено:")
        print(f"   Размер: {compressor._original_pixels.shape}")
        rgb = compressor._original_pixels.copy()
        print(rgb[0][0], rgb[0][1])
        print(rgb[1][0], rgb[1][1])

    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    try:
        ycbcr_pixels = compressor._rgb_to_ycbcr(rgb)
        print("✅ Преобразование RGB → YCbCr выполнено:")
        print(f"   Размер: {ycbcr_pixels.shape}")
        print(ycbcr_pixels[0][0], ycbcr_pixels[0][1])
        print(ycbcr_pixels[1][0], ycbcr_pixels[1][1])
        
    except Exception as e:
        print(f"❌ Ошибка преобразования: {e}")
        
    try:
        subsampled = compressor._chroma_subsampling(ycbcr_pixels)
        print("✅ Прореживание YCbCr-массива выполнено:")
        print(f"   Размер: {subsampled.shape}")
        print(subsampled[0][0], subsampled[0][1])
        print(subsampled[1][0], subsampled[1][1])
    except Exception as e:
        print(f"❌ Ошибка прореживания: {e}")
        
        
    try:
        dict_blocks = compressor._split_into_blocks(subsampled)
        print("✅ Разделение на блоки 8х8 выполнено:")
        print(dict_blocks['Y_blocks'][0][0])
        print(f"   Блоков: {dict_blocks['Y_blocks'].shape[0]}x{dict_blocks['Y_blocks'].shape[1]}")
        print(f"   Размер: {dict_blocks['Y_blocks'].shape}")
    except Exception as e:
        print(f"❌ Ошибка разделения на блоки 8х8: {e}")
        
        
    try:
        dict_dct_blocks = compressor._apply_dct(dict_blocks)
        print("✅ DCT-кодирование выполнено:")
        print(dict_dct_blocks['Y_dct'][0][0])
        print(f"   Обработано блоков: {dict_dct_blocks['Y_dct'].shape[0] * dict_dct_blocks['Y_dct'].shape[1]} на канал")
        print(f"   Размер: {dict_dct_blocks['Y_dct'].shape}")
        
    except Exception as e:
        print(f"❌ Ошибка DCT-кодирования: {e}")
        
        
    try:
        dict_quant_blocks = compressor._apply_quantization(dict_dct_blocks)
        print("✅ Квантование блоков выполнено:")
        print(dict_quant_blocks['Y_quant'][0][0])
        print(f'   {compressor.quality=}')
        print(f"   Размер: {dict_quant_blocks['Y_quant'].shape}")
        
    except Exception as e:
        print(f"❌ Ошибка квантования: {e}")
    
    
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
    

if __name__ == "__main__":
    #test_quant()
    test1()