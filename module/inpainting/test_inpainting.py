import unittest
import numpy as np
import cv2
import os
import asyncio
import pytest
import pytest_asyncio
from pathlib import Path

from module.config import InpainterConfig, Inpainter, InpaintPrecision
from module.inpainting.common import CommonInpainter, OfflineInpainter
from module.inpainting.inpainting_lama_mpe import LamaMPEInpainter

# Đường dẫn đến thư mục chứa dữ liệu test
TEST_DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "test_data"

print(TEST_DATA_DIR)
os.makedirs(TEST_DATA_DIR, exist_ok=True)

@pytest.mark.asyncio
class TestInpainting(unittest.TestCase):
    """Test cho module inpainting"""
    
    def setUp(self):
        """Thiết lập môi trường test"""
        # Tạo ảnh test đơn giản
        self.test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255  # Ảnh trắng
        
        # Vẽ một hình chữ nhật đen lên ảnh
        cv2.rectangle(self.test_image, (200, 200), (300, 300), (0, 0, 0), -1)
        
        # Tạo mask (vùng cần inpaint là hình chữ nhật đen)
        self.test_mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.rectangle(self.test_mask, (200, 200), (300, 300), 255, -1)
        
        # Lưu ảnh và mask để kiểm tra thủ công
        test_image_path = os.path.join(TEST_DATA_DIR, "test_image.png")
        test_mask_path = os.path.join(TEST_DATA_DIR, "test_mask.png")
        
        cv2.imwrite(test_image_path, self.test_image)
        cv2.imwrite(test_mask_path, self.test_mask)
        
        # Cấu hình mặc định cho inpainting
        self.default_config = InpainterConfig(
            inpainter=Inpainter.lama_large,
            inpainting_size=512,  # Kích thước nhỏ hơn cho test
            inpainting_precision=InpaintPrecision.fp32  # Precision thấp hơn cho test
        )
        
    @pytest.mark.asyncio
    async def test_lama_mpe_inpainter(self):
        """Test LamaMPEInpainter với ảnh và mask đơn giản"""
        # Khởi tạo inpainter
        inpainter = LamaMPEInpainter()
        
        # Tải mô hình
        await inpainter.load(device="cpu")
        
        # Thực hiện inpainting
        result = await inpainter.inpaint(
            image=self.test_image,
            mask=self.test_mask,
            config=self.default_config,
            inpainting_size=512,
            verbose=True
        )
        
        # Lưu kết quả để kiểm tra thủ công
        result_path = os.path.join(TEST_DATA_DIR, "result_lama_mpe.png")
        cv2.imwrite(result_path, result)
        
        # Kiểm tra kết quả
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Kiểm tra khu vực đã được inpaint không còn chứa màu đen
        roi = result[200:300, 200:300]
        black_pixels = np.sum((roi == [0, 0, 0]).all(axis=2))
        self.assertLess(black_pixels / roi.size, 0.1)  # Ít hơn 10% pixel đen trong vùng đã inpaint
        
        # Giải phóng tài nguyên
        await inpainter.unload()

    @pytest.mark.asyncio
    async def test_different_inpainters(self):
        """Test các loại inpainter khác nhau"""
        # Danh sách các inpainter cần test
        inpainter_types = [
            (Inpainter.lama_mpe, LamaMPEInpainter()),
            # Thêm các loại inpainter khác khi cần
        ]
        
        for inpainter_type, inpainter_instance in inpainter_types:
            # Cấu hình cho inpainter cụ thể
            config = InpainterConfig(
                inpainter=inpainter_type,
                inpainting_size=512,
                inpainting_precision=InpaintPrecision.fp32
            )
            
            # Tải mô hình
            await inpainter_instance.load(device="cpu")
            
            # Thực hiện inpainting
            result = await inpainter_instance.inpaint(
                image=self.test_image,
                mask=self.test_mask,
                config=config,
                inpainting_size=512,
                verbose=True
            )
            
            # Lưu kết quả để kiểm tra thủ công
            result_path = os.path.join(TEST_DATA_DIR, f"result_{inpainter_type.value}.png")
            cv2.imwrite(result_path, result)
            
            # Kiểm tra kết quả
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, self.test_image.shape)
            
            # Giải phóng tài nguyên
            await inpainter_instance.unload()
            
    async def test_different_config(self):
        """Test với các cấu hình InpainterConfig khác nhau"""
        # Khởi tạo inpainter
        inpainter = LamaMPEInpainter()
        
        # Tải mô hình
        await inpainter.load(device="cpu")
        
        # Danh sách các cấu hình cần test
        configs = [
            InpainterConfig(inpainter=Inpainter.lama_mpe, inpainting_size=256, inpainting_precision=InpaintPrecision.fp32),
            InpainterConfig(inpainter=Inpainter.lama_mpe, inpainting_size=512, inpainting_precision=InpaintPrecision.fp32),
            # Thêm các cấu hình khác khi cần
        ]
        
        for idx, config in enumerate(configs):
            # Thực hiện inpainting
            result = await inpainter.inpaint(
                image=self.test_image,
                mask=self.test_mask,
                config=config,
                inpainting_size=config.inpainting_size,
                verbose=True
            )
            
            # Lưu kết quả để kiểm tra thủ công
            result_path = os.path.join(TEST_DATA_DIR, f"result_config_{idx}.png")
            cv2.imwrite(result_path, result)
            
            # Kiểm tra kết quả
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, self.test_image.shape)
        
        # Giải phóng tài nguyên
        await inpainter.unload()
        
    def test_create_custom_mask(self):
        """Test tạo mask tùy chỉnh cho inpainting"""
        # Tạo mask hình tròn
        circle_mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(circle_mask, (256, 256), 100, 255, -1)
        
        # Lưu mask để kiểm tra thủ công
        circle_mask_path = os.path.join(TEST_DATA_DIR, "circle_mask.png")
        cv2.imwrite(circle_mask_path, circle_mask)
        
        # Kiểm tra mask
        self.assertEqual(circle_mask.shape, (512, 512))
        self.assertTrue(np.any(circle_mask > 0))  # Mask có pixel trắng
        
        # Tạo mask dạng text
        text_mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.putText(text_mask, "TEST", (150, 256), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5)
        
        # Lưu mask để kiểm tra thủ công
        text_mask_path = os.path.join(TEST_DATA_DIR, "text_mask.png")
        cv2.imwrite(text_mask_path, text_mask)
        
        # Kiểm tra mask
        self.assertEqual(text_mask.shape, (512, 512))
        self.assertTrue(np.any(text_mask > 0))  # Mask có pixel trắng


if __name__ == "__main__":
    unittest.main()
