# Hướng dẫn test module inpainting

Tài liệu này hướng dẫn cách chạy và mở rộng các test cho module inpainting.

## Yêu cầu

- Python 3.8+
- pytest
- pytest-asyncio (sẽ được tự động cài đặt nếu chưa có)

## Cấu trúc test

Hệ thống test bao gồm:
- `test_inpainting.py`: Chứa các test case cho module inpainting
- `run_tests.py`: Script để chạy các test
- `test_data/`: Thư mục chứa dữ liệu test (được tạo tự động)

## Cách chạy test

### Chạy tất cả các test

```bash
python module/inpainting/run_tests.py
```

### Chạy test với tùy chọn verbose

```bash
python module/inpainting/run_tests.py --verbose
```

### Chạy một test cụ thể

```bash
python module/inpainting/run_tests.py --test-name test_lama_mpe_inpainter
```

### Chạy test trên GPU (nếu có)

```bash
python module/inpainting/run_tests.py --device cuda
```

## Các test case hiện có

1. **test_lama_mpe_inpainter**: Test LamaMPEInpainter với ảnh và mask đơn giản
2. **test_different_inpainters**: Test các loại inpainter khác nhau
3. **test_different_config**: Test với các cấu hình InpainterConfig khác nhau
4. **test_create_custom_mask**: Test tạo mask tùy chỉnh cho inpainting

## Cách thêm test case mới

Để thêm một test case mới, hãy làm theo các bước sau:

1. Mở file `test_inpainting.py`
2. Thêm một phương thức mới bắt đầu bằng `test_` (không cần decorator `@pytest.mark.asyncio` vì đã được đặt ở cấp lớp)
3. Nếu là test bất đồng bộ, hãy khai báo phương thức với từ khóa `async`
4. Viết logic test và các assertion

Ví dụ:

```python
async def test_my_new_test(self):
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
    
    # Kiểm tra kết quả
    self.assertIsNotNone(result)
    
    # Giải phóng tài nguyên
    await inpainter.unload()
```

## Xem kết quả test

Sau khi chạy test, các ảnh kết quả sẽ được lưu trong thư mục `module/inpainting/test_data/`. Bạn có thể kiểm tra các ảnh này để đánh giá chất lượng inpainting.

## Lưu ý

- Các test mặc định sẽ chạy trên CPU để đảm bảo tính ổn định và khả năng chạy trên mọi máy
- Nếu muốn chạy test trên GPU, hãy sử dụng tùy chọn `--device cuda`
- Một số test có thể mất nhiều thời gian do phải tải mô hình và thực hiện inpainting
- Đảm bảo đã cài đặt đầy đủ các thư viện cần thiết trước khi chạy test

## Xử lý lỗi phổ biến

### Lỗi "Unknown pytest.mark.asyncio"

Nếu bạn gặp cảnh báo về "Unknown pytest.mark.asyncio", đảm bảo rằng:
1. Đã cài đặt pytest-asyncio: `pip install pytest-asyncio`
2. Sử dụng tùy chọn `--asyncio-mode=auto` khi chạy pytest

### Lỗi "coroutine was never awaited"

Nếu bạn gặp cảnh báo về "coroutine was never awaited", đảm bảo rằng:
1. Lớp test đã được đánh dấu với `@pytest.mark.asyncio`
2. Các phương thức test bất đồng bộ được khai báo với từ khóa `async`
3. Sử dụng pytest-asyncio để chạy test
