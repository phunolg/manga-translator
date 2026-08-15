#!/usr/bin/env python3
"""
Script chạy các test cho module inpainting
"""

import os
import sys
import pytest
import argparse
from pathlib import Path


def main():
    """
    Hàm chính để chạy các test
    """
    parser = argparse.ArgumentParser(description='Chạy test cho module inpainting')
    parser.add_argument('--verbose', '-v', action='store_true', help='Hiển thị chi tiết quá trình test')
    parser.add_argument('--test-name', '-t', type=str, help='Chỉ chạy test có tên cụ thể')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Thiết bị để chạy test (cpu, cuda)')
    
    args = parser.parse_args()
    
    # Thêm thư mục gốc vào sys.path để import module
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    sys.path.insert(0, str(root_dir))
    
    # Chuẩn bị các tham số cho pytest
    pytest_args = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_inpainting.py'),
        '-v' if args.verbose else '',
        '--asyncio-mode=auto',  # Thêm tùy chọn cho pytest-asyncio
    ]
    
    # Thêm biến môi trường để truyền thiết bị cho test
    os.environ['TEST_DEVICE'] = args.device
    
    # Nếu có tên test cụ thể, chỉ chạy test đó
    if args.test_name:
        pytest_args.append(f'-k {args.test_name}')
    
    # Chạy pytest
    return pytest.main(pytest_args)

if __name__ == '__main__':
    sys.exit(main())
