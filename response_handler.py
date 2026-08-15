# response_factory.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import io
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import zipfile
import base64
import numpy as np


class ResponseStrategy(ABC):
    @abstractmethod
    def process_images(
        self,
        processed_images: List[Image.Image],
        page_names: List[str],
        original_images: Optional[List[Image.Image]] = None,
    ) -> Any:
        pass


class ZipResponseStrategy(ResponseStrategy):
    def process_images(
        self,
        processed_images: List[Image.Image],
        page_names: List[str],
        original_images: Optional[List[Image.Image]] = None,
    ) -> StreamingResponse:
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, img in enumerate(processed_images):
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                page_name = page_names[i].split(".")[0]
                zip_file.writestr(
                    f"{page_name}.png", img_buffer.getvalue())

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=translated_pages.zip"}
        )


class JsonResponseStrategy(ResponseStrategy):
    def process_images(
        self,
        processed_images: List[Image.Image],
        page_names: List[str],
        original_images: Optional[List[Image.Image]] = None,
    ) -> JSONResponse:
        images_data = []

        for i, img in enumerate(processed_images):
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

            images_data.append({
                "page_name": page_names[i],
                "image_base64": f"data:image/png;base64,{img_str}"
            })

        return JSONResponse(content={"images": images_data})


class InlinePairResponseStrategy(ResponseStrategy):
    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

    def process_images(
        self,
        processed_images: List[Image.Image],
        page_names: List[str],
        original_images: Optional[List[Image.Image]] = None,
    ) -> JSONResponse:
        if not original_images or len(original_images) != len(processed_images):
            raise ValueError("Original images are required for inline response mode")

        pages: List[Dict[str, str]] = []
        for name, original, translated in zip(page_names, original_images, processed_images):
            pages.append(
                {
                    "page_name": name,
                    "original": self._encode_image(original),
                    "translated": self._encode_image(translated),
                }
            )

        return JSONResponse(content={"mode": "inline", "pages": pages})


class ResponseFactory:
    @staticmethod
    def get_strategy(response_type: str) -> ResponseStrategy:
        print(f"Response type: {response_type}")
        strategies = {
            "zip": ZipResponseStrategy(),
            "json": JsonResponseStrategy(),
            "inline": InlinePairResponseStrategy(),
        }

        return strategies.get(response_type, ZipResponseStrategy())
