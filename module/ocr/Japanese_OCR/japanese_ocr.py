"""
Japanese OCR module - stub implementation.
Dùng LLM_OCR thay thế cho Japanese text recognition.
"""
from module.ocr.base import OCR
from module.ocr.llm_ocr import LLM_OCR
from utils.log import setup_logger

logger = setup_logger(__name__)


class JapaneseOCR(OCR):
    """
    Japanese OCR - currently delegates to LLM_OCR for text recognition.
    Can be replaced with a dedicated Japanese OCR model (e.g., manga-ocr).
    """

    def __init__(self, processor):
        super().__init__(processor)
        self._llm_ocr = LLM_OCR(processor)
        logger.info("JapaneseOCR initialized (using LLM_OCR backend)")

    async def predict(self, images, crop_bboxes, move_to_device_fn=None,
                      batch_size=32, max_new_tokens=64, **kwargs):
        """Delegate to LLM_OCR for Japanese text recognition."""
        logger.info("JapaneseOCR.predict: delegating to LLM_OCR")
        return await self._llm_ocr.predict(
            images, crop_bboxes,
            move_to_device_fn=move_to_device_fn,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )

    def unload(self):
        """Release resources."""
        if self._llm_ocr:
            self._llm_ocr.unload()
