import asyncio
import base64
import io
import json
from pprint import pformat
import re
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from transformers import AutoTokenizer
import httpx
from module.translator.base import LLM
from settings import MAX_TOTAL_TOKENS, VLLM_API_KEY, VLLM_BASE_URL, VLLM_MODEL_NAME, VLLM_TOKENIZER
from utils.log import setup_logger

# Số lượng yêu cầu vllm tối đa có thể gửi đồng thời
semaphore = asyncio.Semaphore(4)
logger = setup_logger(__name__)


class TokenLimitExceeded(Exception):
    """Exception raised when token limit is exceeded."""
    pass


class Gemma3(LLM):
    _BASE_URL = VLLM_BASE_URL
    _TIMEOUT = 60.0
    _API_KEY = VLLM_API_KEY
    _MODEL = VLLM_MODEL_NAME
    _TOKENIZER = VLLM_TOKENIZER
    _MAX_TOTAL_TOKENS = MAX_TOTAL_TOKENS

    def __init__(self):
        self.model = self._MODEL
        self.token_count = 0
        self.token_count_last = 0
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._TOKENIZER, use_fast=True)

    def _normalize_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chuẩn hóa lịch sử để đảm bảo luân phiên user/assistant đúng cách.

        Quy tắc:
        1. Loại bỏ vai trò system khỏi chuỗi luân phiên
        2. Đảm bảo lịch sử bắt đầu bằng user
        3. Đảm bảo các vai trò luân phiên user/assistant/user/assistant...
        4. Loại bỏ hoàn toàn các vai trò liên tiếp giống nhau
        """
        if not history:
            return []

        # Lọc ra các message hợp lệ (chỉ user và assistant)
        filtered_messages = []
        for msg in history:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                continue

            if msg["role"] in ["user", "assistant"]:
                filtered_messages.append(msg)

        # Nếu không có messages hợp lệ, trả về rỗng
        if not filtered_messages:
            return []

        # Tạo danh sách mới với luân phiên nghiêm ngặt
        strict_alternating = []
        expected_role = "user"  # Luôn bắt đầu với user

        # Thêm user trống nếu cần
        if filtered_messages[0]["role"] != "user":
            strict_alternating.append({"role": "user", "content": ""})
            expected_role = "assistant"

        # Xử lý từng message
        for msg in filtered_messages:
            if msg["role"] == expected_role:
                strict_alternating.append(msg)
                expected_role = "assistant" if expected_role == "user" else "user"
            else:
                # Nếu vai trò không đúng kỳ vọng, bỏ qua để duy trì luân phiên
                logger.debug(
                    f"Skipping message with role {msg['role']} to maintain alternation")

        # Log để debug
        roles_before = [msg["role"] for msg in filtered_messages]
        roles_after = [msg["role"] for msg in strict_alternating]
        logger.debug(f"Original history roles: {roles_before}")
        logger.debug(f"Normalized history roles: {roles_after}")

        return strict_alternating

    def _build_messages_with_history(
        self,
        prompt_system: str,
        history: Optional[List[Dict[str, Any]]],
        user_content: Any,
        response_token_budget: int,
    ) -> List[Dict[str, Any]]:
        """Xây dựng danh sách messages gồm system + history + current user,
        sau đó cắt tỉa history để vừa ngân sách token.

        response_token_budget: số token dự kiến dành cho phần sinh trả lời.
        """
        # Luôn bắt đầu với system message
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": prompt_system}
        ]

        if history:
            # Đảm bảo lịch sử luân phiên đúng thứ tự user/assistant
            normalized_history = self._normalize_history(history)
            messages.extend(normalized_history)

        # Thêm user message hiện tại
        messages.append({"role": "user", "content": user_content})

        # Log để debug
        # logger.debug(f"Messages:\n{pformat(messages)}")

        return self._truncate_messages_to_fit_budget(messages, response_token_budget)

    def _truncate_messages_to_fit_budget(
        self,
        messages: List[Dict[str, Any]],
        response_token_budget: int,
    ) -> List[Dict[str, Any]]:
        """Cắt tỉa lịch sử cũ nếu tổng token vượt quá ngân sách dành cho prompt.

        Chiến lược:
        - Luôn giữ message đầu (system) và cuối (user hiện tại).
        - Xoá dần từ lịch sử cũ nhất trở đi cho tới khi vừa ngân sách.
        - Nếu vẫn vượt, rút gọn nội dung câu hỏi hiện tại theo ước lượng ký tự.
        """
        # Ngân sách dành cho messages (prompt)
        allowed_prompt_tokens = max(
            0, self._MAX_TOTAL_TOKENS - max(1, response_token_budget))

        def calc_tokens(msgs: List[Dict[str, Any]]) -> int:
            return self.estimate_message_tokens(msgs)

        total_tokens = calc_tokens(messages)
        if total_tokens <= allowed_prompt_tokens:
            return messages

        # Xác định phạm vi history: từ index 1 tới len-2
        history_start = 1
        history_end = max(history_start, len(messages) - 1)

        # Xoá dần từ đầu history
        pruned_messages = list(messages)
        while total_tokens > allowed_prompt_tokens and (history_end - history_start) > 0:
            del pruned_messages[history_start]
            total_tokens = calc_tokens(pruned_messages)

        if total_tokens <= allowed_prompt_tokens:
            return pruned_messages

        # Nếu vẫn vượt, rút gọn nội dung user hiện tại (cuối danh sách)
        last_idx = len(pruned_messages) - 1
        last_msg = pruned_messages[last_idx]
        content = last_msg.get("content")

        # Trường hợp content là list (multi-modal): chỉ có thể rút gọn phần text
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    # Ước lượng: 1 token ~ 4 ký tự
                    max_chars = max(0, (allowed_prompt_tokens // 4))
                    if len(text) > max_chars:
                        item["text"] = text[-max_chars:]
            total_tokens = calc_tokens(pruned_messages)
        else:
            text = str(content)
            max_chars = max(0, (allowed_prompt_tokens // 4))
            if len(text) > max_chars:
                pruned_messages[last_idx]["content"] = text[-max_chars:]
            total_tokens = calc_tokens(pruned_messages)

        return pruned_messages

    def count_tokens(self, text: str) -> int:
        """Đếm số token trong một đoạn văn bản."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback to rough estimation
            return len(text) // 4

    def estimate_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Ước tính số token trong danh sách messages."""
        total = 0
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')

            # Xử lý nội dung có thể là văn bản hoặc danh sách các phần tử đa phương tiện
            if isinstance(content, list):
                text_content = ""
                for item in content:
                    if item.get('type') == 'text':
                        text_content += item.get('text', '')
                    # Với hình ảnh, ta cộng thêm một lượng token cố định (ước lượng)
                    elif item.get('type') == 'image_url':
                        total += 500  # Ước lượng cho mỗi hình ảnh
                total += self.count_tokens(text_content)
            else:
                total += self.count_tokens(str(content))

            # Thêm token cho role và định dạng
            total += 3  # ~3 tokens for role formatting

        return total
      
    async def get_answer(
        self,
        question: str,
        prompt_system: str = "Bạn là trợ lý AI hữu ích.",
        image=None,
        history: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.01,
        top_p: float = 0.5,
        max_tokens: int = None,
    ) -> str:
        async with semaphore:
            try:
                if isinstance(question, list):
                    question = "\n".join(question)

                if image is not None:
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(image)
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(
                        buffered.getvalue()).decode('utf-8')
                    user_content = [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"}},
                    ]
                else:
                    user_content = question

                # Dự trù ngân sách token cho phần sinh trả lời
                provisional_response_budget = max_tokens if max_tokens is not None else min(
                    1024, self._MAX_TOTAL_TOKENS // 2)

                # Xây dựng messages với history và cắt tỉa nếu cần
                messages = self._build_messages_with_history(
                    prompt_system=prompt_system,
                    history=history,
                    user_content=user_content,
                    response_token_budget=provisional_response_budget,
                )

                messages_tokens = self.estimate_message_tokens(messages)
                logger.debug(f"messages_tokens: {messages_tokens}")
                self.check_token_limit(messages)

                async with httpx.AsyncClient(
                    base_url=self._BASE_URL,
                    timeout=self._TIMEOUT,
                    headers={"Authorization": f"Bearer {self._API_KEY}"},
                ) as client:
                    response = await client.post(
                        "/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_tokens": max_tokens if max_tokens is not None else max(1, self._MAX_TOTAL_TOKENS - messages_tokens),
                        },
                    )

                    response.raise_for_status()
                    result = response.json()
                content = result.get("choices", [{}])[0].get(
                    "message", {}).get("content", "")

                # await self.reset_cache()
                processed_content = self._process_output_answer(content)
                logger.debug(f"LLM Response:\n{(pformat(processed_content))}")
                return processed_content
            except TokenLimitExceeded as e:
                logger.error(f"Token limit exceeded: {e}")
                raise e
            except Exception as e:
                logger.error(f"Error: {e}")
                return ""

    def check_token_limit(self, messages: List[Dict[str, Any]]) -> Tuple[bool, int]:
        """Kiểm tra xem tổng số token có vượt quá giới hạn không."""
        estimated_tokens = self.estimate_message_tokens(messages)
        remaining_space = self._MAX_TOTAL_TOKENS - estimated_tokens
        if remaining_space < 0:
            raise TokenLimitExceeded(
                f"Token limit exceeded. Estimated tokens: {estimated_tokens}, "
                f"Max allowed: {self._MAX_TOTAL_TOKENS}"
            )

async def _demo_main():
    # pil_image = Image.open("page_0.png")
    # # Không ảnh
    # await llm.get_answer("Câu hỏi?", "Bạn là trợ lý...")
    # # Có ảnh
    # await llm.get_answer("Mô tả ảnh này", "Bạn là trợ lý...", image=pil_image)
    # Test batch
    import time
    start_time = time.time()
    await llm.get_batch_answer([
        "Hôm nay là ngày mấy?",
        "Bạn là ai?",
        "Xin chào!",
        "Hôm nay là ngày mấy?",
        "Bạn là ai?",
        "Xin chào!",
        "Hôm nay là ngày mấy?",
        "Bạn là ai?",
        "Xin chào!",
        "Hôm nay là ngày mấy?",
        "Bạn là ai?",
        "Xin chào!",
        "Hôm nay là ngày mấy?",
        "Bạn là ai?",
        "Xin chào!",
        "Hôm nay là ngày mấy?",
        "Bạn là ai?",
        "Xin chào!",
    ])
    end_time = time.time()
    logger.debug(f"Time taken: {end_time - start_time} seconds")
llm = Gemma3()
if __name__ == "__main__":
    asyncio.run(_demo_main())
