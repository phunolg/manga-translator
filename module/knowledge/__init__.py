from __future__ import annotations
import json
import os
from pprint import pformat
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional

from module.knowledge.type import Topic
from settings import BASE_DIR
from utils.log import setup_logger

logger = setup_logger(__name__)

@dataclass
class Knowledge:
    topic: Topic
    name_mapping: Optional[Dict[str, str]] = None
    address_matrix: Optional[Dict[str, Dict[str, str]]] = None
    translate_dict: Optional[Dict[str, str]] = None
    characters: Optional[List[Dict[str, str]]] = None

    def __init__(self, 
        story_name: str, 
        topic: Topic = Topic.BASIC, 
        mapping_name: dict[str, str] | None = None, 
        translate_dict: dict[str, str] | None = None, 
        address_matrix: dict[str, dict[str, str]] | None = None,
        characters: Optional[List[Dict[str, str]]] = None,
    ):
        
        self.name_mapping = mapping_name or {}
        self.translate_dict = translate_dict or {}
        self.topic = topic
        self.address_matrix = address_matrix or {}
        self.story_dictionary = translate_dict or {}
        self.characters = characters or []
                                
        # Khởi tạo từ điển và ví dụ minh họa
        self._dictionary = {
            Topic.BASIC: {
                "boss": "ông chủ",
                "teacher": "thầy",
                "master": "sư phụ",  # có thể bị override theo context khác
            },
            Topic.HISTORY_FICTION: {
                "emperor": "hoàng đế",
                "empress": "hoàng hậu",
                "empress dowager": "thái hậu",
                "king": "quốc vương",
                "queen": "nữ vương",
                "crown prince": "thái tử",
                "prince": "hoàng tử",
                "princess": "công chúa",
                "lord": "lãnh chúa",
                "noble": "quý tộc",
                "duke": "công tước",
                "marquis": "hầu tước",
                "count": "bá tước",
                "earl": "bá tước",
                "viscount": "tử tước",
                "baron": "nam tước",
                "palace": "hoàng cung",
                "court": "triều đình",
                "imperial edict": "thánh chỉ",
                "minister": "đại thần",
                "general": "tướng quân",
                "army": "đại quân",
                "boss": "chủ nhân",
                "leader": "chủ nhân",
                "adults": "tiền bối",
                "pass romantic": "tình cũ",
                "cultivator": "tu sĩ",
                "selfish": "lòng lang dạ sói",
                "leader": "chủ nhân",
                "get intercepted": "bị phục kích",
                "spiritual vein": "linh mạch",
                "spiritual street": "mạch linh khí",
            },
            Topic.COURT_INTRIGUE: {
                "inner palace": "hậu cung",
                "concubine": "phi tần",
                "imperial concubine": "hoàng quý phi",
                "maid": "cung nữ",
                "eunuch": "thái giám",
                "faction": "phe phái",
                "royal decree": "thánh chỉ",
                "audience": "thượng triều",
                "memorial": "tấu chương",
                "treason": "mưu nghịch",
                "assassination": "thích sát",
                "scheme": "âm mưu",
                "poison": "kịch độc",
            },
            Topic.WUXIA: {
                "martial world": "giang hồ",
                "sect": "môn phái",
                "clan": "gia tộc",
                "patriarch": "tộc trưởng",
                "grandmaster": "đại tông sư",
                "elder": "trưởng lão",
                "disciple": "đệ tử",
                "inner disciple": "nội môn đệ tử",
                "outer disciple": "ngoại môn đệ tử",
                "manual": "bí kíp",
                "technique": "võ học",
                "sword": "kiếm",
                "saber": "đao",
                "blade": "lưỡi đao",
                "dagger": "đoản đao",
                "realm": "cảnh giới",
                "brother": "sư huynh",
                "sister": "sư tỷ",
                "junior brother": "sư đệ",
                "junior sister": "sư muội",
                "senior brother": "sư huynh",
                "senior sister": "sư tỷ",
            },
            Topic.XIANXIA: {
                "cultivation": "tu luyện",
                "cultivator": "tu sĩ",
                "spiritual energy": "linh khí",
                "qi": "chân khí",
                "meridian": "kinh mạch",
                "dantian": "đan điền",
                "breakthrough": "đột phá",
                "immortal": "tiên nhân",
                "ascend": "phi thăng",
                "tribulation": "thiên kiếp",
                "divine sense": "thần thức",
                "pill": "đan dược",
                "alchemy": "luyện đan",
                "array": "trận pháp",
                "spirit beast": "linh thú",
                "talent": "thiên phú",
                "root": "linh căn",
            },
            Topic.FANTASY_GENERAL: {
                "magic": "ma pháp",
                "mage": "pháp sư",
                "spell": "pháp thuật",
                "dragon": "long tộc",
                "ancient relic": "cổ vật",
                "prophecy": "lời sấm",
                "artifact": "thần khí",
                "abyss": "vực sâu",
            },
        }
        
        self._demonstrations = {
            Topic.BASIC: [
                ("The boss called everyone.", "Ông chủ gọi mọi người."),
                ("Master taught me this.", "Sư phụ đã dạy ta điều này."),
            ],
            Topic.HISTORY_FICTION: [
                ("The crown prince entered the palace hall.",
                 "Thái tử bước vào đại điện."),
                ("Her loyalty to the emperor never wavered.",
                 "Nàng chưa từng lay chuyển lòng trung thành với hoàng đế."),
                ("The general returned with the army at dusk.",
                 "Tướng quân dẫn đại quân hồi triều lúc hoàng hôn."),
            ],
            Topic.COURT_INTRIGUE: [
                ("The concubines whispered about the new decree.",
                 "Các phi tần thì thầm về thánh chỉ mới."),
                ("A silent scheme was unfolding in the inner palace.",
                 "Một âm mưu lặng lẽ đang dần mở ra trong hậu cung."),
                ("He submitted a memorial exposing corruption.",
                 "Hắn dâng tấu chương tố cáo tham ô."),
            ],
            Topic.WUXIA: [
                ("The sect elder guarded the ancient manual.",
                 "Trưởng lão của môn phái canh giữ bí kíp cổ."),
                ("He challenged his senior brother at dawn.",
                 "Hắn thách đấu sư huynh vào lúc bình minh."),
                ("The martial world was in turmoil.", "Giang hồ dậy sóng."),
            ],
            Topic.XIANXIA: [
                ("He refined the pill before attempting a breakthrough.",
                 "Hắn luyện đan trước khi thử đột phá."),
                ("The tribulation clouds gathered above the peak.",
                 "Mây thiên kiếp tụ lại trên đỉnh núi."),
                ("Her spiritual root was shattered.", "Linh căn của nàng bị phá hủy."),
            ],
            Topic.FANTASY_GENERAL: [
                ("The mage channeled ancient magic through the relic.",
                 "Pháp sư dẫn ma pháp cổ qua cổ vật."),
                ("A dragon circled the abyss.", "Một con long tộc lượn quanh vực sâu."),
            ],
        }

    # Ví dụ minh họa (định nghĩa class attribute, không dùng trong instance)
    # Đã chuyển sang khởi tạo trong __init__

    def get_topic(self) -> Topic:
        return self.topic
    
    def get_name_mapping(self, return_type: str = "dict") -> Dict[str, str] | str:
        match return_type:
            case "string":
                return "\n".join(f"'{src}' PHẢI ĐƯỢC VIẾT LẠI THÀNH '{tgt}'" for src, tgt in self.name_mapping.items())
            case "dict":
                return self.name_mapping
            case _:
                raise ValueError(f"Invalid return_type: {return_type}. Must be 'string' or 'dict'")
    
    def get_address_matrix(self, return_type: str = "dict"):
        """
        Trả về ma trận xưng hô theo định dạng yêu cầu
        
        Args:
            return_type: Loại dữ liệu trả về, có thể là "string" hoặc "dict"
            
        Returns:
            Chuỗi mô tả hoặc dictionary ma trận xưng hô
        """
        match return_type:
            case "string":
                output = ""
                for current_character, matrix in self.address_matrix.items():
                    t = f"- {current_character}:\n"
                    character_info = next(
                        (char for char in self.characters if char and char.get("name") == current_character),
                        {},
                    )
                    description = character_info.get("description", "Chưa có mô tả")
                    t += f"\t+ Đặc điểm: {description}\n"
                    for target_character, target_address in matrix.items():
                        if target_character == "other":
                            continue
                        target_self, target_others = self._split_address(target_address)
                        target_name = target_character
                        if target_character == "other":
                            continue
                        t += f"\t+ Xưng {target_self} và gọi {target_name} là {target_others}\n"
                    other_self, other_target = self._split_address(matrix.get("other", "ta-ngươi"))
                    t += f"\t+ Xưng {other_self} và gọi {other_target} với tất cả những người khác\n\n"
                    output += t
                return output
            case "dict":
                return self.address_matrix
            case _:
                raise ValueError(f"Invalid return_type: {return_type}. Must be 'string' or 'dict'")
    
    @staticmethod
    def _split_address(value: Optional[str]) -> Tuple[str, str]:
        if not value:
            return "ta", "ngươi"
        parts = value.split("-", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return value.strip(), ""
                
    def get_dictionary(self, return_type: str = "dict") -> Dict[str, str] | str:
        # Lấy từ điển từ topic, nếu không tồn tại thì dùng BASIC
        topic_dict = self._dictionary.get(self.topic, self._dictionary[Topic.BASIC]).copy()
        
        logger.info(pformat(f"using dictionary: {topic_dict}"))
        
        # Merge với story_dictionary (translate_dict)
        if self.story_dictionary:
            merge_dictionary = {**topic_dict, **self.story_dictionary}
        else:
            merge_dictionary = topic_dict
            
        # Trả về theo định dạng yêu cầu
        match return_type:
            case "string":
                return "\n".join(f"'{src}' PHẢI ĐƯỢC VIẾT LẠI BẰNG '{tgt}'" for src, tgt in merge_dictionary.items())
            case "dict":
                return merge_dictionary
            case _:
                raise ValueError(f"Invalid return_type: {return_type}. Must be 'string' or 'dict'")

    def get_demonstrations_of_topic(self) -> List[Tuple[str, str]]:
        return self._demonstrations.get(self.topic, [])

    def __str__(self) -> str:
        dictionary = self.get_dictionary(return_type='string')
        demonstrations = [
            f"'{input_seq}' được chuyển sang tiếng Việt là: '{output_seq}'"
            for input_seq, output_seq in self.get_demonstrations_of_topic()
        ]
        demonstrations = "\n".join(
            demonstrations) if demonstrations else "(Không có ví dụ)"
        return (
            f"**Topic:**\n{self.get_topic().value}\n\n"
            f"**Dictionary:**\n{dictionary}\n\n"
            f"**Ví dụ minh họa:**\n{demonstrations}\n\n"
            f"**Tên riêng phải sử dụng khi viết lại theo (ưu tiên cao nhất):**\n{self.get_name_mapping(return_type='string')}\n\n"
            f"**Ma trận xưng hô (bắt buộc):**\n{self.get_address_matrix(return_type='string')}\n\n"
        )


if __name__ == "__main__":
    knowledge = Knowledge(Topic.HISTORY_FICTION)
    print(knowledge)
