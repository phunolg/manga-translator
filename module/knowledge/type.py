from enum import Enum

class Language(Enum):
    ENGLISH = "english"
    VIETNAMESE = "vietnamese"
    JAPANESE = "japanese"
    KOREAN = "korean"
    CHINESE = "chinese"
    FRENCH = "french"

class Topic(Enum):
    BASIC = "truyện hiện đại"
    HISTORY_FICTION = "truyện cổ trang"
    WUXIA = "wuxia"
    XIANXIA = "xianxia"
    COURT_INTRIGUE = "court_intrigue"
    FANTASY_GENERAL = "fantasy_general"
