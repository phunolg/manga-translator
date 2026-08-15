DEFAULT_CONFIG = {
    "render": {
        "alignment": "auto",
        "direction": "auto",
        "disable_font_border": False,
        "font_size_minimum": -1,
        "font_size_offset": 0,
        "gimp_font": "Sans-serif",
        "lowercase": False,
        "no_hyphenation": False,
        "renderer": "manga2eng",
        "rtl": True,
        "uppercase": False
    },
    "upscale": {
        "revert_upscaling": False,
        "upscaler": "esrgan"
    },
    "translator": {
        "no_text_lang_skip": False,
        "target_lang": "VIN",
        "translator": "gemma3"
    },
    "detector": {
        "box_threshold": 0.5,
        "det_auto_rotate": False,
        "det_gamma_correct": False,
        "det_invert": False,
        "det_rotate": False,
        "detection_size": 2048,
        "detector": "default",
        "text_threshold": 0.7,
        "unclip_ratio": 2.3
    },
    "colorizer": {
        "colorization_size": 576,
        "colorizer": "none",
        "denoise_sigma": 30
    },
    "inpainter": {
        "inpainter": "lama_mpe",
        "inpainting_precision": "bf16",
        "inpainting_size": 2048
    },
    "ocr": {
        "ignore_bubble": 0,
        "min_text_length": 0,
        "ocr1": "48px",
        "ocr2": "48px_ctc",
        "use_mocr_merge": False
    },
    "kernel_size": 3,
    "mask_dilation_offset": 0
}