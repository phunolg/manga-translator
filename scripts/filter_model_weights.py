import os
import torch
from safetensors.torch import load_file, save_file
from settings import BASE_DIR

from configuration_magiv2 import Magiv2Config

ckpt_dir = os.path.join(BASE_DIR, "models")

def save_weights(output_name: str, key: str):
    out_path = os.path.join(BASE_DIR, "models", output_name)

    bin_path = os.path.join(ckpt_dir, "magiv2.pytorch_model.bin")

    if os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError("Không tìm thấy pytorch_model.bin")
    keys = set()
    for k in state.keys():
        keys.add(k.split(".")[0])
    print(f"{keys = }")
    

    # Lọc key bắt đầu bằng 'ocr_model.' và bỏ prefix để khớp trực tiếp với VisionEncoderDecoderModel
    filtered = {k.replace(f"{key}.", "", 1): v for k, v in state.items() if k.startswith(f"{key}.")}
   

    if not filtered:
        raise ValueError(f"Không tìm thấy bất kỳ trọng số nào với prefix '{key}.'")

    save_file(filtered, out_path)
    print(f"Saved {len(filtered)} tensors to {out_path}")
    
if __name__ == "__main__":
    from safetensors.torch import load_file
    from transformers import VisionEncoderDecoderModel

    save_weights("ocr.safetensors", "ocr_model")
    save_weights("crop_embedding.safetensors", "crop_embedding_model")
    save_weights("class_labels_classifier.safetensors", "class_labels_classifier")
    save_weights("bbox_predictor.safetensors", "bbox_predictor")
    save_weights("character_character_matching_head.safetensors", "character_character_matching_head")
    save_weights("text_character_matching_head.safetensors", "text_character_matching_head")
    save_weights("text_tail_matching_head.safetensors", "text_tail_matching_head")
    save_weights("detection_transformer.safetensors", "detection_transformer")
    save_weights("is_this_text_a_dialogue.safetensors", "is_this_text_a_dialogue")
    
    config: Magiv2Config = Magiv2Config.from_pretrained(BASE_DIR, trust_remote_code=True)
    # model_config phải đúng kiến trúc OCR đã dùng để train
    ocr = VisionEncoderDecoderModel(config.ocr_model_config)
    sd = load_file(os.path.join(BASE_DIR, "models", "ocr.safetensors"))
    missing, unexpected = ocr.load_state_dict(sd, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)