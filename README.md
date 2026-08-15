# MagiV2 - Manga Translation System

## Installation and Usage

To run the application:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

Once the server is running, you can access the API documentation at:
http://localhost:8000/docs

## Project Structure

```
magiv2/
├── module/                  # Core modules
│   ├── colorization/        # Manga colorization module
│   ├── crop_embedding/      # Image crop and embedding processing
│   ├── detection/           # Text detection in manga images
│   ├── features.py          # Feature extraction utilities
│   ├── history_retrival/    # Handles retrieval of translation history
│   ├── inpainting/          # Text inpainting on manga images
│   ├── knowledge/           # Knowledge base for context-aware translation
│   ├── mask_refinement/     # Refines text masks for better inpainting
│   ├── ocr/                 # Optical Character Recognition for manga text
│   ├── quality_assurance/   # QA checks for translations
│   ├── rendering/           # Renders translated text onto images
│   ├── textline_merge/      # Merges detected text lines
│   ├── translator/          # Translation module with LLM integration
│   ├── upscaling/           # Image upscaling for better quality
│   └── utils/               # Utility functions used across modules
├── transcript_history/      # Stores OCR results and translation history
│   └── [story_name]/        # Organized by story name
│       └── [chapter_number]/# Contains page transcripts for each chapter
├── main.py                  # FastAPI server entry point
└── settings.py              # Configuration settings
```

## API Workflow

The translation process involves two main steps:

1. **Get Transcript** (`/get-transcript`):
   - Upload manga pages
   - System performs OCR on the images
   - Extracted text is saved in `transcript_history/[story_name]/[chapter_number]/`
   - Returns the extracted text for review

2. **Translate and Inpaint** (`/translate-and-inpaint`):
   - Uses the previously extracted text from `transcript_history`
   - Translates the text using LLM with context awareness (2 steps bellow in this api have been comment to debug)
   - Inpaints the translated text back onto the original images
   - Returns the translated manga pages

## Example Usage

Use test_file.ipynb in workspace/test_comic folder

# License and Citation

The provided model and datasets are available for unrestricted use in personal, research, non-commercial, and not-for-profit endeavors. For any other usage scenarios, kindly contact me via email, providing a detailed description of your requirements, to establish a tailored licensing arrangement.
My contact information can be found on my website: ragavsachdeva [dot] github [dot] io

```
@misc{magiv2,
      title={Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names}, 
      author={Ragav Sachdeva and Gyungin Shin and Andrew Zisserman},
      year={2024},
      eprint={2408.00298},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00298}, 
}
```