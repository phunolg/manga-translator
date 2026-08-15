import asyncio
import cv2
import json
import langcodes
import langdetect
import os
import regex as re
import time
from module.default_config import DEFAULT_CONFIG
from module.utils.generic import Quadrilateral
import torch
import logging
import sys
import traceback
import numpy as np
from PIL import Image
from typing import List, Optional, Any

from module.utils.textblock import TextBlock
from settings import BASE_DIR, VRAM_USAGE_FRACTION
from utils.log import setup_logger

from .config import Config, Colorizer, Detector, Translator, Renderer, Inpainter
from module.utils import (
    BASE_PATH,
    LANGUAGE_ORIENTATION_PRESETS,
    ModelWrapper,
    Context,
    load_image,
    dump_image,
    visualize_textblocks,
    is_valuable_text,
    sort_regions,
)
from .textline_merge import dispatch as dispatch_textline_merge
from .detection import detector_cache, dispatch as dispatch_detection, prepare as prepare_detection, unload as unload_detection
from .mask_refinement import dispatch as dispatch_mask_refinement
from .inpainting import dispatch as dispatch_inpainting, prepare as prepare_inpainting, unload as unload_inpainting
from .rendering import dispatch as dispatch_rendering, dispatch_eng_render, dispatch_eng_render_pillow
from .colorization import dispatch as dispatch_colorization, prepare as prepare_colorization, unload as unload_colorization
from .upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling, unload as unload_upscaling

logger = setup_logger(__name__)


def load_dictionary(file_path):
    dictionary = []
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                # Ignore empty lines and lines starting with '#' or '//'
                if not line.strip() or line.strip().startswith('#') or line.strip().startswith('//'):
                    continue
                # Remove comment parts
                line = line.split('#')[0].strip()
                line = line.split('//')[0].strip()
                parts = line.split()
                if len(parts) == 1:
                    # If there is only the left part, the right part defaults to an empty string, meaning delete the left part
                    pattern = re.compile(parts[0])
                    dictionary.append((pattern, '', line_number))
                elif len(parts) == 2:
                    # If both left and right parts are present, perform the replacement
                    pattern = re.compile(parts[0])
                    dictionary.append((pattern, parts[1], line_number))
                else:
                    logger.error(
                        f'Invalid dictionary entry at line {line_number}: {line.strip()}')
    return dictionary


def apply_dictionary(text, dictionary):
    for pattern, value, line_number in dictionary:
        original_text = text
        text = pattern.sub(value, text)
        if text != original_text:
            logger.info(
                f'Line {line_number}: Replaced "{original_text}" with "{text}" using pattern "{pattern.pattern}" and value "{value}"')
    return text


LANGDETECT_MAP = {
    'zh-cn': 'CHS',
    'zh-tw': 'CHT',
    'cs': 'CSY',
    'nl': 'NLD',
    'en': 'ENG',
    'fr': 'FRA',
    'de': 'DEU',
    'hu': 'HUN',
    'it': 'ITA',
    'ja': 'JPN',
    'ko': 'KOR',
    'pl': 'PLK',
    'pt': 'PTB',
    'ro': 'ROM',
    'ru': 'RUS',
    'es': 'ESP',
    'tr': 'TRK',
    'uk': 'UKR',
    'vi': 'VIN',
    'ar': 'ARA',
    'hr': 'HRV',
    'th': 'THA',
    'id': 'IND',
    'tl': 'FIL'
}


class Features:
    verbose: bool
    ignore_errors: bool
    _gpu_limited_memory: bool
    device: Optional[str]
    kernel_size: Optional[int]
    models_ttl: int
    _progress_hooks: list[Any]
    result_sub_folder: str

    def __init__(self):
        params = {}
        self.pre_dict = params.get('pre_dict', None)
        self.post_dict = params.get('post_dict', None)
        self.font_path = None
        self.use_mtpe = False
        self.kernel_size = DEFAULT_CONFIG.get('kernel_size', 3)
        self.device = None
        self._gpu_limited_memory = False
        self.ignore_errors = False
        self.verbose = False
        self.models_ttl = 0

        self._progress_hooks = []
        self._add_logger_hook()

        self.parse_init_params(params)
        self.result_sub_folder = ''

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

        self._model_usage_timestamps = {}
        self._detector_cleanup_task = None
        self.prep_manual = params.get('prep_manual', None)
        self.context_size = params.get('context_size', 0)
        self.all_page_translations = []

    def parse_init_params(self, params: dict):
        self.verbose = params.get('verbose', False)
        self.use_mtpe = params.get('use_mtpe', False)
        self.font_path = params.get('font_path', None)
        self.models_ttl = params.get('models_ttl', 0)

        self.ignore_errors = params.get('ignore_errors', False)
        # check mps for apple silicon or cuda for nvidia
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'
        self.device = device
        self._gpu_limited_memory = params.get('use_gpu_limited', False)
        if self._gpu_limited_memory and not self.using_gpu:
            self.device = device
        if self.using_gpu and (not torch.cuda.is_available() and not torch.backends.mps.is_available()):
            raise Exception(
                'CUDA or Metal compatible device could not be found in torch whilst --use-gpu args was set.\n')
        ModelWrapper._MODEL_DIR = params.get(
            'model_dir', os.path.join(BASE_DIR, 'models'))

        logger.info(f"Model directory: {ModelWrapper._MODEL_DIR}")
        # todo: fix why is kernel size loaded in the constructor
        # Set input files
        self.input_files = params.get('input', [])
        # Set save_text
        self.save_text = params.get('save_text', False)
        # Set load_text
        self.load_text = params.get('load_text', False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(VRAM_USAGE_FRACTION)

    @property
    def using_gpu(self):
        return self.device.startswith('cuda') or self.device == 'mps'

    async def inpaint(self, image: np.ndarray, config: Config, text_bboxes: [TextBlock], target_lang:  str = "vi",) -> Context:

        # TODO: Take list of images to speed up batch processing
        import time
        start_time = time.time()
        ctx = Context()

        ctx.input = Image.fromarray(image)
        ctx.result = None
        ctx.img_rgb = image

        # preload and download models (not strictly necessary, remove to lazy load)
        if (self.models_ttl == 0):
            logger.info('Loading models')
            await prepare_detection(config.detector.detector)
            await prepare_inpainting(config.inpainter.inpainter, self.device)

        result = await self._inpaint(config, ctx, text_bboxes)
        return result

    async def _inpaint(self, config: Config, ctx: Context, text_bboxes: [TextBlock]) -> Context:
        # Start the background cleanup job once if not already started.
        # if self._detector_cleanup_task is None:
        #     self._detector_cleanup_task = asyncio.create_task(
        #         self._detector_cleanup_job())
        # -- Colorization
        if config.colorizer.colorizer != Colorizer.none:
            await self._report_progress('colorizing')
            try:
                ctx.img_colorized = await self._run_colorizer(config, ctx)
            except Exception as e:
                logger.error(
                    f"Error during colorizing:\n{traceback.format_exc()}")
                if not self.ignore_errors:
                    raise
                ctx.img_colorized = ctx.input  # Fallback to input image if colorization fails

        else:
            ctx.img_colorized = ctx.input

        # -- Upscaling
        # The default text detector doesn't work very well on smaller images, might want to
        # consider adding automatic upscaling on certain kinds of small images.
        if config.upscale.upscale_ratio:
            await self._report_progress('upscaling')
            try:
                ctx.upscaled = await self._run_upscaling(config, ctx)
            except Exception as e:
                logger.error(
                    f"Error during upscaling:\n{traceback.format_exc()}")
                if not self.ignore_errors:
                    raise
                # Fallback to colorized (or input) image if upscaling fails
                ctx.upscaled = ctx.img_colorized
        else:
            ctx.upscaled = ctx.img_colorized

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)

        # -- Detection
        await self._report_progress('detection')
        try:
            ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
        except Exception as e:
            logger.error(f"Error during detection:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.textlines = []
            ctx.mask_raw = None
            ctx.mask = None

        if ctx.mask_raw is not None:
            cv2.imwrite(os.path.join(BASE_DIR, 'debug',
                        'mask_raw.png'), ctx.mask_raw)

        if not ctx.textlines:
            ctx.result = ctx.input
            return await self._revert_upscale(config, ctx)

        print("run textline merge")

        # -- Textline merge
        await self._report_progress('textline_merge')
        try:
            ctx.text_regions = await self._run_textline_merge(config, ctx)
        except Exception as e:
            logger.error(
                f"Error during textline_merge:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.text_regions = []  # Fallback to empty text_regions if textline merge fails

        if ctx.text_regions:
            bboxes = visualize_textblocks(cv2.cvtColor(
                ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions)
            cv2.imwrite(os.path.join(BASE_DIR, 'debug', 'bboxes.png'), bboxes)

        # Apply pre-dictionary after textline merge
        pre_dict = load_dictionary(self.pre_dict)
        pre_replacements = []
        for region in ctx.text_regions:
            original = region.text
            region.text = apply_dictionary(region.text, pre_dict)
            if original != region.text:
                pre_replacements.append(f"{original} => {region.text}")

        if pre_replacements:
            logger.info("Pre-translation replacements:")
            for replacement in pre_replacements:
                logger.info(replacement)
        else:
            logger.info("No pre-translation replacements made.")

        # -- Mask refinement
        # (Delayed to take advantage of the region filtering done after ocr and translation)
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
            except Exception as e:
                logger.error(
                    f"Error during mask-generation:\n{traceback.format_exc()}")
                if not self.ignore_errors:
                    raise
                ctx.mask = ctx.mask_raw if ctx.mask_raw is not None else np.zeros_like(
                    # Fallback to raw mask or empty mask
                    ctx.img_rgb, dtype=np.uint8)[:, :, 0]

        if ctx.mask is not None:
            inpaint_input_img = await dispatch_inpainting(Inpainter.lama_mpe, ctx.img_rgb, ctx.mask, config.inpainter, config.inpainter.inpainting_size,
                                                          self.device, self.verbose)
            cv2.imwrite(os.path.join(BASE_DIR, 'debug', 'inpaint_input.png'), cv2.cvtColor(
                inpaint_input_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(BASE_DIR, 'debug',
                        'mask_final.png'), ctx.mask)

        # -- Inpainting
        await self._report_progress('inpainting')
        try:
            ctx.img_inpainted = await self._run_inpainting(config, ctx)
        except Exception as e:
            logger.error(f"Error during inpainting:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            # Fallback to original RGB image if inpainting fails
            ctx.img_inpainted = ctx.img_rgb
        ctx.gimp_mask = np.dstack(
            (cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), ctx.mask))
        cv2.imwrite(os.path.join(BASE_DIR, 'debug', 'inpainted.png'),
                    cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR))

        # -- Rendering
        await self._report_progress('rendering')
        try:
            print(f"Rendering {len(text_bboxes)} text regions")
            ctx.text_bboxes = text_bboxes
            ctx.img_rendered = await self._run_text_rendering(config, ctx)
        except Exception as e:
            logger.error(f"Error during rendering:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            # Fallback to inpainted (or original RGB) image if rendering fails
            ctx.img_rendered = ctx.img_inpainted

        cv2.imwrite(os.path.join(BASE_DIR, 'debug', 'rendered.png'),
                    cv2.cvtColor(ctx.img_rendered, cv2.COLOR_RGB2BGR))
        await self._report_progress('finished', True)
        ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)

        return await self._revert_upscale(config, ctx)

    async def _report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            await ph(state, finished)

    # If `revert_upscaling` is True, revert to input size
    # Else leave `ctx` as-is
    async def _revert_upscale(self, config: Config, ctx: Context):
        if config.upscale.revert_upscaling:
            await self._report_progress('downscaling')
            ctx.result = ctx.result.resize(ctx.input.size)
        return ctx

    async def _run_colorizer(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[(
            "colorizer", config.colorizer.colorizer)] = current_time
        # todo: im pretty sure the ctx is never used. does it need to be passed in?
        output = await dispatch_colorization(
            config.colorizer.colorizer,
            colorization_size=config.colorizer.colorization_size,
            denoise_sigma=config.colorizer.denoise_sigma,
            device=self.device,
            image=ctx.input,
            **ctx
        )
        await unload_colorization(config.colorizer.colorizer)
        return output

    async def _run_upscaling(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[(
            "upscaling", config.upscale.upscaler)] = current_time
        output = (await dispatch_upscaling(config.upscale.upscaler, [ctx.img_colorized], config.upscale.upscale_ratio, self.device))[0]
        await unload_upscaling(config.upscale.upscaler)
        return output

    async def _run_detection(self, config: Config, ctx: Context):
        with torch.no_grad():
            current_time = time.time()
            self._model_usage_timestamps[(
                "detection", config.detector.detector)] = current_time
            output = await dispatch_detection(config.detector.detector, ctx.img_rgb, config.detector.detection_size, config.detector.text_threshold,
                                              config.detector.box_threshold,
                                              config.detector.unclip_ratio, config.detector.det_invert, config.detector.det_gamma_correct, config.detector.det_rotate,
                                              config.detector.det_auto_rotate,
                                              self.device, self.verbose)
            await unload_detection(config.detector.detector)
            return output

    async def _run_mask_refinement(self, config: Config, ctx: Context):
        return await dispatch_mask_refinement(ctx.text_regions, ctx.img_rgb, ctx.mask_raw, 'fit_text',
                                              config.mask_dilation_offset, self.kernel_size)

    async def _run_inpainting(self, config: Config, ctx: Context):
        with torch.no_grad():
            current_time = time.time()
            self._model_usage_timestamps[(
                "inpainting", config.inpainter.inpainter)] = current_time
            output = await dispatch_inpainting(config.inpainter.inpainter, ctx.img_rgb, ctx.mask, config.inpainter, config.inpainter.inpainting_size, self.device,
                                               self.verbose)
            await unload_inpainting(config.inpainter.inpainter)
            return output

    async def _run_text_rendering(self, config: Config, ctx: Context, target_lang: str = "vi"):
        current_time = time.time()

        self._model_usage_timestamps[(
            "rendering", config.render.renderer)] = current_time
        if config.render.renderer == Renderer.none:
            return ctx.img_inpainted
        print("config.render.renderer: ", config.render.renderer)
        print("LANGUAGE_ORIENTATION_PRESETS.get(ctx.text_regions[0].target_lang): ", LANGUAGE_ORIENTATION_PRESETS.get(
            ctx.text_regions[0].target_lang))
        # manga2eng currently only supports horizontal left to right rendering
        match config.render.renderer:
            case Renderer.manga2Eng:
                return dispatch_eng_render(
                    ctx.img_inpainted,
                    ctx.img_rgb,
                    ctx.text_bboxes,
                    ctx.text_regions,
                    self.font_path,
                    config.render.line_spacing,
                    config.render.disable_font_border,
                    lang=target_lang
                )
                
            case Renderer.manga2EngPillow:
                return dispatch_eng_render_pillow(
                    ctx.img_inpainted,
                    ctx.img_rgb,
                    ctx.text_regions,
                    self.font_path,
                    config.render.line_spacing,
                    config.render.disable_font_border,
                    lang=target_lang
                )
                
            case _:
                return  dispatch_rendering(
                    ctx.img_inpainted,
                    ctx.text_bboxes,
                    ctx.text_regions,
                    font_path=self.font_path,
                    font_size_fixed=config.render.font_size,
                    font_size_offset=config.render.font_size_offset, font_size_minimum=config.render.font_size_minimum,
                    hyphenate=not config.render.no_hyphenation,
                    render_mask=ctx.render_mask,
                    line_spacing=config.render.line_spacing
                )

    async def _run_textline_merge(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[(
            "textline_merge")] = current_time
        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0],
                                                    verbose=self.verbose)
        for region in text_regions:
            if not hasattr(region, "text_raw"):
                # <- Save the initial OCR results to expand the render detection box. Also, prevent affecting the forbidden translation function.
                region.text_raw = region.text
        # Filter out languages to skip
        if config.translator.skip_lang is not None:
            skip_langs = [lang.strip().upper()
            for lang in config.translator.skip_lang.split(',')]
            filtered_textlines = []
            for txtln in ctx.textlines:
                try:
                    detected_lang = langdetect.detect(txtln.text)
                    source_language = LANGDETECT_MAP.get(
                        detected_lang.lower(), 'UNKNOWN').upper()
                except Exception:
                    source_language = 'UNKNOWN'

                # Print detected source_language and whether it's in skip_langs
                # logger.info(f'Detected source language: {source_language}, in skip_langs: {source_language in skip_langs}, text: "{txtln.text}"')

                if source_language in skip_langs:
                    logger.info(f'Filtered out: {txtln.text}')
                    logger.info(
                        f'Reason: Detected language {source_language} is in skip_langs')
                    continue  # Skip this region
                filtered_textlines.append(txtln)
            ctx.textlines = filtered_textlines

        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0],
                                                     verbose=self.verbose)
        logger.info(f"Text regions after filtering: {len(text_regions)}")
        new_text_regions = []
        for region in text_regions:
            # Remove leading spaces after pre-translation dictionary replacement
            original_text = region.text
            stripped_text = original_text.strip()

            # Record removed leading characters
            removed_start_chars = original_text[:len(
                original_text) - len(stripped_text)]
            if removed_start_chars:
                logger.info(
                    f'Removed leading characters: "{removed_start_chars}" from "{original_text}"')

            # Modified filtering condition: handle incomplete parentheses
            bracket_pairs = {
                '(': ')', '（': '）', '[': ']', '【': '】', '{': '}', '〔': '〕', '〈': '〉', '「': '」',
                '"': '"', '＂': '＂', "'": "'", "“": "”", '《': '》', '『': '』', '"': '"', '〝': '〞', '﹁': '﹂', '﹃': '﹄',
                '⸂': '⸃', '⸄': '⸅', '⸉': '⸊', '⸌': '⸍', '⸜': '⸝', '⸠': '⸡', '‹': '›', '«': '»', '＜': '＞', '<': '>'
            }
            left_symbols = set(bracket_pairs.keys())
            right_symbols = set(bracket_pairs.values())

            has_brackets = any(s in stripped_text for s in left_symbols) or any(
                s in stripped_text for s in right_symbols)

            if has_brackets:
                result_chars = []
                stack = []
                to_skip = []

                # 第一次遍历：标记匹配的括号
                # First traversal: mark matching brackets
                for i, char in enumerate(stripped_text):
                    if char in left_symbols:
                        stack.append((i, char))
                    elif char in right_symbols:
                        if stack:
                            # 有对应的左括号，出栈
                            # There is a corresponding left bracket, pop the stack
                            stack.pop()
                        else:
                            # 没有对应的左括号，标记为删除
                            # No corresponding left parenthesis, marked for deletion
                            to_skip.append(i)

                # 标记未匹配的左括号为删除
                # Mark unmatched left brackets as delete
                for pos, _ in stack:
                    to_skip.append(pos)

                has_removed_symbols = len(to_skip) > 0

                # 第二次遍历：处理匹配但不对应的括号
                # Second pass: Process matching but mismatched brackets
                stack = []
                for i, char in enumerate(stripped_text):
                    if i in to_skip:
                        # 跳过孤立的括号
                        # Skip isolated parentheses
                        continue

                    if char in left_symbols:
                        stack.append(char)
                        result_chars.append(char)
                    elif char in right_symbols:
                        if stack:
                            left_bracket = stack.pop()
                            expected_right = bracket_pairs.get(left_bracket)

                            if char != expected_right:
                                # 替换不匹配的右括号为对应左括号的正确右括号
                                # Replace mismatched right brackets with the correct right brackets corresponding to the left brackets
                                result_chars.append(expected_right)
                                logger.info(
                                    f'Fixed mismatched bracket: replaced "{char}" with "{expected_right}"')
                            else:
                                result_chars.append(char)
                    else:
                        result_chars.append(char)

                new_stripped_text = ''.join(result_chars)

                if has_removed_symbols:
                    logger.info(
                        f'Removed unpaired bracket from "{stripped_text}"')

                if new_stripped_text != stripped_text and not has_removed_symbols:
                    logger.info(
                        f'Fixed brackets: "{stripped_text}" → "{new_stripped_text}"')

                stripped_text = new_stripped_text

            region.text = stripped_text.strip()

            if config.render.font_color_fg or config.render.font_color_bg:
                if config.render.font_color_bg:
                    region.adjust_bg_color = False
            new_text_regions.append(region)
        text_regions = new_text_regions

        text_regions = sort_regions(
            text_regions,
            right_to_left=config.render.rtl,
            img=ctx.img_rgb
        )

        return text_regions

    async def _unload_model(self, tool, model):
        match tool:
            case "detection":
                await unload_detection(model)
            case "inpainting":
                await unload_inpainting(model)
            case "upscaling":
                await unload_upscaling(model)
            case "colorization":
                await unload_colorization(model)
            case _:
                logger.warning(f"Unknown tool: {tool}")
        logger.info(f"{detector_cache=}")
        # Giải phóng bộ nhớ CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def _detector_cleanup_job(self):
        while True:
            if self.models_ttl == 0:
                await asyncio.sleep(1)
                continue
            now = time.time()
            logger.info(f"{self._model_usage_timestamps.items()=}")
            for (tool, model), last_used in list(self._model_usage_timestamps.items()):
                if now - last_used > self.models_ttl:
                    await self._unload_model(tool, model)
                    del self._model_usage_timestamps[(tool, model)]
            await asyncio.sleep(1)

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)

    def _add_logger_hook(self):
        # TODO: Pass ctx to logger hook
        LOG_MESSAGES = {
            'upscaling': 'Running upscaling',
            'detection': 'Running text detection',
            'ocr': 'Running ocr',
            'mask-generation': 'Running mask refinement',
            'translating': 'Running text translation',
            'rendering': 'Running rendering',
            'colorizing': 'Running colorization',
            'downscaling': 'Running downscaling',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-regions': 'No text regions! - Skipping',
            'skip-no-text': 'No text regions with text! - Skipping',
            'error-translating': 'Text translator returned empty queries',
            'cancelled': 'Image translation cancelled',
        }
        LOG_MESSAGES_ERROR = {
            # 'error-lang':           'Target language not supported by chosen translator',
        }

        async def ph(state, finished):
            if state in LOG_MESSAGES:
                logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                logger.warn(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)
