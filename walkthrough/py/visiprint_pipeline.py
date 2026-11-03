# VisiPrint pipeline
# Adapted from ZeST (https://github.com/ttchengab/zest_code)
# Cheng et al., "ZeST: Zero-shot Material Transfer from a Single Image", ECCV 2024

import os
import sys
import argparse
import cv2
import nltk
import torch
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision.transforms import Compose

from DPT.dpt.models import DPTDepthModel
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "IP-Adapter"))

from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from ip_adapter import IPAdapterXL
from ip_adapter.utils import register_cross_attention_hook
from ip_adapter.attention_processor import CNAttnProcessor
from rembg import remove

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--color_influence", type=float, default=0.8)
parser.add_argument("--luminance_influence", type=float, default=0.6)
parser.add_argument("--gcode_influence", type=float, default=0.8)
parser.add_argument("--inference_steps", type=int, default=30)
parser.add_argument("--use_finetuned", type=str, default="False")
parser.add_argument("--use_cropped_exemplar", type=str, default="False")
parser.add_argument("--text_prompt", type=str, default="")
parser.add_argument("--text_prompt_weight", type=float, default=0.5)
args = parser.parse_args()

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Paths
output_dir = "output_ui"
target_dir = "target_ui"
texture_dir = "src_ui"
os.makedirs(output_dir, exist_ok=True)

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"

controlnet_depth_path = "diffusers/controlnet-depth-sdxl-1.0"
controlnet_edges_path = "diffusers/controlnet-canny-sdxl-1.0"

device = "cuda"
torch.cuda.empty_cache()

# Discover input images
target_paths = [os.path.join(root, f)
                for root, _, files in os.walk(target_dir)
                for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

texture_paths = [os.path.join(root, f)
                 for root, _, files in os.walk(texture_dir)
                 for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

# BLIP captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=20)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def generate_main_noun(image):
    caption = generate_caption(image)
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)
    for word, tag in pos_tags:
        if tag in ["NN", "NNS", "NNP", "NNPS"]:
            return word.lower()
    return "object"

# DPT depth estimation
model_path = "DPT/weights/dpt_hybrid-midas-501f0c75.pt"
net_w = net_h = 384
dpt_model = DPTDepthModel(
    path=model_path,
    backbone="vitb_rn50_384",
    non_negative=True,
    enable_attention_hooks=False,
)
normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

transform = Compose([
    Resize(net_w, net_h,
           resize_target=None,
           keep_aspect_ratio=True,
           ensure_multiple_of=32,
           resize_method="minimal",
           image_interpolation_method=cv2.INTER_CUBIC),
    normalization,
    PrepareForNet(),
])

dpt_model.eval()

# ControlNets + SDXL pipeline
print("Loading ControlNet models...")
cn_depth = ControlNetModel.from_pretrained(
    controlnet_depth_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16
).to(device)

cn_edges = ControlNetModel.from_pretrained(
    controlnet_edges_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16
).to(device)

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    base_model_path,
    controlnet=[cn_depth, cn_edges],
    torch_dtype=torch.float16
)

pipe.unet = register_cross_attention_hook(pipe.unet)

def _set_controlnet_attn_processors(controlnet):
    """Set attention processors on individual controlnets, handling MultiControlNetModel."""
    if hasattr(controlnet, "nets"):
        for cn in controlnet.nets:
            cn.set_attn_processor(CNAttnProcessor())
    else:
        controlnet.set_attn_processor(CNAttnProcessor())

_set_controlnet_attn_processors(pipe.controlnet)
pipe = pipe.to(device)

# Patch IPAdapter.set_ip_adapter to work with MultiControlNetModel.
# The upstream IP-Adapter code uses a stale import path for MultiControlNetModel,
# causing the isinstance check to fail with newer diffusers versions.
from ip_adapter.ip_adapter import IPAdapter
from ip_adapter.attention_processor import AttnProcessor, IPAttnProcessor

def _patched_set_ip_adapter(self):
    unet = self.pipe.unet
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                scale=1.0, num_tokens=self.num_tokens,
            ).to(self.device, dtype=torch.float16)
    unet.set_attn_processor(attn_procs)
    if hasattr(self.pipe, "controlnet"):
        _set_controlnet_attn_processors(self.pipe.controlnet)

IPAdapter.set_ip_adapter = _patched_set_ip_adapter

ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

# Optional LoRA fine-tuned weights
lora_weights_path = os.path.join(os.path.dirname(__file__), "finetuned_lora", "pytorch_lora_weights.safetensors")
if args.use_finetuned.lower() == "true" and os.path.exists(lora_weights_path):
    print(f"Loading LoRA weights from {lora_weights_path}...")
    pipe.load_lora_weights(lora_weights_path)
    pipe.fuse_lora(lora_scale=0.8)
    print("LoRA weights loaded and fused.")
elif args.use_finetuned.lower() == "true":
    print(f"WARNING: LoRA weights not found at {lora_weights_path}, running without fine-tuning.")


def match_color_profile(source_img, mask_img, target_img, blend_factor=0.8, contrast_weight=0.6):
    """Statistical color transfer within the object mask (Reinhard et al. style)."""
    mw, mh = mask_img.size
    if source_img.size != (mw, mh):
        source_img = source_img.resize((mw, mh), resample=Image.BICUBIC)
    if target_img.size != (mw, mh):
        target_img = target_img.resize((mw, mh), resample=Image.BICUBIC)

    source_np = np.array(source_img).astype(np.float32)
    target_np = np.array(target_img).astype(np.float32)
    mask_bin = (np.array(mask_img.convert("L")) > 128).astype(np.uint8)

    if mask_bin.sum() == 0:
        return target_img

    matched = target_np.copy()
    epsilon = 1e-6

    for c in range(3):
        src_vals = source_np[..., c][mask_bin == 1]
        tgt_vals = target_np[..., c][mask_bin == 1]
        if src_vals.size == 0 or tgt_vals.size == 0:
            continue
        mean_s, std_s = float(src_vals.mean()), float(src_vals.std())
        mean_t, std_t = float(tgt_vals.mean()), float(tgt_vals.std())
        std_scale = (std_s / (std_t + epsilon))
        adjusted_std = 1 + (std_scale - 1) * contrast_weight
        adjusted = (tgt_vals - mean_t) * adjusted_std + mean_s
        matched[..., c][mask_bin == 1] = adjusted

    blended = (1 - blend_factor) * target_np + blend_factor * matched
    blended = np.clip(blended, 0, 255)
    return Image.fromarray(np.uint8(blended))


def autocrop_material_exemplar(pil_img, erode_px=8, min_area=5000):
    """Crop exemplar to its foreground, removing background artifacts."""
    rgba = remove(pil_img)
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    alpha = np.array(rgba.split()[-1])
    mask = (alpha > 128).astype(np.uint8) * 255
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
        mask = cv2.erode(mask, k, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return pil_img
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + np.argmax(areas)
    x, y, w, h, area = (*stats[max_idx, :4], stats[max_idx, cv2.CC_STAT_AREA])
    if area < min_area or w < 5 or h < 5:
        return pil_img
    return pil_img.convert("RGB").crop((x, y, x + w, y + h))


def get_adaptive_canny_edges(target_rgb_pil, mask_L_pil,
                             density_bounds=(0.12, 0.25), p_range=(60.0, 99.0),
                             low_ratio=0.25, max_iters=8, blur_sigma=0.8,
                             mask_erode=1, min_blob_area=3):
    """Adaptive Canny edge detection inside the object mask.

    Automatically tunes thresholds so that edge density falls within
    density_bounds, avoiding both over- and under-detection on dark or
    low-contrast slicer screenshots.
    """
    gray = cv2.cvtColor(np.array(target_rgb_pil), cv2.COLOR_RGB2GRAY)
    mask = (np.array(mask_L_pil) > 128).astype(np.uint8)

    if mask.sum() == 0:
        return Image.fromarray(np.zeros_like(gray)).convert('RGB')

    # Fill outside with median of inside to avoid silhouette edges
    med = np.median(gray[mask == 1])
    gray_out = gray.copy()
    gray_out[mask == 0] = med

    k = int(max(3, 2 * int(blur_sigma * 3) + 1))
    gray_blur = cv2.GaussianBlur(gray_out, (k, k), blur_sigma)

    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    mags = np.sqrt(gx * gx + gy * gy)[mask == 1]

    p_lo, p_hi = p_range
    lower, upper = density_bounds
    mask_er = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=mask_erode)
    mask255 = (mask_er * 255).astype(np.uint8)
    area = max(1, int(mask_er.sum()))
    edges = np.zeros_like(gray)

    for _ in range(max_iters):
        p = 0.5 * (p_lo + p_hi)
        t_high = float(np.percentile(mags, p))
        t_high = max(1.0, min(255.0, t_high))
        t_low = low_ratio * t_high

        e = cv2.Canny(gray_blur, t_low, t_high, L2gradient=True)
        e = cv2.bitwise_and(e, mask255)

        num, lab, stats, _ = cv2.connectedComponentsWithStats((e > 0).astype(np.uint8), 8)
        clean = np.zeros_like(e)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
                clean[lab == i] = 255

        dens = (clean > 0).sum() / area
        edges = clean
        if dens < lower:
            p_hi = p
        elif dens > upper:
            p_lo = p
        else:
            break

    return Image.fromarray(edges).convert('RGB')


def generate_preview(input_image, material_exemplar):
    """Run the full VisiPrint pipeline on a single target/material pair."""

    # Build text prompt
    if args.text_prompt:
        prompt = args.text_prompt
    else:
        noun = generate_main_noun(input_image)
        print(f"Identified noun: {noun}")
        prompt = f"A 3D printed {noun}"
    print(f"Prompt: {prompt}")

    # Depth map
    img = np.array(input_image)
    img_input = transform({"image": img})["image"]

    with torch.no_grad():
        sample = torch.from_numpy(img_input).unsqueeze(0).to(torch.float32)
        prediction = dpt_model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    depth_min = prediction.min()
    depth_max = prediction.max()
    bits = 2
    max_val = (2 ** (8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (prediction - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(prediction.shape, dtype=prediction.dtype)

    out = (out / 256).astype('uint8')
    depth_map = Image.fromarray(out).resize((1024, 1024))

    # Foreground mask and grayscale init image
    rm_bg = remove(input_image)
    mask = rm_bg.split()[-1]
    mask = mask.point(lambda x: 255 if x > 0 else 0)

    mask_np = np.array(mask).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_eroded_np = cv2.erode(mask_np, kernel, iterations=1)
    mask_eroded = Image.fromarray(mask_eroded_np).convert('RGB')

    target_mask = mask_eroded
    invert_target_mask = ImageChops.invert(target_mask)

    gray_target_image = input_image.convert('L').convert('RGB')
    gray_target_image = ImageEnhance.Brightness(gray_target_image).enhance(1.0)
    grayscale_img = ImageChops.darker(gray_target_image, target_mask)
    img_black_mask = ImageChops.darker(input_image, invert_target_mask)
    init_img = ImageChops.lighter(img_black_mask, grayscale_img)

    # Prepare material exemplar
    use_crop = args.use_cropped_exemplar.lower() == "true"
    if use_crop:
        src_ex = autocrop_material_exemplar(material_exemplar)
    else:
        src_ex = material_exemplar.convert("RGB")

    # Resize everything to 1024
    ip_image_1024  = src_ex.resize((1024, 1024), resample=Image.BICUBIC)
    init_img_1024  = Image.fromarray(np.array(init_img)).resize((1024, 1024), resample=Image.BICUBIC)
    mask_1024      = target_mask.resize((1024, 1024), resample=Image.BICUBIC).convert("L")
    target_resized = input_image.resize((1024, 1024), resample=Image.BICUBIC)

    # Adaptive Canny edges inside mask
    canny_edges = get_adaptive_canny_edges(target_resized, mask_1024)

    canny_weight = args.gcode_influence
    # Higher text_prompt_weight means more text influence, so reduce IP-Adapter scale.
    ip_scale = (1.0 - args.text_prompt_weight) if args.text_prompt else 0.8

    # Generate
    images = ip_model.generate(
        pil_image=ip_image_1024,
        image=init_img_1024,
        control_image=[depth_map, canny_edges],
        mask_image=mask_1024.convert('RGB'),
        controlnet_conditioning_scale=[0.8, canny_weight],
        num_samples=1,
        num_inference_steps=args.inference_steps,
        seed=42,
        prompt=prompt,
        scale=ip_scale
    )

    # Color harmonization
    final_1024 = match_color_profile(ip_image_1024, mask_1024, images[0],
                                     blend_factor=args.color_influence,
                                     contrast_weight=args.luminance_influence)

    return final_1024, depth_map, canny_edges


# Process all target/material pairs
for target_path in target_paths:
    for texture_path in texture_paths:
        target_img_name = os.path.splitext(os.path.basename(target_path))[0]
        texture_img_name = os.path.splitext(os.path.basename(texture_path))[0]

        output_path = os.path.join(output_dir, f"{target_img_name}_{texture_img_name}.png")

        if os.path.exists(output_path):
            print(f"Output {output_path} already exists. Skipping...")
            continue

        print(f"Loading target image from {target_path}...")
        target_image = Image.open(target_path).convert('RGB')
        original_size = target_image.size

        print(f"Loading material exemplar image from {texture_path}")
        tex_img = Image.open(texture_path)
        if tex_img.mode == 'RGBA':
            tex_img = tex_img.convert('RGB')

        result_1024, depth_map, canny_edges = generate_preview(target_image, tex_img)
        result_1024.resize(original_size, Image.BICUBIC).save(output_path)
        print(f"SAVED {output_path}")
