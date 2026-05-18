"""
Constants and configuration values for TextBoost.
"""

# Model configurations
DIFFUSERS_MODEL_DICT = {
    "sd14": "CompVis/stable-diffusion-v1-4",
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd21base": "models/stable-diffusion-2-1-base",
    "sd21": "models/stable-diffusion-2-1",
    "sdxlbase": "stabilityai/stable-diffusion-xl-base-1.0",
    "sana600m512": "Efficient-Large-Model/Sana_600M_512px_diffusers",
    "sana600m1024": "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    "sana1600m512": "Efficient-Large-Model/Sana_1600M_512px_diffusers",
    "sana1600m1024": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    "sana1.5_1.6b": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    "sana1.5_4.8b": "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
}

SD_MODEL_DICT = {
    "sd14": "CompVis/stable-diffusion-v1-4",
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd21base": "models/stable-diffusion-2-1-base",
    "sd21": "models/stable-diffusion-2-1",
}

FLUX_MODEL_DICT = {
    "dev": "black-forest-labs/FLUX.1-dev",
    "schnell": "black-forest-labs/FLUX.1-schnell",
}


# Token thresholds for different models
class TokenThresholds:
    # CLIP model thresholds
    CLIP_EOS_TOKEN = 49407
    CLIP_VOCAB_SIZE = 49408

    # SANA model thresholds
    SANA_SPECIAL_TOKEN_START = 256000


# StyleDrop instances configuration
STYLEDROP_INSTANCES = [
    ("00", "A seascape and cliffs in {} style", "watercolor painting"),
    ("01", "A house in {} style", "watercolor painting"),
    ("02", "A cat in {} style", "watercolor painting"),
    ("03", "Row of flowers in {} style", "watercolor painting"),
    ("04", "A village in {} style", "oil painting"),
    ("05", "A village in {} style", "line drawing"),
    ("07", "A portrait of a person wearing a hat in {} style", "oil painting"),
    ("08", "A woman walking a dog in {} style", "flat cartoon illustration"),
    ("09", "A woman working on a laptop in {} style", "flat cartoon illustration"),
    ("10", "A Christmas tree in {} style", "sticker"),
    ("11", "A wave in {} style", "abstract rainbow colored flowing smoke wave design"),
    ("12", "A mushroom in {} style", "glowing"),
    (
        "15",
        "Slices of watermelon and clouds in the background in {} style",
        "3D rendering",
    ),
    ("16", "A house in {} style", "3D rendering"),
    ("17", "A thumbs up in {} style", "glowing"),
    (
        "18",
        "A female figure with exaggerated proportions in {} style",
        "modern 3D rendering",
    ),
    ("19", "A bear in {} style animal", "kid crayon drawing"),
    ("21", "A flower in {} style", "melting golden 3D rendering"),
    ("22", "A viking face with beard in {} style", "wooden sculpture"),
]

# Default validation prompts
DEFAULT_VALIDATION_PROMPTS = [
    "A man in <*>",
    "A cat in <*>",
    "Flowers in <*>",
    "A dog in <*>",
]
