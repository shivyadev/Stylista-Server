from django.apps import AppConfig
from django.conf import settings
import torch
import torchvision.models as models
import numpy as np
import os
from torch.serialization import add_safe_globals
from dotenv import load_dotenv
from .ml_models import EnhancedFocalMLPClassifier
from .utlis import download_file_from_url, load_npy_from_url

class AiModelsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai_models'

    def ready(self):
        """Loads all AI models and category mappings at Django startup."""
        if not hasattr(settings, "MODEL_REGISTRY"):
            settings.MODEL_REGISTRY = {}

        def get_input_size_from_state_dict(state_dict):
            # Get the shape of the first layer's weight matrix
            if 'network.0.weight' in state_dict:
                return state_dict['network.0.weight'].shape[1]
            return 3328

        load_dotenv()

         # Paths to state dict files (new paths)
        CASUAL_MAPPING_PATH = os.getenv("AWS_CASUAL_MAPPING")
        FORMAL_MAPPING_PATH = os.getenv("AWS_FORMAL_MAPPING")
        SPORTS_MAPPING_PATH = os.getenv("AWS_SPORTS_MAPPING")

        # Load category mappings first
        settings.MODEL_REGISTRY["casual_mapping"] = load_npy_from_url(CASUAL_MAPPING_PATH)
        settings.MODEL_REGISTRY["formal_mapping"] = load_npy_from_url(FORMAL_MAPPING_PATH)
        settings.MODEL_REGISTRY["sports_mapping"] = load_npy_from_url(SPORTS_MAPPING_PATH)

        # Get number of classes from mappings
        casual_num_classes = len(settings.MODEL_REGISTRY["casual_mapping"])
        formal_num_classes = len(settings.MODEL_REGISTRY["formal_mapping"])
        sports_num_classes = len(settings.MODEL_REGISTRY["sports_mapping"])

        print(f"Loading casual model with {casual_num_classes} classes...")
        # Create model instances with correct number of classes
        casual_state_dict = download_file_from_url(os.getenv("AWS_CASUAL_MODEL"))  # Loads .pt
        casual_input_size = get_input_size_from_state_dict(casual_state_dict)
        print(casual_input_size)

        settings.MODEL_REGISTRY["casual_model"] = EnhancedFocalMLPClassifier(
            input_size=casual_input_size,
            num_classes=casual_num_classes
        )
        settings.MODEL_REGISTRY["casual_model"].load_state_dict(casual_state_dict)
        settings.MODEL_REGISTRY["casual_model"].eval()

        print(f"Loading formal model with {formal_num_classes} classes...")
        
        formal_state_dict = download_file_from_url(os.getenv("AWS_FORMAL_MODEL"))
        formal_input_size = get_input_size_from_state_dict(formal_state_dict)
        print(formal_input_size)

        settings.MODEL_REGISTRY["formal_model"] = EnhancedFocalMLPClassifier(
            input_size=formal_input_size,
            num_classes=formal_num_classes
        )
        settings.MODEL_REGISTRY["formal_model"].load_state_dict(formal_state_dict)
        settings.MODEL_REGISTRY["formal_model"].eval()

        print(f"Loading sports model with {sports_num_classes} classes...")
        sports_state_dict = download_file_from_url(os.getenv("AWS_SPORTS_MODEL"))
        sports_input_size = get_input_size_from_state_dict(sports_state_dict)
        print(sports_input_size)

        settings.MODEL_REGISTRY["sports_model"] = EnhancedFocalMLPClassifier(
            input_size=sports_input_size,
            num_classes=sports_num_classes
        )
        settings.MODEL_REGISTRY["sports_model"].load_state_dict(sports_state_dict)
        settings.MODEL_REGISTRY["sports_model"].eval()

