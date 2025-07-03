from typing import Any, Dict, List, Tuple

import clip
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torchtyping import TensorType
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from omegaconf.listconfig import ListConfig

from src.training.diffuser_lenscript import Diffuser
from src.datasets.multimodal_dataset import MultimodalDataset

# ------------------------------------------------------------------------------------- #

batch_size, context_length = None, None
collate_fn = DataLoader([]).collate_fn

# ------------------------------------------------------------------------------------- #


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def load_clip_model(version: str, device: str) -> clip.model.CLIP:
    model, _ = clip.load(version, device=device, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def encode_text(
    caption_raws: List[str],  # batch_size
    clip_model: clip.model.CLIP,
    max_token_length: int,
    device: str,
) -> TensorType["batch_size", "context_length"]:
    if max_token_length is not None:
        default_context_length = 77
        context_length = max_token_length + 2  # start_token + 20 + end_token
        assert context_length < default_context_length
        # [bs, context_length] # if n_tokens > context_length -> will truncate
        texts = clip.tokenize(
            caption_raws, context_length=context_length, truncate=True
        )
        zero_pad = torch.zeros(
            [texts.shape[0], default_context_length - context_length],
            dtype=texts.dtype,
            device=texts.device,
        )
        texts = torch.cat([texts, zero_pad], dim=1)
    else:
        # [bs, context_length] # if n_tokens > 77 -> will truncate
        texts = clip.tokenize(caption_raws, truncate=True)

    # [batch_size, n_ctx, d_model]
    x = clip_model.token_embedding(texts.to(device)).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest in each sequence)
    x_tokens = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)].float()
    x_seq = [x[k, : (m + 1)].float() for k, m in enumerate(texts.argmax(dim=-1))]

    return x_seq, x_tokens


def get_batch(
    prompt: str,
    sample_id: str,
    clip_model: clip.model.CLIP,
    dataset: MultimodalDataset,
    seq_feat: bool,
    device: torch.device,
    custom_first_frame=None
) -> Dict[str, Any]:
    # Get base batch
    sample_index = dataset.root_filenames.index(sample_id)
    raw_batch = dataset[sample_index]
    print(f"RAW BATCH KAYS: {raw_batch.keys()}")
    batch = collate_fn([to_device(raw_batch, device)])
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"Key: {key}, Shape: {value.shape}\n")
        else:
            print(f"Key: {key}, Type: {type(value)}\n")

    # Encode text
    caption_seq, caption_tokens = encode_text([prompt], clip_model, None, device)

    if seq_feat:
        caption_feat = caption_seq[0]
        caption_feat = F.pad(caption_feat, (0, 0, 0, 77 - caption_feat.shape[0]))
        caption_feat = caption_feat.unsqueeze(0).permute(0, 2, 1)
    else:
        caption_feat = caption_tokens

    # Update batch
    batch["caption_raw"] = [prompt]
    batch["caption_feat"] = caption_feat

    if custom_first_frame is not None:
        traj_dataset = dataset.trajectory_dataset

        if isinstance(custom_first_frame, list) and len(custom_first_frame) == 13:
            matrix = torch.eye(4, device=device)
            matrix[0, 0:3] = torch.tensor(custom_first_frame[0:3], device=device)
            matrix[1, 0:3] = torch.tensor(custom_first_frame[4:7], device=device)
            matrix[2, 0:3] = torch.tensor(custom_first_frame[8:11], device=device)
            matrix[0, 3] = custom_first_frame[3]
            matrix[1, 3] = custom_first_frame[7]
            matrix[2, 3] = custom_first_frame[11]
            
            fov = torch.tensor([custom_first_frame[12]], device=device)
            
            trans = matrix[:3, 3].clone().unsqueeze(0)  # [1, 3]
            if traj_dataset.standardize:
                trans -= traj_dataset.shift_mean.to(device)
                trans /= traj_dataset.shift_std.to(device)
                
            rot = matrix[:3, :3]
            rot6d = rot[:, :2].permute(1, 0).reshape(1, 6)  # [1, 6]
            
            fov_normalized = fov.clone()
            if traj_dataset.standardize:
                fov_normalized -= traj_dataset.shift_mean_fov.to(device)
                fov_normalized /= traj_dataset.shift_std_fov.to(device)
                
            first_frame_feat = torch.cat([rot6d, trans, fov_normalized.reshape(1, 1)], dim=1)   # [1, 10]
            
            batch["custom_first_frame"] = first_frame_feat
        else:
            print("Warning: Custom first frame format invalid. Expecting 13 values.")

    return batch


def init(
    config_name: str,
) -> Tuple[Diffuser, clip.model.CLIP, MultimodalDataset, torch.device]:
    with initialize(version_base="1.3", config_path="../configs"):
        config = compose(config_name=config_name)

    OmegaConf.register_new_resolver("eval", eval)

    # Initialize model
    device = torch.device(config.compnode.device)
    diffuser = instantiate(config.diffuser)

    add_safe_globals([ListConfig])
    state_dict = torch.load(
        config.checkpoint_path, 
        map_location=device,
        weights_only=False  # Allow loading non-tensor objects
    )["state_dict"]
    
    state_dict["ema.initted"] = diffuser.ema.initted
    state_dict["ema.step"] = diffuser.ema.step
    diffuser.load_state_dict(state_dict, strict=False)
    diffuser.to(device).eval()

    # Initialize CLIP model
    clip_model = load_clip_model("ViT-B/32", device)

    # Initialize dataset
    config.batch_size = 1
    dataset = instantiate(config.dataset)
    dataset.set_split("test")
    diffuser.modalities = list(dataset.modality_datasets.keys())
    diffuser.get_matrix = dataset.get_matrix
    diffuser.v_get_matrix = dataset.get_matrix

    return diffuser, clip_model, dataset, device
