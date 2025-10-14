"""
Configuration and parameter management for 3D Gaussian Splatting.

Copyright (C) 2023, Inria
GRAPHDECO research group, https://team.inria.fr/graphdeco
All rights reserved.

This software is free for non-commercial, research and evaluation use 
under the terms of the LICENSE.md file.

For inquiries contact george.drettakis@inria.fr
"""

import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Dict


@dataclass
class IDUParams:
    """Parameters for Iterative Dataset Update (IDU)."""
    elevation_list: List[float] = field(default_factory=list)
    radius_list: List[float] = field(default_factory=list)
    fov: float = 60.0


class GroupParams:
    """Empty class to hold extracted parameter groups."""
    pass


class ParamGroup:
    """Base class for parameter groups with automatic argument parsing."""
    
    def __init__(self, parser: ArgumentParser, name: str, fill_none: bool = False):
        """
        Initialize parameter group and add arguments to parser.
        
        Args:
            parser: ArgumentParser to add arguments to
            name: Name of the argument group
            fill_none: If True, set all defaults to None
        """
        group = parser.add_argument_group(name)
        
        for attr_name, default_value in vars(self).items():
            self._add_argument(group, attr_name, default_value, fill_none)
    
    def _add_argument(self, group, attr_name: str, default_value: Any, fill_none: bool):
        """Add a single argument to the parser group."""
        # Check if attribute should have shorthand (starts with underscore)
        has_shorthand = attr_name.startswith("_")
        clean_name = attr_name[1:] if has_shorthand else attr_name
        
        value_type = type(default_value)
        final_default = None if fill_none else default_value
        
        # Build argument parameters
        arg_names = [f"--{clean_name}"]
        if has_shorthand:
            arg_names.append(f"-{clean_name[0]}")
        
        # Add argument based on type
        if value_type == bool:
            group.add_argument(*arg_names, default=final_default, action="store_true")
        else:
            group.add_argument(*arg_names, default=final_default, type=value_type)
    
    def extract(self, args: Namespace) -> GroupParams:
        """
        Extract this group's parameters from parsed arguments.
        
        Args:
            args: Parsed arguments namespace
            
        Returns:
            GroupParams object with extracted values
        """
        group = GroupParams()
        
        for arg_name, arg_value in vars(args).items():
            # Check both with and without underscore prefix
            if arg_name in vars(self) or f"_{arg_name}" in vars(self):
                setattr(group, arg_name, arg_value)
        
        return group


class ModelParams(ParamGroup):
    """Parameters for model loading and configuration."""
    
    def __init__(self, parser: ArgumentParser, sentinel: bool = False):
        # Appearance modeling
        self.sh_degree: int = 3
        self.appearance_enabled: bool = False
        self.appearance_n_fourier_freqs: int = 4
        self.appearance_embedding_dim: int = 32
        
        # Paths (underscore prefix indicates shorthand option available)
        self._source_path: str = ""
        self._model_path: str = ""
        self._images: str = "images"
        
        # Resolution and rendering
        self._resolution: int = -1
        self._white_background: bool = False
        self.data_device: str = "cuda"
        self._kernel_size: float = 0.1
        
        # Training options
        self.eval: bool = False
        self.ray_jitter: bool = False
        self.resample_gt_image: bool = False
        self.load_allres: bool = False
        self.sample_more_highres: bool = False
        
        super().__init__(parser, "Loading Parameters", sentinel)
    
    def extract(self, args: Namespace) -> GroupParams:
        """Extract parameters and convert source_path to absolute path."""
        group = super().extract(args)
        group.source_path = str(Path(group.source_path).resolve())
        return group


class PipelineParams(ParamGroup):
    """Parameters for rendering pipeline configuration."""
    
    def __init__(self, parser: ArgumentParser):
        self.convert_SHs_python: bool = False
        self.compute_cov3D_python: bool = False
        self.debug: bool = False
        
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    """Parameters for optimization and training."""
    
    # Constants
    DEFAULT_ITERATIONS = 30_000
    DEFAULT_DENSIFY_UNTIL = 20_000
    
    def __init__(self, parser: ArgumentParser):
        # Basic training parameters
        self.iterations: int = self.DEFAULT_ITERATIONS
        
        # Learning rates - position
        self.position_lr_init: float = 0.00016
        self.position_lr_final: float = 0.0000016
        self.position_lr_delay_mult: float = 0.01
        self.position_lr_max_steps: int = self.DEFAULT_ITERATIONS
        
        # Learning rates - features
        self.feature_lr: float = 0.0025
        self.opacity_lr: float = 0.05
        self.scaling_lr: float = 0.005
        self.rotation_lr: float = 0.001
        
        # Densification parameters
        self.percent_dense: float = 0.01
        self.densification_interval: int = 100
        self.opacity_reset_interval: int = 3000
        self.densify_from_iter: int = 1000
        self.densify_until_iter: int = self.DEFAULT_DENSIFY_UNTIL
        self.densify_grad_threshold: float = 0.0002
        
        # Loss weights
        self.lambda_dssim: float = 0.2
        self.lambda_depth: float = 0.5
        self.lambda_opacity: float = 0.1  # Entropy-based opacity regularization
        
        # Appearance modeling learning rates
        self.embedding_lr: float = 0.005
        self.appearance_embedding_lr: float = 0.001
        self.appearance_embedding_regularization: float = 0
        self.appearance_mlp_lr: float = 0.0005
        
        # Pruning
        self.size_threshold: int = 20
        
        # LPIPS loss
        self.use_lpips_loss: bool = False
        self.lpips_net: str = "alex"  # Options: 'vgg', 'alex', 'squeeze'
        
        # Pseudo camera depth supervision
        self.sample_pseudo_interval: int = 10
        self.start_sample_pseudo: int = 2000
        self.end_sample_pseudo: int = 9500
        self.lambda_pseudo_depth: float = 0.0
        self.num_pseudo_cams: int = 24
        self.target_std: float = 64.0
        
        # IDU (Iterative Dataset Update) parameters
        self._init_idu_params()
        
        # DDIM inversion parameters
        self._init_ddim_params()
        
        # FlowEdit parameters
        self._init_flowedit_params()
        
        # Difix3D parameters
        self._init_difix3d_params()
        
        # DreamScene parameters
        self.idu_use_dreamscene: bool = False
        self.idu_use_sd21: bool = True
        
        # Post-training
        self.post_training_iterations: int = 500
        
        super().__init__(parser, "Optimization Parameters")
    
    def _init_idu_params(self):
        """Initialize IDU-specific parameters."""
        # Basic IDU settings
        self.idu_no_curriculum: bool = False
        self.idu_episode_iterations: int = 10000
        self.idu_densify_until_iter: int = 7500
        self.idu_opacity_reset_interval: int = 5000
        self.idu_opacity_cooling_iterations: int = 1000
        self.idu_testing_interval: int = 5000  # idu_episode_iterations // 2
        
        # IDU refinement
        self.idu_refine: bool = False
        self.idu_random_ap: bool = False
        self.idu_iter_full_train: int = 0
        self.idu_num_cams: int = 12
        self.idu_num_samples_per_view: int = 4
        self.idu_train_ratio: float = 0.5
        
        # Dataset configuration
        self.datasets_type: str = "jax_v1"
        self.idu_params: Dict[str, IDUParams] = {
            "jax_v1": IDUParams(
                elevation_list=[85., 75., 65., 55., 45.],
                radius_list=[300., 275., 275., 250., 250.],
                fov=60.0
            ),
            "nyc_v1": IDUParams(
                elevation_list=[85., 75., 65., 55., 45., 25.],
                radius_list=[600., 600., 600., 600., 600.],
                fov=20.0
            )
        }
        
        # IDU rendering
        self.idu_position_lr_max_steps: int = self.idu_episode_iterations
        self.idu_render_size: int = 1024
        
        # Look-at point grid
        self.idu_grid_width: int = 256
        self.idu_grid_height: int = 256
        self.idu_grid_size: int = 2
    
    def _init_ddim_params(self):
        """Initialize DDIM inversion parameters."""
        self.idu_ddim_strength: float = 0.2
        self.idu_ddim_eta: float = 0.5
        self.idu_ddim_step: int = 50
        self.idu_ddim_guidance_scale: float = 3.5
    
    def _init_flowedit_params(self):
        """Initialize FlowEdit parameters."""
        self.idu_use_flow_edit: bool = False
        self.idu_flow_edit_n_min: int = 0
        self.idu_flow_edit_n_max: int = 15
        self.idu_flow_edit_n_max_end: int = -1  # -1 means not using sampling
        self.idu_flow_edit_n_avg: int = 1
        self.idu_model_type: str = "FLUX"
    
    def _init_difix3d_params(self):
        """Initialize Difix3D parameters."""
        self.idu_use_difix3d: bool = False
        self.idu_difix3d_model: str = "nvidia/difix"
        self.idu_difix3d_steps: int = 1
        self.idu_difix3d_guidance: float = 0.0
        self.idu_difix3d_timesteps: List[int] = [199]
        self.idu_difix3d_use_reference: bool = False
        self.idu_difix3d_prompt: str = "remove degradation"


def get_combined_args(parser: ArgumentParser) -> Namespace:
    """
    Combine command line arguments with saved configuration file.
    
    Args:
        parser: ArgumentParser with defined arguments
        
    Returns:
        Namespace with merged arguments (command line takes precedence)
    """
    # Parse command line arguments
    cmdline_args = parser.parse_args(sys.argv[1:])
    
    # Try to load saved configuration
    config_args = Namespace()
    try:
        if cmdline_args.model_path:
            config_path = Path(cmdline_args.model_path) / "cfg_args"
            print(f"Looking for config file in {config_path}")
            
            if config_path.exists():
                print(f"Config file found: {config_path}")
                with open(config_path, 'r') as cfg_file:
                    config_args = eval(cfg_file.read())
            else:
                print(f"Config file not found at {config_path}")
    except (TypeError, AttributeError) as e:
        print(f"Could not load config file: {e}")
    
    # Merge configurations (command line overrides config file)
    merged_dict = vars(config_args).copy()
    for key, value in vars(cmdline_args).items():
        if value is not None:
            merged_dict[key] = value
    
    return Namespace(**merged_dict)