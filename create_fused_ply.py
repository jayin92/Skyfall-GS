import torch
import os
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--output_ply", type=str, default="./output.ply")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--load_from_checkpoints", action="store_true")
    args = get_combined_args(parser)
    print("create fused ply for " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset.appearance_enabled, dataset.appearance_n_fourier_freqs, dataset.appearance_embedding_dim)
    if args.load_from_checkpoints:
        checkpoint_path = os.path.join(dataset.model_path, f"chkpnt{args.iteration}.pth")
        print(f"Loading model from checkpoint {checkpoint_path}")
        (model_params, first_iter) = torch.load(checkpoint_path)
        gaussians.load_from_checkpoints(model_params)
    gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply"))
    gaussians.save_fused_ply(args.output_ply, args.load_from_checkpoints)
    
