# gen_render_path.py
# Generate a camera path for orbit view around a target point.
# Usage example:
# python gen_render_path.py --fov 20 --target 0,0,0 --elevation 80 --radius 200 --num_frame 240 --fps 24 --height 512 --width 512 --output_folder camera_path

import os
import argparse
import json
import math
import numpy as np


def look_at_to_c2w(eye, target, up):
    # Convert inputs to numpy arrays if needed
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)
    
    # Calculate forward (negative z-axis)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    # Calculate right vector (x-axis)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Calculate corrected up vector (y-axis)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Construct rotation matrix
    R = np.array([
        [right[0], up[0], -forward[0]],
        [right[1], up[1], -forward[1]],
        [right[2], up[2], -forward[2]]
    ])
    
    # Construct full 4x4 transformation matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    
    return c2w

def gen_path(target, elevation, radius, num_frame):
    # Convert target to numpy array
    target = np.array(target)
        
    # Calculate up vector
    up = np.array([0, 0, 1])
    
    # Generate camera-to-world matrices
    c2ws = []
    for i in range(num_frame):
        theta = -2 * np.pi * i / num_frame # + np.pi / 2  # Start from the top
        phi = np.pi * elevation / 180
        eye = target + np.array([
            radius * np.cos(theta) * np.cos(phi),
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(phi)       
        ])
        c2w = look_at_to_c2w(eye, target, up)
        c2ws.append(c2w)
    
    return c2ws

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--target", type=str, default="0,0,0")
    parser.add_argument("--elevation", type=float, default=0)
    parser.add_argument("--radius", type=float, default=200)   # Radius of the sphere
    parser.add_argument("--num_frame", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--ges", action="store_true", help="Use Google Earth Studio parameters")
    parser.add_argument("--alt_tar", type=float)
    parser.add_argument("--alt_cam", type=float)

    
    args = parser.parse_args()
    if args.ges:
        assert args.alt_tar is not None, "alt_tar must be specified when using --ges"
        assert args.alt_cam is not None, "alt_cam must be specified when using --ges"
        print("Convert Google Earth Studio parameters to camera path")
        alt_delta = args.alt_cam - args.alt_tar
        args.elevation = math.degrees(math.atan2(alt_delta, args.radius))
        args.radius = math.sqrt(args.radius**2 + alt_delta**2)
        print(f"Converted elevation: {args.elevation}, radius: {args.radius}")

    args.target = [float(x) for x in args.target.split(",")]
    output_dict = {
        "_target": args.target,
        "_radius": args.radius,
        "_elevation": args.elevation,
        "camera_type": "perspective",
        "render_height": args.height,
        "render_width": args.width,
        "fps": args.fps,
    }
    c2ws = gen_path(
        args.target,
        args.elevation,
        args.radius,
        args.num_frame
    )
    output_dict["camera_path"] = [
        {
            "camera_to_world": c2w.flatten().tolist(),
            "fov": args.fov,
            "aspect": 1  # not sure what this is, but appears in nerfstudio
        }
        for c2w in c2ws
    ]
    output_path = os.path.join(
        args.output_folder, 
        f"r{int(args.radius)}_e{int(args.elevation)}_fov{int(args.fov)}.json"
    )
    os.makedirs(args.output_folder, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=4)
    
    print("Camera path saved to", output_path)
