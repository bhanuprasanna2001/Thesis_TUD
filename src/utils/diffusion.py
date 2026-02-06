import cv2
import torch
import lightning as L
import torchvision as tv
from pathlib import Path
from PIL import Image
import numpy as np


def get_scheduler(start=0.0001, end=0.02, timesteps=1000, type="linear"):
    if type == "linear":
        return torch.linspace(start, end, timesteps)
    else:
        raise NotImplementedError(f"{type} has not been implemented.")


@torch.no_grad()
def sample_denoising_process(model, n_samples=1, save_every=50):
    """Sample images while capturing intermediate denoising steps.
    
    Args:
        model: Diffusion model instance
        n_samples: Number of samples to generate
        save_every: Save intermediate step every N timesteps
        
    Returns:
        dict with 'final' samples and 'intermediate' steps
    """
    was_training = model.training
    model.eval()
    device = model.device
    
    # Use EMA weights if available
    if model.use_ema and model.ema is not None:
        model.ema.apply(model.network)
    
    try:
        x = torch.randn(n_samples, *model.img_shape, device=device)
        intermediates = []
        
        for i in reversed(range(model.timesteps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            x = model.p_sample(x, t, i)
            
            if i % save_every == 0 or i == 0:
                intermediates.append({
                    'timestep': i,
                    'image': x.cpu().clone()
                })
        
        return {
            'final': x.cpu(),
            'intermediate': intermediates
        }
    finally:
        if model.use_ema and model.ema is not None:
            model.ema.restore(model.network)
        
        if was_training:
            model.train()


def create_denoising_grid(intermediates, nrow=None, save_path=None):
    """Create grid visualization of denoising process.
    
    Shows progression from noise to clean image across columns.
    Each row is a different sample.
    
    Args:
        intermediates: List of dicts with 'timestep' and 'image' keys
        nrow: Number of images per row (default: number of timesteps)
        save_path: Path to save the grid image
        
    Returns:
        PIL Image of the grid
    """
    if nrow is None:
        nrow = len(intermediates)
    
    # Collect all timestep images
    images = []
    for step_data in intermediates:
        img = step_data['image']
        img = (img.clamp(-1, 1) + 1) / 2
        images.append(img)
    
    # Stack: [n_timesteps, n_samples, C, H, W]
    images = torch.stack(images, dim=0)
    n_timesteps, n_samples, C, H, W = images.shape
    
    # Transpose to [n_samples, n_timesteps, C, H, W]
    images = images.transpose(0, 1)
    
    # Reshape to [n_samples * n_timesteps, C, H, W]
    images = images.reshape(n_samples * n_timesteps, C, H, W)
    
    # Create grid
    grid = tv.utils.make_grid(images, nrow=n_timesteps, padding=2, pad_value=1.0)
    
    # Convert to PIL
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    grid_pil = Image.fromarray(grid_np)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        grid_pil.save(save_path)
    
    return grid_pil


def create_transition_video(model, n_samples=16, output_path="transition.mp4", fps=30, duration=10):
    """Create video showing transition from noise to clean samples.
    
    Args:
        model: Diffusion model instance
        n_samples: Number of samples to show in grid
        output_path: Path to save MP4 video
        fps: Frames per second
        duration: Video duration in seconds
        
    Returns:
        Path to saved video file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate grid dimensions
    nrow = int(np.sqrt(n_samples))
    
    # Sample with intermediate steps
    total_frames = fps * duration
    save_every = max(1, model.timesteps // total_frames)
    
    print(f"Generating {n_samples} samples with {total_frames} frames...")
    result = sample_denoising_process(model, n_samples=n_samples, save_every=save_every)
    
    intermediates = result['intermediate']
    
    # Prepare video writer
    first_grid = tv.utils.make_grid(
        (intermediates[0]['image'].clamp(-1, 1) + 1) / 2,
        nrow=nrow, padding=2, pad_value=1.0
    )
    height, width = first_grid.shape[1], first_grid.shape[2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Creating video at {output_path}...")
    for step_data in intermediates:
        img = step_data['image']
        img = (img.clamp(-1, 1) + 1) / 2
        
        grid = tv.utils.make_grid(img, nrow=nrow, padding=2, pad_value=1.0)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        grid_bgr = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)
        writer.write(grid_bgr)
    
    writer.release()
    print(f"Video saved to {output_path}")
    
    return output_path


def generate_sample_report(model, output_dir, n_samples=16, n_grids=4):
    """Generate sampling report with images and video.
    
    Creates:
    - Multiple sample grids
    - Denoising progression grid
    - Transition video
    
    Args:
        model: Diffusion model instance
        output_dir: Directory to save outputs
        n_samples: Number of samples per visualization
        n_grids: Number of sample grids to generate
        
    Returns:
        dict with paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating sample report in {output_dir}")
    print("=" * 60)
    
    generated_files = {}
    
    # 1. Generate regular sample grids
    print(f"\n1. Generating {n_grids} sample grids...")
    nrow = int(np.sqrt(n_samples))
    for i in range(n_grids):
        print(f"   Grid {i+1}/{n_grids}...")
        samples = model.sample(n_samples)
        samples = (samples.clamp(-1, 1) + 1) / 2
        
        save_path = output_dir / f"samples_grid_{i+1:02d}.png"
        tv.utils.save_image(samples, save_path, nrow=nrow, padding=2, pad_value=1.0)
        generated_files[f'grid_{i+1}'] = save_path
    
    # 2. Generate denoising progression
    print(f"\n2. Generating denoising progression grid...")
    result = sample_denoising_process(model, n_samples=4, save_every=100)
    
    save_path = output_dir / "denoising_progression.png"
    create_denoising_grid(result['intermediate'], save_path=save_path)
    generated_files['denoising_grid'] = save_path
    
    # 3. Generate transition video
    print(f"\n3. Generating transition video...")
    video_path = output_dir / "denoising_transition.mp4"
    create_transition_video(model, n_samples=16, output_path=video_path, fps=30, duration=10)
    generated_files['video'] = video_path
    
    print("\n" + "=" * 60)
    print("Sample report complete!")
    print(f"\nGenerated files:")
    for key, path in generated_files.items():
        print(f"  {key}: {path}")
    
    return generated_files

