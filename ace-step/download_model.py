#!/usr/bin/env python3
"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License

This script downloads the ACE-Step model during Docker build
to include the model files in the Docker image.
"""

import os
import sys
import json
from acestep.pipeline_ace_step import ACEStepPipeline

def download_model(checkpoint_dir=None, home_dir=None):
    """
    Download the ACE-Step model to the specified checkpoint directory.
    
    Args:
        checkpoint_dir (str, optional): Directory to store the downloaded model
        home_dir (str, optional): Home directory to use for default cache location
    """
    # Set environment variables for model download
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    if home_dir:
        # Override HOME environment variable to control where .cache is located
        original_home = os.environ.get('HOME')
        os.environ['HOME'] = home_dir
        print(f"Temporarily setting HOME to {home_dir}")
    
    try:
        # Initialize the pipeline which will download the model
        print(f"Downloading ACE-Step model...")
        if checkpoint_dir:
            print(f"Using custom checkpoint directory: {checkpoint_dir}")
        pipeline = ACEStepPipeline(checkpoint_dir=checkpoint_dir)
        pipeline.load_checkpoint()
        
        # Print the actual location where models were downloaded
        actual_checkpoint_dir = pipeline.checkpoint_dir
        print(f"Models were downloaded to: {actual_checkpoint_dir}")
        
        # Create a flag file to indicate the model has been downloaded
        flag_file = os.path.join(actual_checkpoint_dir, "model_downloaded.flag")
        with open(flag_file, 'w') as f:
            json.dump({
                'downloaded': True,
                'timestamp': str(os.path.getmtime(actual_checkpoint_dir)),
                'path': actual_checkpoint_dir
            }, f)
        print(f"Created flag file at: {flag_file}")
        
        print("Model download complete!")
        return actual_checkpoint_dir
    finally:
        # Restore original HOME if we changed it
        if home_dir and original_home:
            os.environ['HOME'] = original_home

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    else:
        checkpoint_dir = None
        
    # Use /home/appuser as home directory if no specific checkpoint_dir is provided
    # This ensures models are downloaded to /home/appuser/.cache/ace-step/checkpoints
    home_dir = "/home/appuser" if not checkpoint_dir else None
    
    actual_dir = download_model(checkpoint_dir, home_dir)
    print(f"Final checkpoint directory: {actual_dir}")
    
    # Write the actual directory to a file for the Dockerfile to use
    with open('/tmp/checkpoint_path.txt', 'w') as f:
        f.write(actual_dir)
