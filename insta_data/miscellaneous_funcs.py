import subprocess
import os
import random

def cleanup_resources():
    """Clean up browser processes and Docker resources to prevent container bloat."""
    try:
        print("Cleaning up resources...")
        # Kill any lingering browser processes
        subprocess.run(["pkill", "-f", "playwright"], check=False)
        subprocess.run(["pkill", "-f", "chromium"], check=False)
        
        # Clean Docker if available
        try:
            # Remove unused containers
            subprocess.run(["docker", "container", "prune", "-f"], check=False)
            # Remove unused images
            subprocess.run(["docker", "image", "prune", "-f"], check=False)
            # Remove unused volumes
            subprocess.run(["docker", "volume", "prune", "-f"], check=False)
            # General system prune
            subprocess.run(["docker", "system", "prune", "-f"], check=False)
        except Exception as e:
            print(f"Docker cleanup error: {e}")
            
        # Clear temporary files
        temp_dirs = ["/tmp/playwright", "/tmp/pyppeteer"]
        for temp_dir in temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    subprocess.run(["rm", "-rf", temp_dir], check=False)
            except Exception as e:
                print(f"Error cleaning temp dir {temp_dir}: {e}")
                
        print("Resource cleanup completed")
    except Exception as e:
        print(f"Error during resource cleanup: {e}")

def get_optimized_browser_config(port=None):
    """
    Get an optimized browser configuration with memory-saving settings.
    
    Args:
        port (int, optional): Playwright port to use. If None, a random port is chosen.
        
    Returns:
        BrowserConfig: Optimized browser configuration
    """
    from insta.configs import get_browser_config
    
    if port is None:
        port = 3000 + random.randint(0, 8)
        
    return get_browser_config(
        playwright_port=port,
        clear_browser_data=True,
        browser_args=[
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-extensions",
            "--disable-software-rasterizer",
            "--disable-background-networking",
            "--disable-default-apps",
            "--mute-audio",
            "--no-default-browser-check",
            "--no-first-run",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-background-periodic-tasks",
            "--memory-pressure-off",
            "--disable-features=BlinkGenPropertyTrees,TranslateUI",
            "--disable-ipc-flooding-protection"
        ],
        # Set reasonable memory limits
        browser_context_kwargs={
            "viewport": {"width": 1280, "height": 720},  # Smaller viewport
            "java_script_enabled": True,
            "bypass_csp": True
        }
    )