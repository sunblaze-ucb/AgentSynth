#!/usr/bin/env python3
"""
Launcher script for the AgentSynth evaluation system.
This script provides easy access to the evaluation system from the main directory.
"""

import sys
import os
from pathlib import Path

# Add the evaluation system to the path
evaluation_system_path = Path(__file__).parent / "evaluation_system"
sys.path.insert(0, str(evaluation_system_path))

# Import and run the main evaluation script
if __name__ == "__main__":
    # Change to the evaluation system scripts directory
    scripts_dir = evaluation_system_path / "scripts"
    os.chdir(scripts_dir)
    
    # Import and run the main script
    from run_agentsynth_evaluation import main
    main()
