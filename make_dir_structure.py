import os
from pathlib import Path

def create_mtuq_tutorial_structure():
    base_dir = "mtuq_tutorials"
    
    # Define the exact directory tree we discussed
    directories = [
        f"{base_dir}/shared_data",
        f"{base_dir}/1_GFs_calculation/FK_GFs",
        f"{base_dir}/1_GFs_calculation/3D_GFs",
        f"{base_dir}/2_synthetic_calculation",
        f"{base_dir}/3_single_source_evaluation",
        f"{base_dir}/4_MTUQ_mt_search_scripts/Grid_searches",
        f"{base_dir}/4_MTUQ_mt_search_scripts/Sampling_methods"
    ]
    
    print(f"Creating repository structure in: ./{base_dir}/\n")
    
    # Generate the folders
    for dir_path in directories:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {path}")
        
        # Optional: Add a .gitkeep file to each leaf directory so Git tracks them 
        # even before you copy your scripts over.
        gitkeep = path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    # Generate a master README.md
    readme_path = Path(f"{base_dir}/README.md")
    if not readme_path.exists():
        readme_text = (
            "# MTUQ Tutorials\n\n"
            "This repository contains structured tutorials and workflows for Earth Scientists "
            "learning to use the Moment Tensor Uncertainty Quantification (MTUQ) code.\n"
        )
        readme_path.write_text(readme_text)
        print(f"\nCreated file:   {readme_path}")
        
    print("\nDirectory structure successfully created! You are ready to copy your scripts.")

if __name__ == "__main__":
    create_mtuq_tutorial_structure()