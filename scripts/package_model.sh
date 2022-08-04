find . -not \( -path "./venv*" -or -path "./.git*" -or -path "./.ipynb_checkpoints*" \
    -or -name ".DS_Store" -or -path "./trained*" -or -path "*__pycache__*" -or -path "./results*" \) \
    | zip -@ model-package