#!/bin/bash

ENV_FILE="environment.yaml"
ANACONDA_DIR="$HOME/anaconda3"
ANACONDA_BIN="$ANACONDA_DIR/bin/conda"
ANACONDA_INSTALLER="Anaconda3.sh"
ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh"

# Function to install Anaconda
install_anaconda() {
    echo "📦 Downloading and installing Anaconda..."

    curl -L -o "$ANACONDA_INSTALLER" "$ANACONDA_URL"
    bash "$ANACONDA_INSTALLER" -b -p "$ANACONDA_DIR"
    rm "$ANACONDA_INSTALLER"

    echo "✅ Anaconda installed to $ANACONDA_DIR"

    # Add to PATH for future shells
    SHELL_RC="$HOME/.bashrc"
    [[ "$SHELL" == *"zsh" ]] && SHELL_RC="$HOME/.zshrc"

    echo "export PATH=\"$ANACONDA_DIR/bin:\$PATH\"" >> "$SHELL_RC"
    echo "✅ Added Conda to PATH in $SHELL_RC"
    export PATH="$ANACONDA_DIR/bin:$PATH"
}

# Step 1: Check if conda is available
if ! command -v conda &> /dev/null; then
    if [ -x "$ANACONDA_BIN" ]; then
        echo "⚠️ Conda not in PATH but found in $ANACONDA_BIN. Adding to PATH."
        export PATH="$ANACONDA_DIR/bin:$PATH"
    else
        install_anaconda
    fi
else
    echo "✅ Conda is already installed."
fi

# Step 2: Check if YAML exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ $ENV_FILE not found in current directory: $(pwd)"
    exit 1
fi

# Step 3: Extract env name
ENV_NAME=$(grep "^name:" "$ENV_FILE" | cut -d ' ' -f2)
if [ -z "$ENV_NAME" ]; then
    echo "❌ Could not extract environment name from $ENV_FILE"
    exit 1
fi

# Step 4: Create the environment
echo "📦 Creating environment: $ENV_NAME"
conda env create -f "$ENV_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create environment"
    exit 1
fi

echo "✅ Environment '$ENV_NAME' created."
echo "🔁 To activate it, run:"
echo "    conda activate $ENV_NAME"


# CV2 library error
apt-get update
apt-get install -y libsm6 libxext6 libxrender1
