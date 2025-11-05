# VSCode Setup Guide for Art Style Classifier

This guide shows you how to run the Art Style Classifier locally in VS Code.

## Prerequisites

- VS Code installed on your machine
- Python 3.11 or higher
- Git (optional, for cloning)

## Step 1: Get the Project Files

Download or clone the project files to your local machine.

## Step 2: Open in VS Code

1. Open VS Code
2. Click **File → Open Folder**
3. Select the project directory

## Step 3: Set Up Python Environment

### Install Python Extension
1. Open the Extensions panel (Ctrl+Shift+X or Cmd+Shift+X)
2. Search for "Python"
3. Install the official Python extension by Microsoft

### Create Virtual Environment (Recommended)

Open the integrated terminal in VS Code (Ctrl+` or Cmd+`) and run:

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

## Step 4: Install Dependencies

### Option A: Using UV (Fast)
```bash
pip install uv
uv sync
```

### Option B: Using pip
```bash
pip install matplotlib numpy pandas pillow scikit-learn seaborn streamlit tensorflow
```

## Step 5: Run the Application

In the VS Code terminal:
```bash
streamlit run app.py --server.port 5000
```

The app will open in your browser at `http://localhost:5000`

## Useful VS Code Commands

### Run Streamlit App
```bash
streamlit run app.py --server.port 5000
```

### Generate Training Dataset
```bash
python create_dataset.py
```

### Install New Package
```bash
pip install <package-name>
```

### Stop the Server
Press `Ctrl+C` in the terminal

## Recommended VS Code Extensions

- **Python** (Microsoft) - Python language support
- **Pylance** - Fast Python language server
- **Python Debugger** - Debugging support
- **autoDocstring** - Generate docstrings automatically

## Project Structure

```
project/
├── app.py              # Main Streamlit application
├── create_dataset.py   # Dataset generation script
├── dataset/            # Training images by style
│   ├── Abstract/
│   ├── Cubist/
│   ├── Impressionist/
│   ├── Renaissance/
│   └── Surrealist/
├── models/             # Saved trained models
└── pyproject.toml      # Dependencies
```

## Debugging in VS Code

1. Set breakpoints by clicking left of line numbers
2. Press `F5` to start debugging
3. Select "Python File" as the debug configuration

For Streamlit apps, create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Streamlit",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": [
                "run",
                "app.py",
                "--server.port",
                "5000"
            ]
        }
    ]
}
```

## Common Issues

### Port Already in Use
Change the port in the command:
```bash
streamlit run app.py --server.port 8501
```

### ModuleNotFoundError
Make sure your virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
# or
uv sync
```

### TensorFlow Import Error
Install TensorFlow separately:
```bash
pip install tensorflow
```

### Permission Errors on Mac/Linux
```bash
chmod +x create_dataset.py
```

## Tips

- **Auto-reload**: Streamlit automatically reloads when you save changes to `app.py`
- **Clear cache**: Click "Clear cache" in the Streamlit app menu (☰) if you encounter issues
- **Terminal shortcuts**: 
  - Open terminal: Ctrl+` (Cmd+` on Mac)
  - New terminal: Ctrl+Shift+` (Cmd+Shift+` on Mac)
- **Multi-cursor editing**: Alt+Click (Option+Click on Mac)

## Performance

For faster development:
1. Use a smaller dataset during testing
2. Reduce epochs when experimenting
3. Close unnecessary browser tabs
4. Use GPU if available (TensorFlow will auto-detect)

## Additional Resources

- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
