# Open WebUI Debug Setup Guide

This guide will help you set up VS Code debugging for the Open WebUI project.

## Prerequisites

1. **Python 3.11+** installed
2. **Node.js 18+** and npm installed  
3. **VS Code** with the following extensions installed:
   - Python
   - Python Debugger (Pylance)
   - Svelte for VS Code (if working with frontend)
   - ESLint
   - Prettier (optional)

## Quick Setup

1. **Clone and open the project in VS Code**
   ```bash
   git clone https://github.com/open-webui/open-webui.git
   cd open-webui
   code .
   ```

2. **Set up Python virtual environment**
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cd ..
   ```

3. **Install Node.js dependencies**
   ```bash
   npm install
   ```

4. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

## Debug Configurations

The project includes several debug configurations in `.vscode/launch.json`:

### 1. Backend Debug (Python)
- **Name**: "Backend Debug (Python)"
- **Purpose**: Debug the FastAPI backend server
- **Port**: 8080 (backend API)
- **Features**: 
  - Hot reload enabled
  - Full debugging with breakpoints
  - Environment variables configured

### 2. Frontend Dev Server (Node.js) 
- **Name**: "Frontend Dev Server (Node.js)"
- **Purpose**: Debug the Svelte frontend
- **Port**: 3000 (frontend dev server)
- **Features**:
  - Hot reload enabled
  - Source maps for debugging

### 3. Full Stack Debug
- **Name**: "Full Stack Debug" 
- **Purpose**: Start both backend and frontend simultaneously
- **Features**:
  - Compound configuration
  - Both servers running with debugging

## Starting Debug Sessions

### Option 1: Using VS Code Debug Panel
1. Open VS Code Debug panel (`Ctrl+Shift+D`)
2. Select the debug configuration you want:
   - "Backend Debug (Python)" - for backend only
   - "Frontend Dev Server (Node.js)" - for frontend only  
   - "Full Stack Debug" - for both
3. Click the green play button or press `F5`

### Option 2: Using Tasks (Alternative)
1. Open Command Palette (`Ctrl+Shift+P`)
2. Type "Tasks: Run Task"
3. Choose from available tasks:
   - "Backend: Start Server"
   - "Frontend: Start Dev Server" 
   - "Full Stack: Start Both Servers"

## Environment Configuration

Edit your `.env` file to configure the application:

```bash
# Basic settings
ENV=dev
WEBUI_SECRET_KEY=your-secret-key-here
PORT=8080
HOST=0.0.0.0

# Enable services as needed
ENABLE_OLLAMA_API=True
OLLAMA_BASE_URLS=http://localhost:11434

# CORS for frontend development
CORS_ALLOW_ORIGIN=["http://localhost:3000"]
```

## Development Workflow

1. **Backend Development:**
   - Start "Backend Debug (Python)" configuration
   - Set breakpoints in Python files
   - API available at: http://localhost:8080
   - API docs at: http://localhost:8080/docs

2. **Frontend Development:**
   - Start "Frontend Dev Server (Node.js)" configuration  
   - Frontend available at: http://localhost:3000
   - Hot reload active for Svelte files

3. **Full Stack Development:**
   - Use "Full Stack Debug" configuration
   - Both servers running simultaneously
   - Frontend proxies API requests to backend

## Breakpoints and Debugging

### Python Backend
- Set breakpoints in any `.py` file in the `backend/` directory
- Debug routes in `backend/open_webui/routers/`
- Debug models in `backend/open_webui/models/`
- Debug utilities in `backend/open_webui/utils/`

### JavaScript/TypeScript Frontend
- Set breakpoints in `.svelte`, `.js`, or `.ts` files
- Debug components in `src/lib/components/`
- Debug routes in `src/routes/`
- Use browser dev tools for additional debugging

## Common Issues

### Backend Issues
- **Port already in use**: Change PORT in .env or stop other processes
- **Module not found**: Ensure PYTHONPATH is set correctly
- **Permission denied**: Make sure `backend/start.sh` is executable

### Frontend Issues  
- **Node modules not found**: Run `npm install`
- **Build errors**: Check Node.js version (18+ required)
- **CORS errors**: Verify CORS_ALLOW_ORIGIN in .env

### Debug Issues
- **Debugger not attaching**: Check if ports are available
- **Breakpoints not hit**: Verify source maps and file paths
- **Environment variables**: Check .env file and VS Code settings

## Advanced Configuration

### Custom Python Path
Update `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./backend/.venv/bin/python"
}
```

### Custom Environment Variables
Add to debug configuration in `.vscode/launch.json`:
```json
"env": {
  "CUSTOM_VAR": "value",
  "ANOTHER_VAR": "another_value"
}
```

## Testing

Run tests using:
- `Backend: Run Tests` task for Python tests
- `Frontend: Run Tests` task for JavaScript tests
- Or use the test discovery in VS Code's Testing panel

## Production Build

To create a production build:
1. Run `Frontend: Build Production` task
2. The built files will be in the appropriate output directory
3. Backend serves both API and static files in production mode

For more detailed information, see the main project README and documentation.