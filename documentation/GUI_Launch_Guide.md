# VoxSigil GUI Launch Guide

## 🚀 How to Launch Everything in the GUI

### Quick Start Options

#### 1. **Double-click the Batch File (Easiest)**
   - Navigate to `batch_files/`
   - Double-click `Launch_VoxSigil_GUI.bat`
   - This will automatically try all GUI options

#### 2. **Use the Python Launcher**
   ```bash
   python launch_everything_gui.py
   ```

#### 3. **Direct GUI Launch (Recommended)**
   ```bash
   python "working_gui/standalone_enhanced_gui.py"
   ```

### Available GUI Versions

1. **Standalone Enhanced GUI** ⭐ (Recommended)
   - Location: `working_gui/standalone_enhanced_gui.py`
   - Features: Full featured, no external dependencies
   - Most stable option

2. **Crash-Proof Enhanced GUI** 🛡️
   - Location: `working_gui/crash_proof_enhanced_gui.py`
   - Features: Extra error handling, safe tab switching
   - Use if standalone version has issues

3. **Ultra Minimal GUI** 🔧
   - Location: `scripts/ultra_minimal_gui.py`
   - Features: Basic functionality, emergency fallback

## 🎛️ GUI Features

The VoxSigil GUI provides access to:

- **Agent Management** - Configure and run AI agents
- **ARC Integration** - Abstract Reasoning Corpus tools
- **BLT Components** - Business Logic Tier operations
- **System Monitoring** - View logs and performance
- **Configuration** - Adjust system settings
- **Testing Tools** - Run tests and diagnostics
- **Documentation** - Built-in help and guides

## 🔧 Troubleshooting

### If GUI Won't Start:
1. Check PyQt5 installation: `python -c "import PyQt5; print('OK')"`
2. Try the crash-proof version
3. Check logs in the `logs/` directory
4. Use the ultra minimal GUI as fallback

### If Tabs Don't Work:
- The standalone version handles tab switching safely
- Avoid rapid clicking between tabs
- Check the console for error messages

### Performance Issues:
- Close unused applications
- Check memory usage in Task Manager
- Use the minimal GUI for lower resource usage

## 🚨 **TAB CLOSING ISSUE - SOLUTION**

### **Problem**: GUI closes when clicking tabs
**Solution**: Use the crash-proof version instead!

```bash
python "working_gui\crash_proof_enhanced_gui.py"
```

### **Why This Happens:**
- Some GUI versions have tab switching issues
- The crash-proof version uses safe placeholder tabs
- Content loads only when you click the "Load" button on each tab

### **How the Crash-Proof Version Works:**
1. Each tab shows a placeholder with a "Load" button
2. Click the "🚀 Load [TabName]" button to load content
3. No crashes when switching between tabs
4. Safe tab handling with error recovery

## 🔥 **SYSTEM AUTO-STARTUP ADDED!**

### **Problem SOLVED:** No more "Waiting for data"

The GUI now **automatically starts up all VoxSigil systems** when launched:

#### **🚀 Auto-Initialization Sequence:**
1. **VantaCore Orchestration Engine** - Starts automatically
2. **Agent Systems** - Initializes all available agents (andy, astra, oracle, echo, dreamer, etc.)
3. **Monitoring Systems** - Starts real-time system monitoring
4. **Training Pipelines** - Initializes training infrastructure
5. **Processing Engines** - GridFormer, ARC, BLT, RAG engines
6. **Live Data Streaming** - Real system metrics (CPU, memory, agent status)

#### **📊 Real Data Sources:**
- **System Performance:** Real CPU, memory, disk usage via psutil
- **Agent Status:** Actual agent initialization and status
- **VantaCore Health:** Real orchestration engine status
- **Training Metrics:** Live training job status and progress
- **Network Activity:** Real network I/O statistics

#### **🎯 What You'll See:**
✅ **"🔄 VantaCore: Starting..."** - System initialization messages  
✅ **"🔄 Agents: Starting..."** - Agent system startup  
✅ **"🟢 System Online - Streaming Live Data"** - Real data flowing  
✅ **Live metrics updating every second** - Real system stats  

## 📁 File Locations

- **Main Launcher**: `launch_everything_gui.py`
- **Batch Launcher**: `batch_files/Launch_VoxSigil_GUI.bat`
- **GUI Files**: `working_gui/` directory
- **Scripts**: `scripts/` directory
- **Configuration**: `config/` directory
- **Logs**: `logs/` directory
