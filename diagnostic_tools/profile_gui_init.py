"""
Detailed profiling for CompleteVoxSigilGUI initialization
"""
import cProfile
import pstats
import io
import sys
import os
import time
from pathlib import Path

# Add parent directory to path so we can import from the library
sys.path.append(str(Path(__file__).parent.parent))

def profile_gui_init():
    """Profile the GUI initialization specifically"""
    print("Profiling GUI initialization...")
    
    # Create a profile object
    pr = cProfile.Profile()
    
    # Start profiling
    pr.enable()
    
    # Import and initialize the GUI
    try:
        from working_gui.complete_live_gui import CompleteVoxSigilGUI
        gui = CompleteVoxSigilGUI()
        print(f"GUI initialized with {gui.main_tabs.count()} tabs")
    except Exception as e:
        print(f"Error during profiling: {e}")
    finally:
        # Stop profiling
        pr.disable()
        
        # Print stats sorted by cumulative time
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(50)  # Print top 50 time-consuming functions
        print(s.getvalue())
        
        # Save stats to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"gui_init_profile_{timestamp}.txt"
        output_path = Path(__file__).parent.parent / "logs" / output_file
        os.makedirs(output_path.parent, exist_ok=True)
        
        with open(output_path, 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
            ps.print_stats()
            
        print(f"Full profile results saved to {output_path}")

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    profile_gui_init()
