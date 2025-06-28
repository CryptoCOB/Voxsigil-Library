#!/usr/bin/env python3
"""
VoxSigil GUI Process Manager
Checks for and manages VoxSigil GUI instances to prevent random pop-ups
"""

import sys

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available - install with: pip install psutil")


def find_gui_processes():
    """Find all VoxSigil GUI processes"""
    if not PSUTIL_AVAILABLE:
        return []

    gui_processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if (
                "python" in proc.info["name"].lower()
                and proc.info["cmdline"]
                and "streamlined_training_gui" in " ".join(proc.info["cmdline"])
            ):
                gui_processes.append(
                    {
                        "pid": proc.info["pid"],
                        "cmdline": " ".join(proc.info["cmdline"]),
                        "create_time": proc.info["create_time"],
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return gui_processes


def kill_gui_processes(keep_newest=True):
    """Kill GUI processes, optionally keeping the newest one"""
    processes = find_gui_processes()

    if not processes:
        print("No VoxSigil GUI processes found")
        return

    print(f"Found {len(processes)} VoxSigil GUI process(es):")
    for proc in processes:
        print(f"  PID {proc['pid']}: {proc['cmdline']}")

    if keep_newest and len(processes) > 1:
        # Sort by creation time, keep the newest
        processes.sort(key=lambda x: x["create_time"], reverse=True)
        newest = processes[0]
        to_kill = processes[1:]

        print(f"\nKeeping newest process (PID {newest['pid']})")
        print("Killing older processes:")

        for proc in to_kill:
            try:
                psutil.Process(proc["pid"]).terminate()
                print(f"  ✅ Terminated PID {proc['pid']}")
            except Exception as e:
                print(f"  ❌ Failed to terminate PID {proc['pid']}: {e}")

    elif not keep_newest:
        print("\nKilling all GUI processes:")
        for proc in processes:
            try:
                psutil.Process(proc["pid"]).terminate()
                print(f"  ✅ Terminated PID {proc['pid']}")
            except Exception as e:
                print(f"  ❌ Failed to terminate PID {proc['pid']}: {e}")


def main():
    """Main function"""
    print("🔍 VoxSigil GUI Process Manager")
    print("=" * 40)

    if len(sys.argv) > 1:
        action = sys.argv[1].lower()

        if action == "list":
            processes = find_gui_processes()
            if processes:
                print(f"Found {len(processes)} VoxSigil GUI process(es):")
                for proc in processes:
                    print(f"  PID {proc['pid']}: {proc['cmdline']}")
            else:
                print("No VoxSigil GUI processes found")

        elif action == "kill":
            kill_gui_processes(keep_newest=False)

        elif action == "cleanup":
            kill_gui_processes(keep_newest=True)

        else:
            print("Usage:")
            print("  python gui_manager.py list     - List all GUI processes")
            print("  python gui_manager.py kill     - Kill all GUI processes")
            print(
                "  python gui_manager.py cleanup  - Kill duplicate processes, keep newest"
            )

    else:
        print("Usage:")
        print("  python gui_manager.py list     - List all GUI processes")
        print("  python gui_manager.py kill     - Kill all GUI processes")
        print(
            "  python gui_manager.py cleanup  - Kill duplicate processes, keep newest"
        )


if __name__ == "__main__":
    main()
