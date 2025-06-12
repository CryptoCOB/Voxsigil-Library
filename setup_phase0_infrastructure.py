#!/usr/bin/env python3
"""
Phase 0: Bug Tracking Infrastructure Setup
Creates comprehensive bug tracking system for 9-phase testing roadmap
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BugTracker:
    """Comprehensive bug tracking system for 9-phase testing"""
    
    def __init__(self, base_dir: str = "testing_infrastructure"):
        self.base_dir = Path(base_dir)
        self.bugs_file = self.base_dir / "bugs.xlsx"
        self.logs_dir = self.base_dir / "logs"
        self.reports_dir = self.base_dir / "reports"
        
        # Initialize directories
        self.setup_directories()
        self.initialize_bug_tracking()
        
    def setup_directories(self):
        """Create all necessary directories for testing infrastructure"""
        directories = [
            self.base_dir,
            self.logs_dir,
            self.reports_dir,
            self.logs_dir / "phase_0",
            self.logs_dir / "phase_1", 
            self.logs_dir / "phase_2",
            self.logs_dir / "phase_3",
            self.logs_dir / "phase_4",
            self.logs_dir / "phase_5",
            self.logs_dir / "phase_6",
            self.logs_dir / "phase_7",
            self.logs_dir / "phase_8",
            self.logs_dir / "phase_9",
            self.base_dir / "configs",
            self.base_dir / "scripts",
            self.base_dir / "baselines"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")
    
    def initialize_bug_tracking(self):
        """Initialize the main bug tracking spreadsheet"""
        if self.bugs_file.exists():
            logger.info(f"📊 Bug tracking file already exists: {self.bugs_file}")
            return
            
        # Create comprehensive bug tracking structure
        bug_data = {
            'Bug_ID': [],
            'Phase': [],
            'Severity': [],
            'Component': [],
            'File_Path': [],
            'Title': [],
            'Description': [],
            'Steps_to_Reproduce': [],
            'Expected_Result': [],
            'Actual_Result': [],
            'Root_Cause': [],
            'Fix_Description': [],
            'Fix_Commit': [],
            'Status': [],
            'Assigned_To': [],
            'Date_Found': [],
            'Date_Fixed': [],
            'Test_Case': [],
            'Regression_Test': []
        }
        
        # Create sample entries for each phase to establish structure
        sample_bugs = [
            {
                'Bug_ID': 'BUG-001',
                'Phase': 'Phase 0',
                'Severity': 'Medium',
                'Component': 'Environment Setup',
                'File_Path': 'config/environment.py',
                'Title': 'Missing dependency validation',
                'Description': 'System does not validate all required dependencies on startup',
                'Steps_to_Reproduce': '1. Fresh install 2. Run main.py 3. Missing deps not caught',
                'Expected_Result': 'Clear error message about missing dependencies',
                'Actual_Result': 'Cryptic import error later in execution',
                'Root_Cause': 'No dependency checker in startup sequence',
                'Fix_Description': '',
                'Fix_Commit': '',
                'Status': 'Open',
                'Assigned_To': 'Auto-detected',
                'Date_Found': datetime.now().strftime('%Y-%m-%d'),
                'Date_Fixed': '',
                'Test_Case': 'test_dependency_validation',
                'Regression_Test': 'test_startup_dependencies'
            }
        ]
        
        # Add sample bug to tracking data
        for bug in sample_bugs:
            for key, value in bug.items():
                bug_data[key].append(value)
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame(bug_data)
        df.to_excel(self.bugs_file, index=False, sheet_name='Bug_Tracking')
        
        logger.info(f"✅ Created bug tracking spreadsheet: {self.bugs_file}")
        
    def add_bug(self, bug_info: Dict[str, Any]) -> str:
        """Add a new bug to the tracking system"""
        try:
            # Read existing data
            df = pd.read_excel(self.bugs_file, sheet_name='Bug_Tracking')
            
            # Generate new bug ID
            existing_ids = df['Bug_ID'].tolist()
            next_id = len(existing_ids) + 1
            bug_id = f"BUG-{next_id:03d}"
            
            # Prepare bug entry
            bug_entry = {
                'Bug_ID': bug_id,
                'Phase': bug_info.get('phase', 'Unknown'),
                'Severity': bug_info.get('severity', 'Medium'),
                'Component': bug_info.get('component', 'Unknown'),
                'File_Path': bug_info.get('file_path', ''),
                'Title': bug_info.get('title', 'Untitled Bug'),
                'Description': bug_info.get('description', ''),
                'Steps_to_Reproduce': bug_info.get('steps', ''),
                'Expected_Result': bug_info.get('expected', ''),
                'Actual_Result': bug_info.get('actual', ''),
                'Root_Cause': bug_info.get('root_cause', ''),
                'Fix_Description': '',
                'Fix_Commit': '',
                'Status': 'Open',
                'Assigned_To': bug_info.get('assigned_to', 'Auto-detected'),
                'Date_Found': datetime.now().strftime('%Y-%m-%d'),
                'Date_Fixed': '',
                'Test_Case': bug_info.get('test_case', ''),
                'Regression_Test': bug_info.get('regression_test', '')
            }
            
            # Add to DataFrame
            new_row = pd.DataFrame([bug_entry])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Save updated data
            df.to_excel(self.bugs_file, index=False, sheet_name='Bug_Tracking')
            
            logger.info(f"🐛 Added bug {bug_id}: {bug_entry['Title']}")
            return bug_id
            
        except Exception as e:
            logger.error(f"Failed to add bug: {e}")
            return ""
    
    def update_bug_status(self, bug_id: str, status: str, fix_details: Dict[str, Any] = None):
        """Update bug status and fix information"""
        try:
            df = pd.read_excel(self.bugs_file, sheet_name='Bug_Tracking')
            
            # Find bug row
            bug_row = df[df['Bug_ID'] == bug_id]
            if bug_row.empty:
                logger.error(f"Bug {bug_id} not found")
                return False
            
            # Update status
            df.loc[df['Bug_ID'] == bug_id, 'Status'] = status
            
            # Add fix details if provided
            if fix_details:
                if status == 'Fixed':
                    df.loc[df['Bug_ID'] == bug_id, 'Date_Fixed'] = datetime.now().strftime('%Y-%m-%d')
                
                for field, value in fix_details.items():
                    if field in df.columns:
                        df.loc[df['Bug_ID'] == bug_id, field] = value
            
            # Save updated data
            df.to_excel(self.bugs_file, index=False, sheet_name='Bug_Tracking')
            
            logger.info(f"✅ Updated bug {bug_id}: {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update bug {bug_id}: {e}")
            return False
    
    def generate_phase_report(self, phase: str) -> Dict[str, Any]:
        """Generate comprehensive report for a testing phase"""
        try:
            df = pd.read_excel(self.bugs_file, sheet_name='Bug_Tracking')
            phase_bugs = df[df['Phase'] == phase]
            
            report = {
                'phase': phase,
                'total_bugs': len(phase_bugs),
                'open_bugs': len(phase_bugs[phase_bugs['Status'] == 'Open']),
                'fixed_bugs': len(phase_bugs[phase_bugs['Status'] == 'Fixed']),
                'high_severity': len(phase_bugs[phase_bugs['Severity'] == 'High']),
                'medium_severity': len(phase_bugs[phase_bugs['Severity'] == 'Medium']),
                'low_severity': len(phase_bugs[phase_bugs['Severity'] == 'Low']),
                'components_affected': phase_bugs['Component'].unique().tolist(),
                'bug_details': phase_bugs.to_dict('records')
            }
            
            # Save report
            report_file = self.reports_dir / f"{phase.lower().replace(' ', '_')}_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Generated {phase} report: {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate {phase} report: {e}")
            return {}
    
    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall bug tracking status across all phases"""
        try:
            df = pd.read_excel(self.bugs_file, sheet_name='Bug_Tracking')
            
            status = {
                'total_bugs': len(df),
                'open_bugs': len(df[df['Status'] == 'Open']),
                'fixed_bugs': len(df[df['Status'] == 'Fixed']),
                'high_severity_open': len(df[(df['Status'] == 'Open') & (df['Severity'] == 'High')]),
                'phases_with_bugs': df['Phase'].value_counts().to_dict(),
                'components_with_bugs': df['Component'].value_counts().to_dict(),
                'bug_trend': df.groupby('Date_Found').size().to_dict()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get overall status: {e}")
            return {}

def create_testing_scripts():
    """Create automated testing scripts for each phase"""
    base_dir = Path("testing_infrastructure/scripts")
    
    # Phase 1: Static Analysis Script
    phase1_script = """#!/usr/bin/env python3
'''Phase 1: Static Analysis Sweep'''
import subprocess
import json
from pathlib import Path

def run_static_analysis():
    results = {}
    
    # Ruff
    print("🔍 Running Ruff analysis...")
    result = subprocess.run(['ruff', 'check', '.', '--output-format=json'], 
                          capture_output=True, text=True)
    results['ruff'] = json.loads(result.stdout) if result.stdout else []
    
    # MyPy
    print("🔍 Running MyPy analysis...")
    result = subprocess.run(['mypy', '.', '--ignore-missing-imports'], 
                          capture_output=True, text=True)
    results['mypy'] = result.stdout.split('\\n') if result.stdout else []
    
    # Bandit
    print("🔍 Running Bandit security analysis...")
    result = subprocess.run(['bandit', '-r', '.', '-f', 'json'], 
                          capture_output=True, text=True)
    results['bandit'] = json.loads(result.stdout) if result.stdout else {}
    
    # Save results
    with open('testing_infrastructure/logs/phase_1/static_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Static analysis complete")
    return results

if __name__ == "__main__":
    run_static_analysis()
"""
    
    # Phase 4: Bus Fuzzing Script  
    phase4_script = """#!/usr/bin/env python3
'''Phase 4: Agent Bus Fuzzing'''
import asyncio
import random
import json
import time
from datetime import datetime

class AgentBusFuzzer:
    def __init__(self):
        self.results = []
        
    async def random_event_storm(self, duration_minutes=5):
        '''Generate random agent messages for chaos testing'''
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            # Generate random events
            event = {
                'type': random.choice(['agent_message', 'system_event', 'user_input']),
                'payload': self._generate_random_payload(),
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                # Send event to bus (mock for now)
                await self._send_to_bus(event)
                await asyncio.sleep(random.uniform(0.01, 0.1))  # Random timing
            except Exception as e:
                self.results.append({
                    'error': str(e),
                    'event': event,
                    'timestamp': datetime.now().isoformat()
                })
    
    def _generate_random_payload(self):
        '''Generate random payload for testing'''
        return {
            'data': ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(10, 100))),
            'number': random.randint(1, 1000),
            'nested': {'key': random.choice([None, 'value', 123, [1,2,3]])}
        }
    
    async def _send_to_bus(self, event):
        '''Mock bus sending - replace with actual implementation'''
        # Simulate processing delay
        await asyncio.sleep(0.001)
        
        # Randomly fail to simulate issues
        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Simulated bus failure")

async def main():
    fuzzer = AgentBusFuzzer()
    await fuzzer.random_event_storm(duration_minutes=2)
    
    # Save results
    with open('testing_infrastructure/logs/phase_4/bus_fuzzing_results.json', 'w') as f:
        json.dump(fuzzer.results, f, indent=2)
    
    print(f"🌪️ Bus fuzzing complete: {len(fuzzer.results)} issues found")

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    # Write scripts to files
    scripts = {
        'phase_1_static_analysis.py': phase1_script,
        'phase_4_bus_fuzzing.py': phase4_script
    }
    
    for filename, content in scripts.items():
        script_path = base_dir / filename
        with open(script_path, 'w') as f:
            f.write(content)
        
        # Make executable on Unix systems
        try:
            os.chmod(script_path, 0o755)
        except:
            pass  # Windows doesn't support chmod the same way
        
        logger.info(f"📜 Created testing script: {script_path}")

def setup_phase_0_infrastructure():
    """Complete Phase 0 setup with all infrastructure"""
    logger.info("🚀 Setting up Phase 0: Bug Tracking Infrastructure")
    
    # Create bug tracker
    tracker = BugTracker()
    
    # Create testing scripts
    create_testing_scripts()
    
    # Create baseline configuration
    baseline_config = {
        'environment': {
            'python_version': '3.8+',
            'required_packages': [
                'PyQt5', 'torch', 'transformers', 'numpy', 
                'pytest', 'pytest-cov', 'ruff', 'mypy', 
                'bandit', 'pylint', 'pandas'
            ],
            'vanta_flags': {
                'VANTA_LOG_LEVEL': 'DEBUG',
                'PYTHONWARNINGS': 'default',
                'VANTA_ENABLE_METRICS': 'true'
            }
        },
        'testing_phases': {
            'phase_0': {'status': 'active', 'expected_bugs': '2-3'},
            'phase_1': {'status': 'pending', 'expected_bugs': '10-16'},
            'phase_2': {'status': 'pending', 'expected_bugs': '10-14'},
            'phase_3': {'status': 'pending', 'expected_bugs': '6-10'},
            'phase_4': {'status': 'pending', 'expected_bugs': '6-10'},
            'phase_5': {'status': 'pending', 'expected_bugs': '6-10'},
            'phase_6': {'status': 'pending', 'expected_bugs': '5-8'},
            'phase_7': {'status': 'pending', 'expected_bugs': '6-10'},
            'phase_8': {'status': 'pending', 'expected_bugs': '4-8'},
            'phase_9': {'status': 'pending', 'expected_bugs': '0-1'}
        }
    }
    
    config_file = Path("testing_infrastructure/configs/baseline_config.json")
    with open(config_file, 'w') as f:
        json.dump(baseline_config, f, indent=2)
    
    logger.info(f"⚙️ Created baseline configuration: {config_file}")
    
    # Create phase execution script
    execution_script = """#!/usr/bin/env python3
'''9-Phase Testing Execution Controller'''
import json
import subprocess
import sys
from pathlib import Path

def execute_phase(phase_num):
    '''Execute specific testing phase'''
    phase_scripts = {
        1: 'phase_1_static_analysis.py',
        4: 'phase_4_bus_fuzzing.py'
    }
    
    if phase_num in phase_scripts:
        script_path = Path(f"testing_infrastructure/scripts/{phase_scripts[phase_num]}")
        if script_path.exists():
            print(f"🚀 Executing Phase {phase_num}...")
            result = subprocess.run([sys.executable, str(script_path)], 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Errors: {result.stderr}")
            return result.returncode == 0
        else:
            print(f"❌ Script not found: {script_path}")
            return False
    else:
        print(f"⚠️ Phase {phase_num} script not implemented yet")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python execute_phase.py <phase_number>")
        sys.exit(1)
    
    try:
        phase = int(sys.argv[1])
        success = execute_phase(phase)
        sys.exit(0 if success else 1)
    except ValueError:
        print("Phase number must be an integer")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    
    exec_script_path = Path("testing_infrastructure/scripts/execute_phase.py")
    with open(exec_script_path, 'w') as f:
        f.write(execution_script)
    
    logger.info(f"🎮 Created phase execution controller: {exec_script_path}")
    
    # Summary
    logger.info("✅ Phase 0 Infrastructure Setup Complete!")
    logger.info("📊 Bug tracking spreadsheet ready")
    logger.info("📁 All logging directories created")
    logger.info("📜 Testing scripts generated")
    logger.info("⚙️ Baseline configuration saved")
    logger.info("🎮 Phase execution controller ready")
    
    return tracker

if __name__ == "__main__":
    setup_phase_0_infrastructure()
