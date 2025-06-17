#!/usr/bin/env python3
"""
Quick syntax validation and GUI launch test
"""

import ast
import sys

def validate_syntax():
    """Validate syntax of key files"""
    files_to_check = [
        'launch_enhanced_gui_clean.py',
        'launch_complete_enhanced_gui.py',
        'gui/components/complete_enhanced_gui.py',
        'gui/components/real_time_data_provider.py'
    ]
    
    print("🔍 Validating syntax...")
    all_good = True
    
    for filepath in files_to_check:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse as AST to check syntax
            ast.parse(content, filename=filepath)
            print(f"✅ {filepath} - syntax OK")
            
        except SyntaxError as e:
            print(f"❌ {filepath} - syntax error: {e}")
            print(f"   Line {e.lineno}: {e.text.strip() if e.text else 'N/A'}")
            all_good = False
        except FileNotFoundError:
            print(f"⚠️ {filepath} - file not found")
        except Exception as e:
            print(f"❌ {filepath} - error: {e}")
            all_good = False
    
    return all_good

def test_imports():
    """Test critical imports"""
    print("\n🔍 Testing imports...")
    
    try:
        from gui.components.real_time_data_provider import RealTimeDataProvider
        print("✅ RealTimeDataProvider import OK")
        
        provider = RealTimeDataProvider()
        print("✅ RealTimeDataProvider instantiation OK")
        
        from gui.components.complete_enhanced_gui import CompleteEnhancedGUI
        print("✅ CompleteEnhancedGUI import OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    print("🧪 GUI LAUNCH VALIDATION")
    print("=" * 40)
    
    syntax_ok = validate_syntax()
    imports_ok = test_imports()
    
    print("\n" + "=" * 40)
    print(f"Syntax validation: {'✅ PASSED' if syntax_ok else '❌ FAILED'}")
    print(f"Import validation: {'✅ PASSED' if imports_ok else '❌ FAILED'}")
    
    if syntax_ok and imports_ok:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ Ready to launch GUI")
        print("\n🚀 Try these commands:")
        print("   python launch_enhanced_gui_clean.py")
        print("   python launch_complete_enhanced_gui.py")
        return True
    else:
        print("\n❌ Some validations failed")
        return False

if __name__ == "__main__":
    main()
