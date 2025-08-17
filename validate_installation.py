#!/usr/bin/env python
"""Validate LIPAC installation."""

import sys
import importlib
from pathlib import Path

def check_python():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if v >= (3, 10):
        print("✅ Python version OK")
        return True
    else:
        print("❌ Python 3.10+ required")
        return False

def check_packages():
    # Core packages (absolutely required)
    core = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy',
            'MDAnalysis', 'tqdm']
    # Nice to have (not critical)
    optional = ['pyyaml', 'pymc', 'arviz', 'joblib']

    print("\nCore packages:")
    core_ok = True
    for pkg in core:
        try:
            importlib.import_module(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg}")
            core_ok = False

    print("\nOptional packages:")
    for pkg in optional:
        try:
            importlib.import_module(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"⚠️  {pkg} (optional)")

    return core_ok

def check_modules():
    print("\nLIPAC modules:")
    modules = [
        'stage1_contact_analysis.main',
        'stage1_contact_analysis.config',
        'stage2_contact_analysis.main',
        'stage2_contact_analysis.config'
    ]

    all_ok = True
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"✅ {mod}")
        except ImportError as e:
            print(f"❌ {mod}: {e}")
            all_ok = False

    return all_ok

def check_test_data():
    print("\nTest data:")
    test_dir = Path("test_data")
    files = ["test_system_minimal.pdb", "test_trajectory_minimal.xtc",
             "config_test.yaml"]

    if all((test_dir / f).exists() for f in files):
        print("✅ Test data found")
        return True
    else:
        print("❌ Test data missing")
        print("   Run: cd test_data && python create_minimal_test_data.py")
        return False

def main():
    print("🔍 LIPAC Installation Validator")
    print("=" * 40)

    python_ok = check_python()
    packages_ok = check_packages()
    modules_ok = check_modules()
    test_ok = check_test_data()

    print(f"\nSummary:")
    print(f"Python: {'✅' if python_ok else '❌'}")
    print(f"Core Packages: {'✅' if packages_ok else '❌'}")
    print(f"Modules: {'✅' if modules_ok else '❌'}")
    print(f"Test Data: {'✅' if test_ok else '❌'}")

    if python_ok and packages_ok and modules_ok:
        print("\n🎉 LIPAC is ready!")
        if not test_ok:
            print("💡 Create test data: cd test_data && python create_minimal_test_data.py")
    else:
        print("\n❌ Fix issues above")
        if not packages_ok:
            print("💡 Install missing packages: pip install -r requirements.txt")
            print("💡 For pyyaml: pip install pyyaml")

if __name__ == "__main__":
    main()