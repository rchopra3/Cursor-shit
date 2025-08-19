#!/usr/bin/env python3
"""
Test script to verify project setup and dependencies.
Run this before running the main analysis to ensure everything is working.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'pandas',
        'numpy', 
        'yfinance',
        'matplotlib',
        'seaborn',
        'plotly',
        'scipy',
        'sqlite3'
    ]
    
    print("🧪 Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_file_structure():
    """Test if all required project files exist."""
    required_files = [
        'requirements.txt',
        'schema.sql',
        'queries.sql', 
        'data_engineer.py',
        'financial_analysis.py',
        'README.md'
    ]
    
    print("\n📁 Testing project file structure...")
    missing_files = []
    
    for file in required_files:
        try:
            with open(file, 'r') as f:
                pass
            print(f"✅ {file}")
        except FileNotFoundError:
            print(f"❌ {file} - Not found")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ All project files found!")
        return True

def test_database_creation():
    """Test if database can be created from schema."""
    print("\n🗄️ Testing database creation...")
    
    try:
        import sqlite3
        
        # Try to create a test database
        conn = sqlite3.connect(':memory:')
        
        # Read and execute schema
        with open('schema.sql', 'r') as f:
            schema_sql = f.read()
        
        cursor = conn.cursor()
        cursor.executescript(schema_sql)
        
        # Test basic queries
        cursor.execute("SELECT COUNT(*) FROM assets")
        asset_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT ticker FROM assets")
        tickers = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        print(f"✅ Database created successfully with {asset_count} assets")
        print(f"✅ Assets: {', '.join(tickers)}")
        return True
        
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Financial Portfolio Analysis Dashboard - Setup Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test file structure
    files_ok = test_file_structure()
    
    # Test database creation
    db_ok = test_database_creation()
    
    print("\n" + "=" * 60)
    
    if all([imports_ok, files_ok, db_ok]):
        print("🎉 All tests passed! Your project is ready to run.")
        print("\nNext steps:")
        print("1. Run: python data_engineer.py")
        print("2. Run: python financial_analysis.py")
    else:
        print("❌ Some tests failed. Please fix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
