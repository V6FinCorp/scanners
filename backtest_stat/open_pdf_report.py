"""
PDF Report Viewer
Simple utility to open the generated PDF report
"""

import os
import sys
import subprocess
import platform

def open_pdf(pdf_path):
    """Open PDF file using the system default PDF viewer"""
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        if platform.system() == "Windows":
            os.startfile(pdf_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", pdf_path])
        else:  # Linux
            subprocess.run(["xdg-open", pdf_path])
        
        print(f"📖 Opening PDF: {pdf_path}")
        return True
        
    except Exception as e:
        print(f"❌ Could not open PDF: {e}")
        print(f"📁 PDF location: {os.path.abspath(pdf_path)}")
        return False

def main():
    """Main function to open the PDF report"""
    pdf_filename = "EMA_Crossover_Strategy_Report.pdf"
    
    print("🔍 Looking for PDF report...")
    
    if os.path.exists(pdf_filename):
        file_size = os.path.getsize(pdf_filename) / 1024
        print(f"✅ Found PDF report ({file_size:.1f} KB)")
        
        # Show file info
        print(f"📄 File: {pdf_filename}")
        print(f"📍 Path: {os.path.abspath(pdf_filename)}")
        
        # Open PDF
        if open_pdf(pdf_filename):
            print("🎉 PDF opened successfully!")
        
    else:
        print(f"❌ PDF report not found: {pdf_filename}")
        print("💡 Run 'python generate_pdf_report.py' to create the report first.")

if __name__ == "__main__":
    main()