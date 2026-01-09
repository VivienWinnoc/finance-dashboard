#!/usr/bin/env python3
"""
Daily Report Generation Script
==============================
This script is designed to be run via cron at 8 PM daily.

Cron configuration (add to crontab with 'crontab -e'):
0 20 * * * /path/to/venv/bin/python /path/to/finance-dashboard/scripts/generate_report.py >> /path/to/logs/cron.log 2>&1

Example for a typical setup:
0 20 * * * /home/user/finance-dashboard/venv/bin/python /home/user/finance-dashboard/scripts/generate_report.py >> /home/user/finance-dashboard/logs/cron.log 2>&1
"""

import sys
import os
from datetime import datetime

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

def main():
    """Generate daily report and log the result."""
    
    print(f"\n{'='*50}")
    print(f"Report Generation Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")
    
    try:
        from utils.report_generator import ReportGenerator
        
        # Initialize generator
        generator = ReportGenerator(
            reports_dir=os.path.join(parent_dir, 'reports')
        )
        
        # Generate daily report
        report_path = generator.generate_daily_report()
        print(f"✅ Daily report generated: {report_path}")
        
        # Cleanup old reports (keep last 30 days)
        generator.cleanup_old_reports(days_to_keep=30)
        print("✅ Old reports cleaned up")
        
        print(f"\n{'='*50}")
        print(f"Report Generation Completed Successfully")
        print(f"{'='*50}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error generating report: {str(e)}")
        print(f"{'='*50}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
