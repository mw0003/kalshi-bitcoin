#!/usr/bin/env python3
"""
Automated Git Workflow for Kalshi Bitcoin Forecasting Project

This script provides automated git operations for the project:
- Auto-commit changes with descriptive messages
- Push to GitHub repository
- Handle common git operations

Usage:
    python git_workflow.py commit "Your commit message"
    python git_workflow.py push
    python git_workflow.py status
    python git_workflow.py auto "Optional commit message"
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, capture_output=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True, 
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def git_status():
    """Show git status"""
    print("=== Git Status ===")
    success, stdout, stderr = run_command("git status", capture_output=False)
    return success

def git_add_project_files():
    """Add specific project files to git (excluding large data files)"""
    print("=== Adding Project Files to Git ===")
    
    files_to_add = [
        "*.py",
        "requirements.txt", 
        "README.md",
        ".gitignore",
        "models/",
        "results/",
        "trained_models/"
    ]
    
    for file_pattern in files_to_add:
        success, stdout, stderr = run_command(f"git add {file_pattern}")
        if success:
            print(f"✓ Added {file_pattern}")
        else:
            print(f"⚠ Could not add {file_pattern}: {stderr}")
    
    return True

def create_feature_branch():
    """Create a feature branch for changes"""
    timestamp = int(datetime.now().timestamp())
    branch_name = f"devin/{timestamp}-bitcoin-forecasting-pipeline"
    
    print(f"=== Creating Feature Branch: {branch_name} ===")
    success, stdout, stderr = run_command(f"git checkout -b {branch_name}")
    if success:
        print(f"✓ Created and switched to branch: {branch_name}")
        return branch_name
    else:
        print(f"✗ Failed to create branch: {stderr}")
        return None

def git_commit(message=None):
    """Commit changes with a message"""
    if not message:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Auto-commit: Updated Bitcoin forecasting pipeline - {timestamp}"
    
    print(f"=== Committing Changes ===")
    print(f"Commit message: {message}")
    
    git_add_project_files()
    
    success, stdout, stderr = run_command("git diff --cached --quiet")
    if success:  # No changes staged
        print("No changes to commit.")
        return True
    
    success, stdout, stderr = run_command(f'git commit -m "{message}"')
    if success:
        print("✓ Changes committed successfully")
        return True
    else:
        print(f"✗ Commit failed: {stderr}")
        return False

def git_push():
    """Push changes to GitHub"""
    print("=== Pushing to GitHub ===")
    
    success, branch_name, stderr = run_command("git branch --show-current")
    if not success:
        print(f"✗ Could not get current branch: {stderr}")
        return False
    
    branch_name = branch_name.strip()
    print(f"Pushing branch: {branch_name}")
    
    success, stdout, stderr = run_command(f"git push origin {branch_name}")
    if success:
        print("✓ Successfully pushed to GitHub")
        return True
    else:
        print(f"✗ Push failed: {stderr}")
        print("Trying to set upstream and push...")
        success, stdout, stderr = run_command(f"git push -u origin {branch_name}")
        if success:
            print("✓ Successfully pushed to GitHub with upstream set")
            return True
        else:
            print(f"✗ Push with upstream failed: {stderr}")
            return False

def auto_commit_and_push(message=None):
    """Automatically commit and push changes"""
    print("=== Auto Commit and Push ===")
    
    if git_commit(message):
        return git_push()
    return False

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Kalshi Bitcoin Forecasting - Git Workflow")
        print("Usage:")
        print("  python git_workflow.py status")
        print("  python git_workflow.py commit [message]")
        print("  python git_workflow.py push")
        print("  python git_workflow.py auto [message]")
        print("  python git_workflow.py branch")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "status":
        git_status()
    
    elif command == "commit":
        message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        git_commit(message)
    
    elif command == "push":
        git_push()
    
    elif command == "auto":
        message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        auto_commit_and_push(message)
    
    elif command == "branch":
        create_feature_branch()
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
