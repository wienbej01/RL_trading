#!/usr/bin/env python3
"""
Plan Update Utility for Multi-Ticker and Reward Overhaul Program

This utility provides a command-line interface for updating the program status,
tracking progress, and generating reports.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import json
from typing import Dict, Any, List, Optional

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logging import get_logger

logger = get_logger(__name__)

# Default paths
DEFAULT_STATUS_FILE = Path(__file__).parent.parent / "docs" / "roadmap" / "STATUS.yaml"
DEFAULT_PLAN_FILE = Path(__file__).parent.parent / "docs" / "roadmap" / "multiticker_reward_overhaul.md"


class PlanUpdater:
    """Utility class for updating and managing the program plan."""
    
    def __init__(self, status_file: Path = DEFAULT_STATUS_FILE):
        """
        Initialize the PlanUpdater.
        
        Args:
            status_file: Path to the STATUS.yaml file
        """
        self.status_file = status_file
        self.status_data = self._load_status()
    
    def _load_status(self) -> Dict[str, Any]:
        """Load the status data from the YAML file."""
        try:
            with open(self.status_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Status file not found: {self.status_file}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing status file: {e}")
            return {}
    
    def _save_status(self):
        """Save the status data to the YAML file."""
        try:
            with open(self.status_file, 'w') as f:
                yaml.dump(self.status_data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Status saved to {self.status_file}")
        except Exception as e:
            logger.error(f"Error saving status file: {e}")
    
    def update_task_status(self, task_path: List[str], status: str, progress: Optional[int] = None):
        """
        Update the status of a specific task.
        
        Args:
            task_path: List of keys to navigate to the task (e.g., ["phases", 0, "tasks", 0])
            status: New status ("pending", "in_progress", "completed", "blocked")
            progress: Progress percentage (0-100)
        """
        try:
            # Navigate to the task
            current = self.status_data
            for key in task_path[:-1]:
                current = current[key]
            
            task = current[task_path[-1]]
            
            # Update task status
            task["status"] = status
            if progress is not None:
                task["progress_percentage"] = progress
            
            if status == "completed":
                task["completed_date"] = datetime.now().strftime("%Y-%m-%d")
            
            # Update parent progress
            self._update_parent_progress(task_path[:-1])
            
            # Update last modified timestamp
            self.status_data["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            
            self._save_status()
            logger.info(f"Updated task status: {' -> '.join(map(str, task_path))} = {status}")
            
        except KeyError as e:
            logger.error(f"Task not found: {e}")
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
    
    def _update_parent_progress(self, parent_path: List[str]):
        """Update the progress of parent elements based on their children."""
        try:
            current = self.status_data
            for key in parent_path:
                current = current[key]
            
            if "tasks" in current:
                # Calculate progress from tasks
                tasks = current["tasks"]
                if tasks:
                    total_progress = sum(task.get("progress_percentage", 0) for task in tasks)
                    avg_progress = total_progress / len(tasks)
                    current["progress_percentage"] = avg_progress
                    
                    # Determine status based on tasks
                    completed_count = sum(1 for task in tasks if task.get("status") == "completed")
                    in_progress_count = sum(1 for task in tasks if task.get("status") == "in_progress")
                    
                    if completed_count == len(tasks):
                        current["status"] = "completed"
                    elif in_progress_count > 0 or completed_count > 0:
                        current["status"] = "in_progress"
                    else:
                        current["status"] = "pending"
            
            # Recursively update parents
            if len(parent_path) > 1:
                self._update_parent_progress(parent_path[:-1])
            
            # Update overall program progress
            if "phases" in self.status_data:
                phases = self.status_data["phases"]
                if phases:
                    total_progress = sum(phase.get("progress_percentage", 0) for phase in phases)
                    avg_progress = total_progress / len(phases)
                    self.status_data["progress_percentage"] = avg_progress
                    
                    # Determine overall status
                    completed_count = sum(1 for phase in phases if phase.get("status") == "completed")
                    in_progress_count = sum(1 for phase in phases if phase.get("status") == "in_progress")
                    
                    if completed_count == len(phases):
                        self.status_data["status"] = "completed"
                    elif in_progress_count > 0 or completed_count > 0:
                        self.status_data["status"] = "in_progress"
                    else:
                        self.status_data["status"] = "pending"
        
        except KeyError as e:
            logger.error(f"Parent not found: {e}")
        except Exception as e:
            logger.error(f"Error updating parent progress: {e}")
    
    def get_task_status(self, task_path: List[str]) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific task.
        
        Args:
            task_path: List of keys to navigate to the task
            
        Returns:
            Task dictionary or None if not found
        """
        try:
            current = self.status_data
            for key in task_path:
                current = current[key]
            return current
        except KeyError:
            logger.error(f"Task not found: {task_path}")
            return None
    
    def list_tasks(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all tasks with optional status filtering.
        
        Args:
            status_filter: Filter tasks by status ("pending", "in_progress", "completed", "blocked")
            
        Returns:
            List of task dictionaries with path information
        """
        tasks = []
        
        def _collect_tasks(current: Dict[str, Any], path: List[str]):
            if "tasks" in current:
                for i, task in enumerate(current["tasks"]):
                    task_path = path + ["tasks", i]
                    task_info = task.copy()
                    task_info["path"] = task_path
                    
                    if status_filter is None or task.get("status") == status_filter:
                        tasks.append(task_info)
                    
                    # Check subtasks
                    if "subtasks" in task:
                        for j, subtask in enumerate(task["subtasks"]):
                            subtask_path = task_path + ["subtasks", j]
                            subtask_info = subtask.copy()
                            subtask_info["path"] = subtask_path
                            
                            if status_filter is None or subtask.get("status") == status_filter:
                                tasks.append(subtask_info)
        
        # Collect tasks from all phases
        for i, phase in enumerate(self.status_data.get("phases", [])):
            _collect_tasks(phase, ["phases", i])
        
        return tasks
    
    def generate_report(self, output_format: str = "text") -> str:
        """
        Generate a status report.
        
        Args:
            output_format: Output format ("text", "json", "markdown")
            
        Returns:
            Formatted report string
        """
        program_info = {
            "name": self.status_data.get("program", {}).get("name", "Unknown"),
            "status": self.status_data.get("status", "Unknown"),
            "progress": self.status_data.get("progress_percentage", 0),
            "last_updated": self.status_data.get("last_updated", "Unknown")
        }
        
        if output_format == "json":
            return json.dumps({
                "program": program_info,
                "phases": self.status_data.get("phases", []),
                "milestones": self.status_data.get("milestones", []),
                "risks": self.status_data.get("risks", [])
            }, indent=2)
        
        elif output_format == "markdown":
            report = f"# {program_info['name']} Status Report\n\n"
            report += f"**Status**: {program_info['status']}\n"
            report += f"**Progress**: {program_info['progress']}%\n"
            report += f"**Last Updated**: {program_info['last_updated']}\n\n"
            
            report += "## Phases\n\n"
            for phase in self.status_data.get("phases", []):
                report += f"### {phase['name']}\n"
                report += f"- **Status**: {phase.get('status', 'Unknown')}\n"
                report += f"- **Progress**: {phase.get('progress_percentage', 0)}%\n"
                report += f"- **Target Completion**: {phase.get('target_completion', 'Unknown')}\n\n"
            
            report += "## Milestones\n\n"
            for milestone in self.status_data.get("milestones", []):
                status_icon = "✅" if milestone.get("status") == "completed" else "⏳"
                report += f"{status_icon} **{milestone['name']}** ({milestone.get('target_completion', 'Unknown')})\n"
                report += f"   - Status: {milestone.get('status', 'Unknown')}\n\n"
            
            return report
        
        else:  # text format
            report = f"{program_info['name']} Status Report\n"
            report += "=" * len(program_info['name']) + "\n\n"
            report += f"Status: {program_info['status']}\n"
            report += f"Progress: {program_info['progress']}%\n"
            report += f"Last Updated: {program_info['last_updated']}\n\n"
            
            report += "Phases:\n"
            for phase in self.status_data.get("phases", []):
                report += f"  - {phase['name']}: {phase.get('status', 'Unknown')} ({phase.get('progress_percentage', 0)}%)\n"
            
            report += "\nMilestones:\n"
            for milestone in self.status_data.get("milestones", []):
                status_icon = "✓" if milestone.get("status") == "completed" else "○"
                report += f"  {status_icon} {milestone['name']} ({milestone.get('target_completion', 'Unknown')})\n"
            
            return report
    
    def add_risk(self, name: str, category: str, probability: str, impact: str, mitigation: str):
        """
        Add a new risk to the status file.
        
        Args:
            name: Risk name
            category: Risk category ("technical", "schedule", "resource")
            probability: Probability level ("low", "medium", "high")
            impact: Impact level ("low", "medium", "high")
            mitigation: Mitigation strategy
        """
        risk = {
            "name": name,
            "category": category,
            "probability": probability,
            "impact": impact,
            "mitigation": mitigation,
            "status": "monitored"
        }
        
        if "risks" not in self.status_data:
            self.status_data["risks"] = []
        
        self.status_data["risks"].append(risk)
        self._save_status()
        logger.info(f"Added risk: {name}")
    
    def update_milestone(self, name: str, status: str):
        """
        Update the status of a milestone.
        
        Args:
            name: Milestone name
            status: New status ("pending", "completed")
        """
        for milestone in self.status_data.get("milestones", []):
            if milestone["name"] == name:
                milestone["status"] = status
                if status == "completed":
                    milestone["completed_date"] = datetime.now().strftime("%Y-%m-%d")
                self._save_status()
                logger.info(f"Updated milestone: {name} = {status}")
                return
        
        logger.error(f"Milestone not found: {name}")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(description="Update program plan status")
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE,
                       help="Path to the STATUS.yaml file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Update task status command
    update_parser = subparsers.add_parser("update", help="Update task status")
    update_parser.add_argument("task_path", help="Task path (e.g., 'phases.0.tasks.0')")
    update_parser.add_argument("status", choices=["pending", "in_progress", "completed", "blocked"],
                               help="New task status")
    update_parser.add_argument("--progress", type=int, help="Progress percentage (0-100)")
    
    # Get task status command
    get_parser = subparsers.add_parser("get", help="Get task status")
    get_parser.add_argument("task_path", help="Task path (e.g., 'phases.0.tasks.0')")
    
    # List tasks command
    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--status", choices=["pending", "in_progress", "completed", "blocked"],
                            help="Filter by status")
    
    # Generate report command
    report_parser = subparsers.add_parser("report", help="Generate status report")
    report_parser.add_argument("--format", choices=["text", "json", "markdown"], default="text",
                               help="Output format")
    
    # Add risk command
    risk_parser = subparsers.add_parser("add-risk", help="Add a new risk")
    risk_parser.add_argument("name", help="Risk name")
    risk_parser.add_argument("category", choices=["technical", "schedule", "resource"],
                            help="Risk category")
    risk_parser.add_argument("probability", choices=["low", "medium", "high"],
                            help="Probability level")
    risk_parser.add_argument("impact", choices=["low", "medium", "high"],
                            help="Impact level")
    risk_parser.add_argument("mitigation", help="Mitigation strategy")
    
    # Update milestone command
    milestone_parser = subparsers.add_parser("update-milestone", help="Update milestone status")
    milestone_parser.add_argument("name", help="Milestone name")
    milestone_parser.add_argument("status", choices=["pending", "completed"],
                                 help="New milestone status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    updater = PlanUpdater(args.status_file)
    
    if args.command == "update":
        # Parse task path
        path_parts = []
        for part in args.task_path.split('.'):
            try:
                path_parts.append(int(part))
            except ValueError:
                path_parts.append(part)
        
        updater.update_task_status(path_parts, args.status, args.progress)
    
    elif args.command == "get":
        # Parse task path
        path_parts = []
        for part in args.task_path.split('.'):
            try:
                path_parts.append(int(part))
            except ValueError:
                path_parts.append(part)
        
        task = updater.get_task_status(path_parts)
        if task:
            print(f"Task: {task.get('name', 'Unknown')}")
            print(f"Status: {task.get('status', 'Unknown')}")
            print(f"Progress: {task.get('progress_percentage', 0)}%")
            if "completed_date" in task:
                print(f"Completed: {task['completed_date']}")
        else:
            print("Task not found")
    
    elif args.command == "list":
        tasks = updater.list_tasks(args.status)
        for task in tasks:
            path_str = '.'.join(str(p) for p in task['path'])
            print(f"{path_str}: {task.get('name', 'Unknown')} - {task.get('status', 'Unknown')} ({task.get('progress_percentage', 0)}%)")
    
    elif args.command == "report":
        report = updater.generate_report(args.format)
        print(report)
    
    elif args.command == "add-risk":
        updater.add_risk(args.name, args.category, args.probability, args.impact, args.mitigation)
    
    elif args.command == "update-milestone":
        updater.update_milestone(args.name, args.status)


if __name__ == "__main__":
    main()