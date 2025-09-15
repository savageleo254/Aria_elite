#!/usr/bin/env python3
"""
ARIA ELITE - Gemini-Powered Audit Agent
Autonomous code auditing, error detection, and patch generation
"""

import os
import ast
import sys
import asyncio
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import re
import subprocess

import google.generativeai as genai
from utils.discord_notifier import discord_notifier
from utils.logger import setup_logger

logger = setup_logger(__name__)

class GeminiAuditAgent:
    """Autonomous Gemini-powered code audit and patch system"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.gemini_model = None
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.audit_config = self._load_audit_config()
        self.is_initialized = False
        
        # Audit patterns and checks
        self.critical_patterns = [
            r"hardcoded.*key|api.*key.*=.*['\"][^'\"]*['\"]",
            r"password.*=.*['\"][^'\"]*['\"]",
            r"secret.*=.*['\"][^'\"]*['\"]",
            r"time\.sleep\(\d+\)",
            r"random\.\w+\(",
            r"simulation.*mode|mock.*mode",
            r"fallback.*to.*mock|fallback.*simulation",
            r"todo|fixme|hack",
            r"print\(|console\.log\(",
        ]
        
        self.file_patterns = [
            "*.py", "*.ts", "*.tsx", "*.js", "*.jsx", 
            "*.json", "*.yaml", "*.yml", "*.env*"
        ]
        
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found - audit agent disabled")
            return
            
    def _load_audit_config(self) -> Dict[str, Any]:
        """Load audit configuration"""
        return {
            "max_file_size": 100000,  # Skip files larger than 100KB
            "ignore_patterns": [
                "node_modules", ".venv", "__pycache__", ".git", 
                "dist", "build", ".next", "logs"
            ],
            "critical_files": [
                "backend/core/execution_engine.py",
                "backend/core/mt5_bridge.py", 
                "start_live_trading.py",
                "discord_agent/discord_bot.py"
            ],
            "patch_mode": "auto",  # auto, manual, alert-only
            "max_patches_per_run": 5
        }
    
    async def initialize(self):
        """Initialize Gemini audit agent"""
        try:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY required for audit agent")
                
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Test connection
            test_response = self.gemini_model.generate_content("Test audit connection")
            if not test_response.text:
                raise Exception("Gemini audit connection test failed")
                
            await discord_notifier.initialize()
            
            self.is_initialized = True
            logger.info("Gemini audit agent initialized successfully")
            
            # Send startup notification
            await discord_notifier.send_system_status({
                "audit_agent": {"status": "healthy", "initialized_at": datetime.now().isoformat()},
                "gemini_connection": {"status": "healthy"},
                "discord_notifications": {"status": "healthy"}
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini audit agent: {str(e)}")
            await discord_notifier.send_critical_error(
                error=str(e),
                component="GeminiAuditAgent",
                traceback=traceback.format_exc()
            )
            raise
    
    async def run_full_audit(self) -> Dict[str, Any]:
        """Run comprehensive audit of entire codebase"""
        if not self.is_initialized:
            raise Exception("Audit agent not initialized")
            
        logger.info("Starting full codebase audit")
        audit_results = {
            "timestamp": datetime.now().isoformat(),
            "files_scanned": 0,
            "issues_found": [],
            "patches_applied": [],
            "critical_errors": [],
            "syntax_errors": []
        }
        
        try:
            # Scan all relevant files
            files_to_scan = self._get_files_to_scan()
            
            for file_path in files_to_scan:
                try:
                    file_result = await self._audit_file(file_path)
                    audit_results["files_scanned"] += 1
                    
                    if file_result.get("issues"):
                        audit_results["issues_found"].extend(file_result["issues"])
                    
                    if file_result.get("syntax_errors"):
                        audit_results["syntax_errors"].extend(file_result["syntax_errors"])
                        
                    # Auto-patch if enabled and issues found
                    if (self.audit_config["patch_mode"] == "auto" and 
                        file_result.get("issues") and 
                        len(audit_results["patches_applied"]) < self.audit_config["max_patches_per_run"]):
                        
                        patch_result = await self._auto_patch_file(file_path, file_result["issues"])
                        if patch_result.get("patched"):
                            audit_results["patches_applied"].append(patch_result)
                    
                except Exception as e:
                    logger.error(f"Error auditing {file_path}: {str(e)}")
                    audit_results["critical_errors"].append({
                        "file": str(file_path),
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
            
            # Generate audit summary
            summary = await self._generate_audit_summary(audit_results)
            audit_results["summary"] = summary
            
            # Send Discord notifications
            await self._send_audit_notifications(audit_results)
            
            logger.info(f"Audit completed: {audit_results['files_scanned']} files, {len(audit_results['issues_found'])} issues")
            return audit_results
            
        except Exception as e:
            logger.error(f"Full audit failed: {str(e)}")
            await discord_notifier.send_critical_error(
                error=f"Full audit failed: {str(e)}",
                component="GeminiAuditAgent.run_full_audit",
                traceback=traceback.format_exc()
            )
            raise
    
    async def audit_specific_file(self, file_path: str) -> Dict[str, Any]:
        """Audit a specific file"""
        if not self.is_initialized:
            raise Exception("Audit agent not initialized")
            
        file_path = Path(file_path)
        logger.info(f"Auditing file: {file_path}")
        
        try:
            result = await self._audit_file(file_path)
            
            if result.get("issues"):
                await discord_notifier.send_audit_alert(
                    severity="medium",
                    component=str(file_path),
                    issue=f"Found {len(result['issues'])} issues",
                    details={"issues": [issue["type"] for issue in result["issues"]]}
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error auditing {file_path}: {str(e)}")
            await discord_notifier.send_critical_error(
                error=f"File audit failed: {str(e)}",
                component=f"audit:{file_path}",
                traceback=traceback.format_exc()
            )
            raise
    
    async def _audit_file(self, file_path: Path) -> Dict[str, Any]:
        """Audit individual file for issues"""
        result = {
            "file": str(file_path),
            "issues": [],
            "syntax_errors": [],
            "file_size": 0,
            "line_count": 0
        }
        
        try:
            if not file_path.exists() or file_path.stat().st_size > self.audit_config["max_file_size"]:
                return result
            
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            result["file_size"] = len(content)
            result["line_count"] = len(content.splitlines())
            
            # Check for syntax errors
            if file_path.suffix == '.py':
                syntax_errors = self._check_python_syntax(content, file_path)
                result["syntax_errors"] = syntax_errors
            
            # Pattern-based checks
            pattern_issues = self._check_patterns(content, file_path)
            result["issues"].extend(pattern_issues)
            
            # Gemini AI analysis for complex issues
            if content.strip():
                ai_issues = await self._gemini_code_analysis(content, file_path)
                result["issues"].extend(ai_issues)
            
            return result
            
        except Exception as e:
            logger.error(f"File audit error for {file_path}: {str(e)}")
            result["audit_error"] = str(e)
            return result
    
    def _check_python_syntax(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check Python file for syntax errors"""
        syntax_errors = []
        
        try:
            ast.parse(content)
        except SyntaxError as e:
            syntax_errors.append({
                "type": "syntax_error",
                "line": e.lineno,
                "column": e.offset,
                "message": e.msg,
                "severity": "critical"
            })
        except Exception as e:
            syntax_errors.append({
                "type": "parse_error", 
                "message": str(e),
                "severity": "high"
            })
        
        return syntax_errors
    
    def _check_patterns(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check for critical patterns in code"""
        issues = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            for pattern in self.critical_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    issue_type = self._classify_pattern_issue(pattern, match.group())
                    issues.append({
                        "type": issue_type,
                        "line": i,
                        "column": match.start(),
                        "content": line.strip(),
                        "pattern": pattern,
                        "severity": self._get_issue_severity(issue_type),
                        "suggestion": self._get_pattern_suggestion(issue_type)
                    })
        
        return issues
    
    def _classify_pattern_issue(self, pattern: str, match: str) -> str:
        """Classify the type of issue based on pattern"""
        if "key" in pattern or "password" in pattern or "secret" in pattern:
            return "hardcoded_secret"
        elif "sleep" in pattern:
            return "blocking_sleep"
        elif "random" in pattern:
            return "non_deterministic"
        elif "simulation" in pattern or "mock" in pattern:
            return "simulation_fallback"
        elif "todo" in pattern or "fixme" in pattern:
            return "unfinished_code"
        elif "print" in pattern or "console.log" in pattern:
            return "debug_statement"
        else:
            return "code_smell"
    
    def _get_issue_severity(self, issue_type: str) -> str:
        """Get severity level for issue type"""
        severity_map = {
            "hardcoded_secret": "critical",
            "syntax_error": "critical", 
            "simulation_fallback": "high",
            "blocking_sleep": "high",
            "non_deterministic": "medium",
            "debug_statement": "medium",
            "unfinished_code": "low",
            "code_smell": "low"
        }
        return severity_map.get(issue_type, "medium")
    
    def _get_pattern_suggestion(self, issue_type: str) -> str:
        """Get suggestion for fixing issue"""
        suggestions = {
            "hardcoded_secret": "Move to environment variable",
            "simulation_fallback": "Remove simulation fallback, fail gracefully",
            "blocking_sleep": "Replace with async sleep or remove",
            "non_deterministic": "Use deterministic alternatives or seed",
            "debug_statement": "Remove debug statements for production",
            "unfinished_code": "Complete implementation or remove",
            "code_smell": "Refactor for better maintainability"
        }
        return suggestions.get(issue_type, "Review and fix")
    
    async def _gemini_code_analysis(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Use Gemini AI to analyze code for complex issues"""
        try:
            analysis_prompt = f"""
            Analyze this {file_path.suffix} code for institutional trading system issues:
            
            File: {file_path}
            Code:
            ```
            {content[:8000]}  # Truncate for token limits
            ```
            
            Look for:
            1. Hardcoded secrets/API keys/passwords
            2. Simulation/mock fallbacks that should be removed
            3. Non-deterministic behavior (random, time-based)
            4. Poor error handling
            5. Resource leaks
            6. Security vulnerabilities
            7. Performance issues
            8. Logic errors
            
            Return JSON array of issues:
            [
                {{
                    "type": "issue_type",
                    "line": line_number,
                    "severity": "critical|high|medium|low", 
                    "message": "Description",
                    "suggestion": "How to fix"
                }}
            ]
            
            Only return the JSON array, no other text.
            """
            
            response = self.gemini_model.generate_content(analysis_prompt)
            
            # Parse JSON response
            try:
                json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
                if json_match:
                    issues = json.loads(json_match.group())
                    return issues if isinstance(issues, list) else []
            except json.JSONDecodeError:
                pass
            
            return []
            
        except Exception as e:
            logger.error(f"Gemini analysis failed for {file_path}: {str(e)}")
            return []
    
    async def _auto_patch_file(self, file_path: Path, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Automatically patch file issues using Gemini AI"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Only patch non-critical issues automatically
            patchable_issues = [
                issue for issue in issues 
                if issue.get("severity") in ["medium", "low"] and
                issue.get("type") in ["debug_statement", "unfinished_code", "code_smell"]
            ]
            
            if not patchable_issues:
                return {"patched": False, "reason": "No auto-patchable issues"}
            
            patch_prompt = f"""
            Fix these issues in the code:
            
            File: {file_path}
            Issues to fix: {json.dumps(patchable_issues, indent=2)}
            
            Original code:
            ```
            {content}
            ```
            
            Return the corrected code with fixes applied. Preserve all functionality.
            Only fix the specific issues listed. Do not make other changes.
            """
            
            response = self.gemini_model.generate_content(patch_prompt)
            
            # Extract code from response
            code_match = re.search(r'```(?:\w+)?\n(.*?)\n```', response.text, re.DOTALL)
            if code_match:
                patched_content = code_match.group(1)
                
                # Backup original file
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{int(datetime.now().timestamp())}")
                file_path.write_text(backup_path.read_text() if backup_path.exists() else content)
                
                # Apply patch
                file_path.write_text(patched_content, encoding='utf-8')
                
                # Verify patch doesn't break syntax
                if file_path.suffix == '.py':
                    try:
                        ast.parse(patched_content)
                    except SyntaxError:
                        # Revert if syntax is broken
                        file_path.write_text(content, encoding='utf-8')
                        return {"patched": False, "reason": "Patch broke syntax"}
                
                patch_result = {
                    "patched": True,
                    "file": str(file_path),
                    "issues_fixed": len(patchable_issues),
                    "backup_created": str(backup_path),
                    "patch_timestamp": datetime.now().isoformat()
                }
                
                # Notify Discord
                await discord_notifier.send_patch_notification(
                    action="Auto-patched code issues",
                    files_modified=[str(file_path)],
                    summary=f"Fixed {len(patchable_issues)} issues: {', '.join([i['type'] for i in patchable_issues])}"
                )
                
                return patch_result
            
            return {"patched": False, "reason": "Could not extract patched code"}
            
        except Exception as e:
            logger.error(f"Auto-patch failed for {file_path}: {str(e)}")
            await discord_notifier.send_critical_error(
                error=f"Auto-patch failed: {str(e)}",
                component=f"patch:{file_path}"
            )
            return {"patched": False, "reason": str(e)}
    
    def _get_files_to_scan(self) -> List[Path]:
        """Get list of files to scan"""
        files = []
        
        for pattern in self.file_patterns:
            files.extend(self.project_root.glob(f"**/{pattern}"))
        
        # Filter out ignored patterns
        filtered_files = []
        for file_path in files:
            if not any(ignore in str(file_path) for ignore in self.audit_config["ignore_patterns"]):
                filtered_files.append(file_path)
        
        return filtered_files
    
    async def _generate_audit_summary(self, audit_results: Dict[str, Any]) -> str:
        """Generate human-readable audit summary"""
        try:
            summary_prompt = f"""
            Generate a concise audit summary for this codebase scan:
            
            Files scanned: {audit_results['files_scanned']}
            Issues found: {len(audit_results['issues_found'])}
            Patches applied: {len(audit_results['patches_applied'])}
            Syntax errors: {len(audit_results['syntax_errors'])}
            
            Issue breakdown:
            {json.dumps(audit_results['issues_found'][:10], indent=2)}
            
            Provide:
            1. Overall health assessment
            2. Top 3 priority fixes needed
            3. Security concerns
            4. Production readiness status
            
            Keep it concise and actionable.
            """
            
            response = self.gemini_model.generate_content(summary_prompt)
            return response.text
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    async def _send_audit_notifications(self, audit_results: Dict[str, Any]):
        """Send audit results to Discord"""
        try:
            critical_issues = [i for i in audit_results['issues_found'] if i.get('severity') == 'critical']
            high_issues = [i for i in audit_results['issues_found'] if i.get('severity') == 'high']
            
            if critical_issues:
                await discord_notifier.send_audit_alert(
                    severity="critical",
                    component="Codebase Audit",
                    issue=f"Found {len(critical_issues)} critical issues requiring immediate attention",
                    details={
                        "critical_issues": len(critical_issues),
                        "files_affected": len(set(i.get('file', 'unknown') for i in critical_issues)),
                        "top_issues": [i.get('type') for i in critical_issues[:5]]
                    }
                )
            
            if high_issues:
                await discord_notifier.send_audit_alert(
                    severity="high", 
                    component="Codebase Audit",
                    issue=f"Found {len(high_issues)} high-priority issues",
                    details={"issue_types": list(set(i.get('type') for i in high_issues))}
                )
            
            # Send summary
            if audit_results.get('summary'):
                await discord_notifier.send_system_status({
                    "audit_complete": {
                        "status": "healthy" if not critical_issues else "issues",
                        "files_scanned": audit_results['files_scanned'],
                        "total_issues": len(audit_results['issues_found']),
                        "patches_applied": len(audit_results['patches_applied'])
                    }
                })
                
        except Exception as e:
            logger.error(f"Failed to send audit notifications: {str(e)}")

# Global audit agent instance
audit_agent = GeminiAuditAgent()
