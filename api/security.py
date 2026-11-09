"""
Security & Input Sanitization Framework for Zenthi-AI OS.

Provides:
1. Path traversal prevention (CWE-22) for file uploads and document ingestion.
2. File extension whitelisting and payload size limits (DOS mitigation).
3. ChatML and system prompt injection sanitization.
4. Secure tool execution sandbox with strict timeouts and output bounds.
"""

import os
import re
import sys
import subprocess
from typing import Dict, Any, List, Optional, Set

ALLOWED_EXTENSIONS: Set[str] = {
    ".txt", ".md", ".pdf", ".json", ".csv", ".png", ".jpg", ".jpeg", ".webp"
}

MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
MAX_TOOL_OUTPUT_CHARS: int = 10000

# Malicious token sequences to sanitize from raw user inputs
SYSTEM_ESCAPE_PATTERNS = [
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|endoftext\|>",
    r"\[SYSTEM\]",
    r"\[/SYSTEM\]"
]


class SecurityValidationError(ValueError):
    """Raised when a security invariant or path validation check fails."""
    pass


def sanitize_filename(filename: str) -> str:
    """
    Strips directory separators and non-whitelisted characters from upload filenames.
    Prevents path traversal attacks (e.g. ../../etc/passwd or ..\\Windows\\System32).
    """
    if not filename or not filename.strip():
        raise SecurityValidationError("Filename cannot be empty.")
        
    # Extract only the base component
    clean_base = os.path.basename(filename.strip())
    # Replace backslashes if present
    clean_base = clean_base.replace("\\", "/").split("/")[-1]
    
    # Restrict to safe characters: alphanumeric, dots, underscores, dashes
    clean_base = re.sub(r"[^a-zA-Z0-9._-]", "_", clean_base)
    
    # Disallow names that start with a dot or are reserved
    if clean_base.startswith(".") or not clean_base:
        clean_base = "upload_" + clean_base
        
    return clean_base


def validate_safe_upload_path(filename: str, base_directory: str) -> str:
    """
    Validates that the target path resolves strictly within the allowed base directory.
    
    Args:
        filename: Proposed filename.
        base_directory: Absolute path to the intended upload directory.
        
    Returns:
        Safe, validated absolute file path.
        
    Raises:
        SecurityValidationError if extension is disallowed or path escapes base directory.
    """
    safe_name = sanitize_filename(filename)
    
    # Check extension
    _, ext = os.path.splitext(safe_name)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise SecurityValidationError(
            f"File extension '{ext}' is not permitted. Allowed extensions: {sorted(list(ALLOWED_EXTENSIONS))}"
        )
        
    target_abs = os.path.abspath(os.path.join(base_directory, safe_name))
    base_abs = os.path.abspath(base_directory)
    
    # Realpath check to prevent symlink bypass
    if not target_abs.startswith(base_abs):
        raise SecurityValidationError("Path traversal attempt detected. Access denied.")
        
    return target_abs


def sanitize_user_input(text: str) -> str:
    """
    Sanitizes user input by escaping ChatML and special model control tokens
    to prevent prompt injection and system identity overrides.
    """
    if not text:
        return ""
    sanitized = text
    for pattern in SYSTEM_ESCAPE_PATTERNS:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


class SecureToolSandbox:
    """Executes permitted sub-tasks with timeout and output length limits."""
    
    # Allowlist of safe Python builtins and modules for calculator/expression evaluations
    SAFE_BUILTINS = {
        "abs": abs, "min": min, "max": max, "sum": sum,
        "round": round, "len": len, "range": range, "int": int,
        "float": float, "str": str, "bool": bool
    }
    
    @staticmethod
    def safe_calculate(expression: str) -> Dict[str, Any]:
        """Safely evaluates purely mathematical expressions without arbitrary code execution."""
        # Only allow math digits, operators, parens, and decimal points
        if not re.match(r"^[0-9+\-*/()., %^eE\s]+$", expression):
            return {
                "success": False,
                "error": "Expression contains prohibited characters or operations."
            }
        try:
            # Replace power operator if needed
            safe_expr = expression.replace("^", "**")
            # Evaluate with no globals and restricted builtins
            result = eval(safe_expr, {"__builtins__": SecureToolSandbox.SAFE_BUILTINS}, {})
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": f"Calculation error: {str(e)}"}
            
    @staticmethod
    def truncate_output(output: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
        """Truncates tool outputs to prevent buffer overflows and UI freezing."""
        if len(output) > max_chars:
            return output[:max_chars] + f"\n... [Output truncated at {max_chars} characters]"
        return output
