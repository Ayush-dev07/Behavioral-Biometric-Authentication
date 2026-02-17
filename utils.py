"""
Utility functions for the authentication system.
"""

import hashlib
import secrets
import time
from functools import wraps
from typing import Callable, Dict, Any
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256.
    
    Args:
        password: Plain text password
    
    Returns:
        hashed: Hexadecimal hash string
    """
    return hashlib.sha256(password.encode()).hexdigest()


def generate_session_token() -> str:
    """
    Generate a secure random session token.
    
    Returns:
        token: 32-character hexadecimal token
    """
    return secrets.token_hex(16)


def validate_user_id(user_id: str) -> tuple[bool, str]:
    """
    Validate user ID format.
    
    Args:
        user_id: User identifier to validate
    
    Returns:
        valid: True if valid
        message: Error message if invalid
    """
    if not user_id:
        return False, "User ID cannot be empty"
    
    if len(user_id) < 3:
        return False, "User ID must be at least 3 characters"
    
    if len(user_id) > 50:
        return False, "User ID must be at most 50 characters"
    
    if not user_id.replace('_', '').replace('-', '').isalnum():
        return False, "User ID can only contain letters, numbers, hyphens, and underscores"
    
    return True, ""


def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password requirements.
    
    Args:
        password: Password to validate
    
    Returns:
        valid: True if valid
        message: Error message if invalid
    """
    if not password:
        return False, "Password cannot be empty"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    if len(password) > 128:
        return False, "Password must be at most 128 characters"
    
    return True, ""


class RateLimiter:
    """
    Simple in-memory rate limiter.
    Tracks requests per IP address.
    """
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, identifier: str) -> tuple[bool, int]:
        """
        Check if request is allowed.
        
        Args:
            identifier: Unique identifier (e.g., IP address)
        
        Returns:
            allowed: True if request is allowed
            retry_after: Seconds until next request allowed (0 if allowed)
        """
        now = time.time()
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                timestamp for timestamp in self.requests[identifier]
                if now - timestamp < self.window_seconds
            ]
        else:
            self.requests[identifier] = []
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            oldest = min(self.requests[identifier])
            retry_after = int(self.window_seconds - (now - oldest)) + 1
            return False, retry_after
        
        # Add current request
        self.requests[identifier].append(now)
        return True, 0


class LockoutManager:
    """
    Manages user account lockouts after failed attempts.
    """
    
    def __init__(self, max_attempts: int = 5, lockout_duration: int = 300):
        """
        Initialize lockout manager.
        
        Args:
            max_attempts: Max failed attempts before lockout
            lockout_duration: Lockout duration in seconds
        """
        self.max_attempts = max_attempts
        self.lockout_duration = lockout_duration
        self.failed_attempts: Dict[str, int] = {}
        self.lockout_until: Dict[str, float] = {}
    
    def record_failed_attempt(self, user_id: str) -> None:
        """Record a failed authentication attempt."""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = 0
        
        self.failed_attempts[user_id] += 1
        
        if self.failed_attempts[user_id] >= self.max_attempts:
            self.lockout_until[user_id] = time.time() + self.lockout_duration
            logger.warning(f"User {user_id} locked out due to failed attempts")
    
    def record_successful_attempt(self, user_id: str) -> None:
        """Record a successful authentication."""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
        if user_id in self.lockout_until:
            del self.lockout_until[user_id]
    
    def is_locked_out(self, user_id: str) -> tuple[bool, int]:
        """
        Check if user is locked out.
        
        Returns:
            locked: True if locked out
            time_remaining: Seconds remaining in lockout
        """
        if user_id not in self.lockout_until:
            return False, 0
        
        now = time.time()
        until = self.lockout_until[user_id]
        
        if now >= until:
            # Lockout expired
            del self.lockout_until[user_id]
            del self.failed_attempts[user_id]
            return False, 0
        
        return True, int(until - now)
    
    def get_failed_attempts(self, user_id: str) -> int:
        """Get number of failed attempts for user."""
        return self.failed_attempts.get(user_id, 0)


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
    
    Returns:
        wrapper: Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        logger.debug(f"{func.__name__} took {elapsed:.2f}ms")
        return result
    return wrapper


def log_api_call(endpoint: str, user_id: str = None, success: bool = True, 
                error: str = None) -> None:
    """
    Log API call for monitoring and debugging.
    
    Args:
        endpoint: API endpoint called
        user_id: User identifier (optional)
        success: Whether call was successful
        error: Error message if failed
    """
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'endpoint': endpoint,
        'user_id': user_id or 'anonymous',
        'success': success
    }
    
    if error:
        log_data['error'] = error
        logger.error(f"API Error: {log_data}")
    else:
        logger.info(f"API Call: {log_data}")


def sanitize_error_message(error: Exception, debug: bool = False) -> str:
    """
    Sanitize error message for user display.
    
    Args:
        error: Exception object
        debug: If True, include full error details
    
    Returns:
        message: Sanitized error message
    """
    if debug:
        return str(error)
    
    # Generic messages for production
    error_type = type(error).__name__
    
    generic_messages = {
        'ValueError': 'Invalid input data',
        'KeyError': 'Missing required data',
        'FileNotFoundError': 'Resource not found',
        'PermissionError': 'Access denied',
        'TypeError': 'Invalid data type'
    }
    
    return generic_messages.get(error_type, 'An error occurred')


class PerformanceMonitor:
    """
    Monitor API performance metrics.
    """
    
    def __init__(self):
        self.call_counts: Dict[str, int] = {}
        self.response_times: Dict[str, list] = {}
        self.error_counts: Dict[str, int] = {}
    
    def record_call(self, endpoint: str, response_time_ms: float, 
                   success: bool = True) -> None:
        """Record an API call."""
        # Count
        if endpoint not in self.call_counts:
            self.call_counts[endpoint] = 0
            self.response_times[endpoint] = []
            self.error_counts[endpoint] = 0
        
        self.call_counts[endpoint] += 1
        self.response_times[endpoint].append(response_time_ms)
        
        if not success:
            self.error_counts[endpoint] += 1
        
        # Keep only last 1000 response times
        if len(self.response_times[endpoint]) > 1000:
            self.response_times[endpoint] = self.response_times[endpoint][-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for endpoint in self.call_counts:
            times = self.response_times[endpoint]
            
            stats[endpoint] = {
                'total_calls': self.call_counts[endpoint],
                'errors': self.error_counts[endpoint],
                'error_rate': self.error_counts[endpoint] / max(self.call_counts[endpoint], 1),
                'avg_response_time_ms': sum(times) / len(times) if times else 0,
                'min_response_time_ms': min(times) if times else 0,
                'max_response_time_ms': max(times) if times else 0
            }
        
        return stats


# Global instances
rate_limiter = RateLimiter(max_requests=60, window_seconds=60)
lockout_manager = LockoutManager(max_attempts=5, lockout_duration=300)
performance_monitor = PerformanceMonitor()