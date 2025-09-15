"""
ARIA ELITE AI Accounts Manager
Multi-account management for various AI services
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import os
import hashlib
import sqlite3
from pathlib import Path

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

class AIProvider(Enum):
    """Supported AI providers"""
    CHATGPT = "chatgpt"
    GROK = "grok"
    GEMINI = "gemini"
    CLAUDE = "claude"
    PERPLEXITY = "perplexity"
    COPILOT = "copilot"

class AccountType(Enum):
    """Account type enumeration"""
    FREE = "free"
    PAID = "paid"
    ENTERPRISE = "enterprise"

class AccountStatus(Enum):
    """Account status enumeration"""
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    EXPIRED = "expired"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"

@dataclass
class AccountCredentials:
    """AI account credentials"""
    email: str
    password: str
    session_token: Optional[str] = None
    refresh_token: Optional[str] = None
    api_key: Optional[str] = None
    oauth_token: Optional[str] = None

@dataclass
class AccountUsage:
    """Account usage tracking"""
    daily_requests: int = 0
    monthly_requests: int = 0
    total_requests: int = 0
    last_request_time: Optional[datetime] = None
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    error_count: int = 0
    
    def can_make_request(self, max_daily: int, max_monthly: int) -> bool:
        """Check if account can make a request"""
        now = datetime.now()
        today = now.date()
        
        # Reset daily counter if it's a new day
        if self.last_request_time and self.last_request_time.date() != today:
            self.daily_requests = 0
        
        return (
            self.daily_requests < max_daily and
            self.monthly_requests < max_monthly
        )

@dataclass
class AIAccount:
    """Complete AI account information"""
    id: str
    provider: AIProvider
    credentials: AccountCredentials
    account_type: AccountType
    status: AccountStatus
    usage: AccountUsage
    max_daily_requests: int
    max_monthly_requests: int
    rate_limit_reset: Optional[datetime] = None
    priority: int = 1
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def update_usage(self, success: bool, response_time: float):
        """Update account usage statistics"""
        now = datetime.now()
        
        self.usage.total_requests += 1
        self.usage.last_request_time = now
        
        if success:
            self.usage.daily_requests += 1
            self.usage.monthly_requests += 1
            
            # Update success rate
            if self.usage.total_requests > 0:
                self.usage.success_rate = (
                    (self.usage.success_rate * (self.usage.total_requests - 1) + 1.0) /
                    self.usage.total_requests
                )
            
            # Update average response time
            if self.usage.total_requests > 0:
                self.usage.avg_response_time = (
                    (self.usage.avg_response_time * (self.usage.total_requests - 1) + response_time) /
                    self.usage.total_requests
                )
        else:
            self.usage.error_count += 1
        
        self.updated_at = now
    
    def is_available(self) -> bool:
        """Check if account is available for use"""
        now = datetime.now()
        
        # Check status
        if self.status not in [AccountStatus.ACTIVE, AccountStatus.RATE_LIMITED]:
            return False
        
        # Check rate limit reset
        if self.rate_limit_reset and self.rate_limit_reset > now:
            return False
        
        # Check usage limits
        if not self.usage.can_make_request(self.max_daily_requests, self.max_monthly_requests):
            return False
        
        return True

class AIAccountsManager:
    """
    Multi-account management system for AI services
    Handles account rotation, usage tracking, and credential management
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.accounts: Dict[str, AIAccount] = {}
        self.provider_accounts: Dict[AIProvider, List[str]] = {}
        self.account_pool: Dict[AIProvider, List[str]] = {}
        self.session_store = {}
        self.db_path = "ai_accounts.db"
        
        # Configuration
        self.rotation_strategy = "priority_based"  # priority_based, round_robin, load_balanced, random
        self.max_concurrent_sessions = 10
        self.session_timeout = 3600  # 1 hour
        self.auto_cleanup = True
        self.cleanup_interval = 300  # 5 minutes
        
        # Performance tracking
        self.stats = {
            'total_accounts': 0,
            'active_accounts': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'provider_stats': {provider.value: {
                'accounts': 0,
                'requests': 0,
                'success_rate': 0.0,
                'avg_time': 0.0
            } for provider in AIProvider}
        }
        
    async def initialize(self):
        """Initialize the AI accounts manager"""
        try:
            logger.info("Initializing AI Accounts Manager")
            
            # Create database
            await self._create_database()
            
            # Load existing accounts
            await self._load_accounts()
            
            # Start cleanup task
            if self.auto_cleanup:
                asyncio.create_task(self._cleanup_loop())
            
            self.is_initialized = True
            logger.info(f"AI Accounts Manager initialized with {len(self.accounts)} accounts")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Accounts Manager: {str(e)}")
            raise
    
    async def _create_database(self):
        """Create SQLite database for account storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create accounts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_accounts (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    email TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    max_daily_requests INTEGER DEFAULT 100,
                    max_monthly_requests INTEGER DEFAULT 3000,
                    priority INTEGER DEFAULT 1,
                    tags TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create usage table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS account_usage (
                    account_id TEXT PRIMARY KEY,
                    daily_requests INTEGER DEFAULT 0,
                    monthly_requests INTEGER DEFAULT 0,
                    total_requests INTEGER DEFAULT 0,
                    last_request_time TIMESTAMP,
                    success_rate REAL DEFAULT 0.0,
                    avg_response_time REAL DEFAULT 0.0,
                    error_count INTEGER DEFAULT 0,
                    FOREIGN KEY (account_id) REFERENCES ai_accounts (id)
                )
            ''')
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_sessions (
                    id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    session_token TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES ai_accounts (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to create database: {str(e)}")
            raise
    
    async def _load_accounts(self):
        """Load accounts from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load accounts
            cursor.execute("SELECT * FROM ai_accounts")
            for row in cursor.fetchall():
                account_id, provider, email, password_hash, account_type, status, \
                max_daily, max_monthly, priority, tags, metadata, created_at, updated_at = row
                
                # Load usage
                cursor.execute("SELECT * FROM account_usage WHERE account_id = ?", (account_id,))
                usage_row = cursor.fetchone()
                
                usage = AccountUsage()
                if usage_row:
                    _, daily, monthly, total, last_time, success_rate, avg_time, errors = usage_row
                    usage.daily_requests = daily
                    usage.monthly_requests = monthly
                    usage.total_requests = total
                    usage.last_request_time = datetime.fromisoformat(last_time) if last_time else None
                    usage.success_rate = success_rate
                    usage.avg_response_time = avg_time
                    usage.error_count = errors
                
                # Create account
                credentials = AccountCredentials(
                    email=email,
                    password=self._decrypt_password(password_hash)
                )
                
                account = AIAccount(
                    id=account_id,
                    provider=AIProvider(provider),
                    credentials=credentials,
                    account_type=AccountType(account_type),
                    status=AccountStatus(status),
                    usage=usage,
                    max_daily_requests=max_daily,
                    max_monthly_requests=max_monthly,
                    priority=priority,
                    tags=json.loads(tags) if tags else [],
                    metadata=json.loads(metadata) if metadata else {},
                    created_at=datetime.fromisoformat(created_at),
                    updated_at=datetime.fromisoformat(updated_at)
                )
                
                self.accounts[account_id] = account
                
                # Update provider mapping
                if provider not in self.provider_accounts:
                    self.provider_accounts[AIProvider(provider)] = []
                self.provider_accounts[AIProvider(provider)].append(account_id)
            
            conn.close()
            
            # Build account pool
            await self._build_account_pool()
            
        except Exception as e:
            logger.error(f"Failed to load accounts: {str(e)}")
            raise
    
    def _encrypt_password(self, password: str) -> str:
        """Encrypt password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _decrypt_password(self, encrypted: str) -> str:
        """Decrypt password (placeholder - in production use proper encryption)"""
        return "decrypted_password"  # Placeholder for demo
    
    async def add_account(self, provider: AIProvider, email: str, password: str, 
                         account_type: AccountType = AccountType.FREE,
                         max_daily_requests: int = 100,
                         max_monthly_requests: int = 3000,
                         priority: int = 1,
                         tags: List[str] = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Add new AI account"""
        try:
            # Generate account ID
            account_id = f"{provider.value}_{email}_{int(datetime.now().timestamp())}"
            
            # Create credentials
            credentials = AccountCredentials(
                email=email,
                password=password
            )
            
            # Create usage tracker
            usage = AccountUsage()
            
            # Create account
            account = AIAccount(
                id=account_id,
                provider=provider,
                credentials=credentials,
                account_type=account_type,
                status=AccountStatus.ACTIVE,
                usage=usage,
                max_daily_requests=max_daily_requests,
                max_monthly_requests=max_monthly_requests,
                priority=priority,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Save to database
            await self._save_account(account)
            
            # Update in-memory structures
            self.accounts[account_id] = account
            
            if provider not in self.provider_accounts:
                self.provider_accounts[provider] = []
            self.provider_accounts[provider].append(account_id)
            
            # Rebuild account pool
            await self._build_account_pool()
            
            logger.info(f"Added {provider.value} account: {email}")
            return account_id
            
        except Exception as e:
            logger.error(f"Failed to add account: {str(e)}")
            raise
    
    async def remove_account(self, account_id: str) -> bool:
        """Remove AI account"""
        try:
            if account_id not in self.accounts:
                return False
            
            # Remove from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM ai_accounts WHERE id = ?", (account_id,))
            cursor.execute("DELETE FROM account_usage WHERE account_id = ?", (account_id,))
            cursor.execute("DELETE FROM ai_sessions WHERE account_id = ?", (account_id,))
            
            conn.commit()
            conn.close()
            
            # Remove from memory
            account = self.accounts[account_id]
            del self.accounts[account_id]
            
            if account.provider in self.provider_accounts:
                self.provider_accounts[account.provider].remove(account_id)
            
            # Rebuild account pool
            await self._build_account_pool()
            
            logger.info(f"Removed account: {account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove account: {str(e)}")
            return False
    
    async def _save_account(self, account: AIAccount):
        """Save account to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Encrypt password
            password_hash = self._encrypt_password(account.credentials.password)
            
            # Save account
            cursor.execute('''
                INSERT OR REPLACE INTO ai_accounts 
                (id, provider, email, password_hash, account_type, status, 
                 max_daily_requests, max_monthly_requests, priority, tags, metadata, 
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                account.id,
                account.provider.value,
                account.credentials.email,
                password_hash,
                account.account_type.value,
                account.status.value,
                account.max_daily_requests,
                account.max_monthly_requests,
                account.priority,
                json.dumps(account.tags),
                json.dumps(account.metadata),
                account.created_at.isoformat(),
                account.updated_at.isoformat()
            ))
            
            # Save usage
            cursor.execute('''
                INSERT OR REPLACE INTO account_usage 
                (account_id, daily_requests, monthly_requests, total_requests, 
                 last_request_time, success_rate, avg_response_time, error_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                account.id,
                account.usage.daily_requests,
                account.usage.monthly_requests,
                account.usage.total_requests,
                account.usage.last_request_time.isoformat() if account.usage.last_request_time else None,
                account.usage.success_rate,
                account.usage.avg_response_time,
                account.usage.error_count
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save account: {str(e)}")
            raise
    
    async def _build_account_pool(self):
        """Build account pool for each provider"""
        self.account_pool = {}
        
        for provider, account_ids in self.provider_accounts.items():
            available_accounts = [
                account_id for account_id in account_ids
                if self.accounts[account_id].is_available()
            ]
            
            # Sort by priority (higher priority first)
            available_accounts.sort(
                key=lambda aid: self.accounts[aid].priority,
                reverse=True
            )
            
            self.account_pool[provider] = available_accounts
    
    async def get_best_account(self, provider: AIProvider) -> Optional[AIAccount]:
        """Get the best available account for a provider"""
        if provider not in self.account_pool or not self.account_pool[provider]:
            return None
        
        # Apply rotation strategy
        if self.rotation_strategy == "priority_based":
            # Use highest priority account
            best_account_id = self.account_pool[provider][0]
            return self.accounts[best_account_id]
        
        elif self.rotation_strategy == "round_robin":
            # Use first account, then rotate
            if not hasattr(self, '_round_robin_index'):
                self._round_robin_index = {}
            
            if provider not in self._round_robin_index:
                self._round_robin_index[provider] = 0
            
            account_id = self.account_pool[provider][self._round_robin_index[provider]]
            self._round_robin_index[provider] = (self._round_robin_index[provider] + 1) % len(self.account_pool[provider])
            
            return self.accounts[account_id]
        
        elif self.rotation_strategy == "load_balanced":
            # Use account with least usage
            best_account_id = min(
                self.account_pool[provider],
                key=lambda aid: self.accounts[aid].usage.total_requests
            )
            return self.accounts[best_account_id]
        
        elif self.rotation_strategy == "random":
            # Use random account
            import random
            account_id = random.choice(self.account_pool[provider])
            return self.accounts[account_id]
        
        else:
            # Default to first available account
            return self.accounts[self.account_pool[provider][0]]
    
    async def update_account_status(self, account_id: str, status: AccountStatus):
        """Update account status"""
        try:
            if account_id not in self.accounts:
                return False
            
            account = self.accounts[account_id]
            account.status = status
            account.updated_at = datetime.now()
            
            await self._save_account(account)
            await self._build_account_pool()
            
            logger.info(f"Updated {account_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update account status: {str(e)}")
            return False
    
    async def record_request(self, account_id: str, success: bool, response_time: float):
        """Record request for account"""
        try:
            if account_id not in self.accounts:
                return False
            
            account = self.accounts[account_id]
            account.update_usage(success, response_time)
            
            await self._save_account(account)
            
            # Update global stats
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1 if success else 0
            self.stats['failed_requests'] += 1 if not success else 0
            
            # Update provider stats
            provider = account.provider.value
            self.stats['provider_stats'][provider]['requests'] += 1
            self.stats['provider_stats'][provider]['success_rate'] = (
                self.stats['provider_stats'][provider]['success_rate'] * 0.9 + 
                (1.0 if success else 0.0) * 0.1
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record request: {str(e)}")
            return False
    
    async def _cleanup_loop(self):
        """Cleanup expired sessions and disabled accounts"""
        while self.auto_cleanup:
            try:
                now = datetime.now()
                
                # Clean up expired sessions
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM ai_sessions WHERE expires_at < ?", (now.isoformat(),))
                conn.commit()
                conn.close()
                
                # Check for accounts that need status updates
                for account_id, account in self.accounts.items():
                    if account.status == AccountStatus.RATE_LIMITED:
                        # Check if rate limit should be reset
                        if account.rate_limit_reset and account.rate_limit_reset < now:
                            await self.update_account_status(account_id, AccountStatus.ACTIVE)
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
                await asyncio.sleep(self.cleanup_interval)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        total_accounts = len(self.accounts)
        active_accounts = sum(1 for account in self.accounts.values() if account.status == AccountStatus.ACTIVE)
        
        return {
            "ai_accounts_manager": {
                "initialized": self.is_initialized,
                "total_accounts": total_accounts,
                "active_accounts": active_accounts,
                "rotation_strategy": self.rotation_strategy,
                "max_concurrent_sessions": self.max_concurrent_sessions,
                "stats": self.stats,
                "providers_status": {
                    provider.value: {
                        "total_accounts": len(self.provider_accounts.get(provider, [])),
                        "active_accounts": len([aid for aid in self.provider_accounts.get(provider, []) 
                                              if self.accounts[aid].status == AccountStatus.ACTIVE]),
                        "available_accounts": len(self.account_pool.get(provider, [])),
                        "accounts": [
                            {
                                "id": account_id,
                                "email": self.accounts[account_id].credentials.email,
                                "status": self.accounts[account_id].status.value,
                                "usage": {
                                    "daily_requests": self.accounts[account_id].usage.daily_requests,
                                    "total_requests": self.accounts[account_id].usage.total_requests,
                                    "success_rate": self.accounts[account_id].usage.success_rate
                                }
                            }
                            for account_id in self.provider_accounts.get(provider, [])
                        ]
                    }
                    for provider in AIProvider
                }
            }
        }
    
    async def import_accounts_from_file(self, file_path: str) -> int:
        """Import accounts from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            imported_count = 0
            
            for account_data in data.get('accounts', []):
                try:
                    account_id = await self.add_account(
                        provider=AIProvider(account_data['provider']),
                        email=account_data['email'],
                        password=account_data['password'],
                        account_type=AccountType(account_data.get('account_type', 'free')),
                        max_daily_requests=account_data.get('max_daily_requests', 100),
                        max_monthly_requests=account_data.get('max_monthly_requests', 3000),
                        priority=account_data.get('priority', 1),
                        tags=account_data.get('tags', []),
                        metadata=account_data.get('metadata', {})
                    )
                    imported_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to import account {account_data.get('email', 'unknown')}: {str(e)}")
            
            logger.info(f"Imported {imported_count} accounts from {file_path}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import accounts: {str(e)}")
            return 0
    
    async def export_accounts_to_file(self, file_path: str) -> bool:
        """Export accounts to JSON file"""
        try:
            accounts_data = []
            
            for account_id, account in self.accounts.items():
                accounts_data.append({
                    'id': account_id,
                    'provider': account.provider.value,
                    'email': account.credentials.email,
                    'account_type': account.account_type.value,
                    'status': account.status.value,
                    'max_daily_requests': account.max_daily_requests,
                    'max_monthly_requests': account.max_monthly_requests,
                    'priority': account.priority,
                    'tags': account.tags,
                    'metadata': account.metadata,
                    'usage': {
                        'daily_requests': account.usage.daily_requests,
                        'monthly_requests': account.usage.monthly_requests,
                        'total_requests': account.usage.total_requests,
                        'success_rate': account.usage.success_rate,
                        'avg_response_time': account.usage.avg_response_time,
                        'error_count': account.usage.error_count
                    },
                    'created_at': account.created_at.isoformat(),
                    'updated_at': account.updated_at.isoformat()
                })
            
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'total_accounts': len(accounts_data),
                'accounts': accounts_data
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(accounts_data)} accounts to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export accounts: {str(e)}")
            return False
    
    async def close(self):
        """Close the accounts manager"""
        try:
            # Save any remaining changes
            for account in self.accounts.values():
                await self._save_account(account)
            
            logger.info("AI Accounts Manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing AI Accounts Manager: {str(e)}")

# Global instance
ai_accounts_manager = AIAccountsManager()