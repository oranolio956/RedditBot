#!/usr/bin/env python3
"""
Telegram Account Initialization and Warming Script
Safely initializes and warms up a Telegram account for natural engagement
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.telegram_account_manager import TelegramAccountManager
from app.services.telegram_safety_monitor import TelegramSafetyMonitor
from app.models.telegram_account import TelegramAccount
from app.config.telegram_config import get_config, get_safety_config
from app.database import get_session
from pyrogram import Client
from pyrogram.errors import FloodWait, SessionPasswordNeeded
import getpass
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelegramAccountInitializer:
    """Handles safe initialization and warming of Telegram accounts"""
    
    def __init__(self):
        self.config = get_config()
        self.safety_config = get_safety_config("conservative")
        self.account_manager = None
        self.safety_monitor = TelegramSafetyMonitor()
        
    async def create_new_account(self) -> TelegramAccount:
        """Create a new Telegram account in the database"""
        print(f"{Fore.CYAN}Creating new Telegram account...{Style.RESET_ALL}")
        
        # Get account details
        phone_number = input(f"{Fore.YELLOW}Enter phone number (with country code): {Style.RESET_ALL}")
        account_name = input(f"{Fore.YELLOW}Enter account name/alias: {Style.RESET_ALL}")
        
        # Create account in database
        async with get_session() as session:
            account = TelegramAccount(
                phone_number=phone_number,
                account_name=account_name,
                status="initializing",
                safety_level="conservative",
                health_score=100.0,
                risk_score=0.0,
                configuration={
                    "personality": "helpful",
                    "formality": 0.5,
                    "response_rate": 0.3,
                    "max_daily_messages": self.safety_config["max_messages_per_day"],
                    "max_daily_groups": self.safety_config["max_groups_per_day"],
                    "max_daily_dms": self.safety_config["max_dms_per_day"]
                }
            )
            session.add(account)
            await session.commit()
            await session.refresh(account)
            
        print(f"{Fore.GREEN}✓ Account created in database{Style.RESET_ALL}")
        return account
    
    async def authenticate_account(self, account: TelegramAccount) -> bool:
        """Authenticate with Telegram and create session"""
        print(f"\n{Fore.CYAN}Authenticating with Telegram...{Style.RESET_ALL}")
        
        session_path = Path(self.config.SESSION_DIRECTORY) / f"{account.id}.session"
        
        # Create Pyrogram client
        client = Client(
            str(session_path),
            api_id=self.config.API_ID,
            api_hash=self.config.API_HASH,
            phone_number=account.phone_number
        )
        
        try:
            await client.connect()
            
            # Send code request
            sent_code = await client.send_code(account.phone_number)
            print(f"{Fore.YELLOW}Code sent to {account.phone_number}{Style.RESET_ALL}")
            
            # Get verification code
            code = input(f"{Fore.YELLOW}Enter verification code: {Style.RESET_ALL}")
            
            try:
                # Sign in with code
                await client.sign_in(account.phone_number, sent_code.phone_code_hash, code)
                print(f"{Fore.GREEN}✓ Successfully authenticated{Style.RESET_ALL}")
                
            except SessionPasswordNeeded:
                # Two-factor authentication required
                password = getpass.getpass(f"{Fore.YELLOW}Enter 2FA password: {Style.RESET_ALL}")
                await client.check_password(password)
                print(f"{Fore.GREEN}✓ 2FA authentication successful{Style.RESET_ALL}")
            
            # Get account info
            me = await client.get_me()
            
            # Update account in database
            async with get_session() as session:
                account.username = me.username
                account.first_name = me.first_name
                account.last_name = me.last_name
                account.telegram_id = me.id
                account.session_string = await client.export_session_string()
                account.status = "warming"
                account.last_active = datetime.utcnow()
                
                session.add(account)
                await session.commit()
            
            await client.disconnect()
            return True
            
        except FloodWait as e:
            print(f"{Fore.RED}⚠ Flood wait: {e.x} seconds{Style.RESET_ALL}")
            return False
            
        except Exception as e:
            print(f"{Fore.RED}✗ Authentication failed: {e}{Style.RESET_ALL}")
            logger.error(f"Authentication error: {e}")
            return False
    
    async def warm_account(self, account: TelegramAccount):
        """Gradually warm up the account over several days"""
        print(f"\n{Fore.CYAN}Starting account warming protocol...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}This process will take 7 days to complete safely{Style.RESET_ALL}")
        
        warming_schedule = [
            {"day": 1, "messages": 5, "groups": 0, "dms": 1, "activities": ["update_profile", "read_messages"]},
            {"day": 2, "messages": 8, "groups": 0, "dms": 2, "activities": ["join_channel", "react_to_messages"]},
            {"day": 3, "messages": 12, "groups": 1, "dms": 2, "activities": ["send_media", "forward_message"]},
            {"day": 4, "messages": 15, "groups": 1, "dms": 3, "activities": ["participate_in_group"]},
            {"day": 5, "messages": 20, "groups": 1, "dms": 3, "activities": ["share_content"]},
            {"day": 6, "messages": 25, "groups": 2, "dms": 4, "activities": ["engage_naturally"]},
            {"day": 7, "messages": 30, "groups": 2, "dms": 5, "activities": ["full_engagement"]}
        ]
        
        # Initialize account manager
        self.account_manager = TelegramAccountManager(account.id)
        await self.account_manager.initialize()
        
        current_day = 1
        for schedule in warming_schedule:
            print(f"\n{Fore.CYAN}Day {schedule['day']} Warming Activities:{Style.RESET_ALL}")
            print(f"  • Messages: {schedule['messages']}")
            print(f"  • Groups to join: {schedule['groups']}")
            print(f"  • DMs: {schedule['dms']}")
            print(f"  • Activities: {', '.join(schedule['activities'])}")
            
            # Update account configuration for the day
            async with get_session() as session:
                account.configuration.update({
                    "max_daily_messages": schedule["messages"],
                    "max_daily_groups": schedule["groups"],
                    "max_daily_dms": schedule["dms"],
                    "warming_day": current_day
                })
                session.add(account)
                await session.commit()
            
            # Perform warming activities
            for activity in schedule["activities"]:
                await self.perform_warming_activity(account, activity)
                
                # Natural delay between activities
                await asyncio.sleep(300 + (300 * 0.5))  # 5-10 minutes
            
            print(f"{Fore.GREEN}✓ Day {current_day} warming complete{Style.RESET_ALL}")
            
            if current_day < 7:
                print(f"{Fore.YELLOW}Waiting 24 hours before next warming phase...{Style.RESET_ALL}")
                # In production, wait 24 hours
                # await asyncio.sleep(86400)
                # For testing, reduced wait
                await asyncio.sleep(60)
            
            current_day += 1
        
        # Mark account as ready
        async with get_session() as session:
            account.status = "active"
            account.warming_completed_at = datetime.utcnow()
            session.add(account)
            await session.commit()
        
        print(f"\n{Fore.GREEN}✓ Account warming complete! Account is ready for normal operations{Style.RESET_ALL}")
    
    async def perform_warming_activity(self, account: TelegramAccount, activity: str):
        """Perform specific warming activity"""
        print(f"  {Fore.CYAN}Performing: {activity}{Style.RESET_ALL}")
        
        if activity == "update_profile":
            # Update bio and profile picture
            bio_text = "🤖 AI Assistant | Here to help and learn | Powered by advanced AI"
            await self.account_manager.update_profile(bio=bio_text)
            
        elif activity == "read_messages":
            # Read some messages without responding
            await self.account_manager.read_recent_messages()
            
        elif activity == "join_channel":
            # Join a public channel
            channels = [
                "@telegram",  # Official Telegram channel
                "@durov",     # Pavel Durov's channel
            ]
            for channel in channels[:1]:  # Join only one initially
                try:
                    await self.account_manager.join_chat(channel)
                    await asyncio.sleep(30)
                except Exception as e:
                    logger.warning(f"Could not join {channel}: {e}")
        
        elif activity == "react_to_messages":
            # Add reactions to some messages
            await self.account_manager.add_reactions(limit=3)
            
        elif activity == "send_media":
            # Send a simple media message (emoji, sticker)
            await self.account_manager.send_emoji_message()
            
        elif activity == "participate_in_group":
            # Send a simple message in a group
            await self.account_manager.participate_naturally(conservative=True)
            
        elif activity == "engage_naturally":
            # More natural engagement
            await self.account_manager.engage_in_conversations(limit=5)
        
        print(f"    {Fore.GREEN}✓ {activity} completed{Style.RESET_ALL}")
    
    async def verify_account_health(self, account: TelegramAccount) -> Dict[str, Any]:
        """Verify account health and readiness"""
        print(f"\n{Fore.CYAN}Verifying account health...{Style.RESET_ALL}")
        
        health_check = {
            "account_id": account.id,
            "phone_number": account.phone_number,
            "status": account.status,
            "health_score": account.health_score,
            "risk_score": account.risk_score,
            "daily_limits": {
                "messages": account.daily_message_count,
                "groups": account.daily_group_joins,
                "dms": account.daily_dm_count
            },
            "safety_events": account.flood_wait_count + account.spam_warning_count,
            "ready_for_production": False
        }
        
        # Check readiness criteria
        if (account.status == "active" and 
            account.health_score >= 80 and 
            account.risk_score <= 30 and
            health_check["safety_events"] == 0):
            health_check["ready_for_production"] = True
            print(f"{Fore.GREEN}✓ Account is healthy and ready for production{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠ Account needs more warming or has issues{Style.RESET_ALL}")
        
        # Display health report
        print(f"\n{Fore.CYAN}Health Report:{Style.RESET_ALL}")
        print(f"  • Status: {health_check['status']}")
        print(f"  • Health Score: {health_check['health_score']:.1f}/100")
        print(f"  • Risk Score: {health_check['risk_score']:.1f}/100")
        print(f"  • Safety Events: {health_check['safety_events']}")
        print(f"  • Ready: {'Yes' if health_check['ready_for_production'] else 'No'}")
        
        return health_check
    
    async def run(self):
        """Main initialization flow"""
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Telegram Account Initialization & Warming{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Check for existing accounts
        async with get_session() as session:
            from sqlalchemy import select
            result = await session.execute(select(TelegramAccount))
            existing_accounts = result.scalars().all()
        
        if existing_accounts:
            print(f"\n{Fore.YELLOW}Found {len(existing_accounts)} existing account(s){Style.RESET_ALL}")
            for acc in existing_accounts:
                print(f"  • {acc.account_name} ({acc.phone_number}) - Status: {acc.status}")
            
            choice = input(f"\n{Fore.YELLOW}Create new account? (y/n): {Style.RESET_ALL}")
            if choice.lower() != 'y':
                # Select existing account
                account_id = input(f"{Fore.YELLOW}Enter account ID to warm: {Style.RESET_ALL}")
                account = next((a for a in existing_accounts if str(a.id) == account_id), None)
                if not account:
                    print(f"{Fore.RED}Account not found{Style.RESET_ALL}")
                    return
            else:
                account = None
        else:
            account = None
        
        # Create new account if needed
        if not account:
            account = await self.create_new_account()
            
            # Authenticate account
            if not await self.authenticate_account(account):
                print(f"{Fore.RED}Authentication failed. Please try again later.{Style.RESET_ALL}")
                return
        
        # Warm account
        if account.status in ["initializing", "warming"]:
            await self.warm_account(account)
        else:
            print(f"{Fore.YELLOW}Account already warmed. Status: {account.status}{Style.RESET_ALL}")
        
        # Verify health
        health_report = await self.verify_account_health(account)
        
        # Save health report
        report_path = Path("logs") / f"account_{account.id}_health.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(health_report, f, indent=2, default=str)
        
        print(f"\n{Fore.GREEN}Health report saved to: {report_path}{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Initialization complete!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


async def main():
    """Main entry point"""
    initializer = TelegramAccountInitializer()
    await initializer.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Initialization cancelled by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        logger.exception("Initialization failed")
        sys.exit(1)