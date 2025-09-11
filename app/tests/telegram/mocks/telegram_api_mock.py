"""
Telegram API Mock System

Comprehensive mocking system for Telegram API interactions:
- Pyrogram client mocking
- API response simulation
- Error condition simulation
- State management for testing
- Network latency simulation
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from unittest.mock import AsyncMock, Mock, MagicMock
from contextlib import asynccontextmanager
import logging

from pyrogram import types, errors
from pyrogram.errors import FloodWait, UserDeactivated, ChatWriteForbidden, SlowmodeWait


class TelegramAPIMock:
    """Comprehensive Telegram API mock system"""
    
    def __init__(self, 
                 simulate_network_delay: bool = True,
                 error_probability: float = 0.02,
                 flood_wait_probability: float = 0.01):
        self.simulate_network_delay = simulate_network_delay
        self.error_probability = error_probability
        self.flood_wait_probability = flood_wait_probability
        
        # State tracking
        self.sent_messages = []
        self.chat_members = {}
        self.user_data = {}
        self.chat_data = {}
        self.flood_wait_history = []
        self.rate_limit_state = {}
        
        # Configuration
        self.network_delay_range = (0.05, 0.3)  # 50-300ms
        self.flood_wait_durations = [10, 30, 60, 120, 300]
        
        self.logger = logging.getLogger(__name__)
    
    async def _simulate_network_delay(self):
        """Simulate network latency"""
        if self.simulate_network_delay:
            delay = random.uniform(*self.network_delay_range)
            await asyncio.sleep(delay)
    
    def _should_trigger_error(self) -> bool:
        """Determine if an error should be triggered"""
        return random.random() < self.error_probability
    
    def _should_trigger_flood_wait(self) -> bool:
        """Determine if flood wait should be triggered"""
        return random.random() < self.flood_wait_probability
    
    def _get_flood_wait_duration(self) -> int:
        """Get flood wait duration based on history"""
        recent_flood_waits = [
            fw for fw in self.flood_wait_history
            if fw["timestamp"] > datetime.utcnow() - timedelta(hours=1)
        ]
        
        # Increase duration with more recent flood waits
        base_duration = random.choice(self.flood_wait_durations)
        multiplier = min(1 + len(recent_flood_waits) * 0.5, 5.0)
        
        return int(base_duration * multiplier)
    
    def create_mock_client(self) -> Mock:
        """Create comprehensive mock Pyrogram client"""
        client = Mock()
        
        # Basic client methods
        client.start = AsyncMock()
        client.stop = AsyncMock()
        client.get_me = AsyncMock(return_value=self._create_mock_user(
            123456789, "TestBot", "testbot", is_bot=True
        ))
        
        # Message methods
        client.send_message = AsyncMock(side_effect=self._mock_send_message)
        client.edit_message_text = AsyncMock(side_effect=self._mock_edit_message)
        client.delete_message = AsyncMock(side_effect=self._mock_delete_message)
        client.forward_messages = AsyncMock(side_effect=self._mock_forward_messages)
        
        # Chat action methods
        client.send_chat_action = Mock(side_effect=self._mock_send_chat_action)
        
        # Chat and user methods
        client.get_chat = AsyncMock(side_effect=self._mock_get_chat)
        client.get_chat_member = AsyncMock(side_effect=self._mock_get_chat_member)
        client.join_chat = AsyncMock(side_effect=self._mock_join_chat)
        client.leave_chat = AsyncMock(side_effect=self._mock_leave_chat)
        
        # File methods
        client.send_photo = AsyncMock(side_effect=self._mock_send_photo)
        client.send_document = AsyncMock(side_effect=self._mock_send_document)
        client.download_media = AsyncMock(side_effect=self._mock_download_media)
        
        # Utility methods
        client.get_history = AsyncMock(side_effect=self._mock_get_history)
        client.search_messages = AsyncMock(side_effect=self._mock_search_messages)
        
        # Handler registration (mock)
        client.on_message = Mock(return_value=lambda func: func)
        client.on_callback_query = Mock(return_value=lambda func: func)
        
        return client
    
    async def _mock_send_message(self, 
                                chat_id: Union[int, str],
                                text: str,
                                reply_to_message_id: Optional[int] = None,
                                reply_markup=None,
                                **kwargs) -> types.Message:
        """Mock send message with error simulation"""
        await self._simulate_network_delay()
        
        # Check for flood wait
        if self._should_trigger_flood_wait():
            duration = self._get_flood_wait_duration()
            self.flood_wait_history.append({
                "timestamp": datetime.utcnow(),
                "duration": duration,
                "chat_id": chat_id
            })
            raise FloodWait(value=duration)
        
        # Check for other errors
        if self._should_trigger_error():
            error_type = random.choice([
                ChatWriteForbidden(),
                UserDeactivated(),
                errors.PeerIdInvalid()
            ])
            raise error_type
        
        # Create successful response
        message_id = len(self.sent_messages) + 1
        message = self._create_mock_message(
            message_id=message_id,
            text=text,
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id
        )
        
        self.sent_messages.append({
            "message": message,
            "timestamp": datetime.utcnow(),
            "chat_id": chat_id,
            "text": text
        })
        
        return message
    
    async def _mock_edit_message(self,
                                chat_id: Union[int, str],
                                message_id: int,
                                text: str,
                                reply_markup=None,
                                **kwargs) -> types.Message:
        """Mock edit message"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            if random.random() < 0.5:
                raise errors.MessageNotModified()
            else:
                raise errors.MessageIdInvalid()
        
        # Find and update message
        for sent_msg in self.sent_messages:
            if (sent_msg["message"].id == message_id and 
                sent_msg["chat_id"] == chat_id):
                sent_msg["message"].text = text
                sent_msg["timestamp"] = datetime.utcnow()
                return sent_msg["message"]
        
        raise errors.MessageIdInvalid()
    
    async def _mock_delete_message(self,
                                  chat_id: Union[int, str], 
                                  message_ids: Union[int, List[int]],
                                  **kwargs) -> bool:
        """Mock delete message"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            raise errors.MessageDeleteForbidden()
        
        # Remove from sent messages
        if isinstance(message_ids, int):
            message_ids = [message_ids]
        
        self.sent_messages = [
            msg for msg in self.sent_messages
            if not (msg["message"].id in message_ids and msg["chat_id"] == chat_id)
        ]
        
        return True
    
    async def _mock_forward_messages(self,
                                    chat_id: Union[int, str],
                                    from_chat_id: Union[int, str],
                                    message_ids: Union[int, List[int]],
                                    **kwargs) -> List[types.Message]:
        """Mock forward messages"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            raise ChatWriteForbidden()
        
        if isinstance(message_ids, int):
            message_ids = [message_ids]
        
        forwarded_messages = []
        for i, msg_id in enumerate(message_ids):
            new_message = self._create_mock_message(
                message_id=len(self.sent_messages) + i + 1,
                text=f"Forwarded message {msg_id}",
                chat_id=chat_id
            )
            forwarded_messages.append(new_message)
        
        return forwarded_messages
    
    @asynccontextmanager
    async def _mock_send_chat_action(self, chat_id: Union[int, str], action: str):
        """Mock send chat action context manager"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            raise ChatWriteForbidden()
        
        # Simulate typing duration based on action
        action_durations = {
            "typing": random.uniform(1, 5),
            "upload_photo": random.uniform(2, 8),
            "upload_document": random.uniform(3, 10)
        }
        
        duration = action_durations.get(action, 2)
        
        class ChatActionContext:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await asyncio.sleep(duration)
        
        yield ChatActionContext()
    
    async def _mock_get_chat(self, chat_id: Union[int, str]) -> types.Chat:
        """Mock get chat info"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            raise errors.PeerIdInvalid()
        
        # Return cached chat or create new one
        if chat_id not in self.chat_data:
            self.chat_data[chat_id] = self._create_mock_chat(chat_id)
        
        return self.chat_data[chat_id]
    
    async def _mock_get_chat_member(self,
                                   chat_id: Union[int, str],
                                   user_id: int) -> types.ChatMember:
        """Mock get chat member"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            raise errors.UserNotParticipant()
        
        key = f"{chat_id}:{user_id}"
        if key not in self.chat_members:
            self.chat_members[key] = self._create_mock_chat_member(user_id)
        
        return self.chat_members[key]
    
    async def _mock_join_chat(self, chat_id: Union[int, str]) -> types.Chat:
        """Mock join chat"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            error_type = random.choice([
                errors.InviteHashExpired(),
                errors.UserAlreadyParticipant(),
                errors.ChatAdminRequired()
            ])
            raise error_type
        
        chat = await self._mock_get_chat(chat_id)
        
        # Add bot to chat members
        bot_key = f"{chat_id}:123456789"  # Bot's user ID
        self.chat_members[bot_key] = self._create_mock_chat_member(
            123456789, status="member"
        )
        
        return chat
    
    async def _mock_leave_chat(self, chat_id: Union[int, str]) -> bool:
        """Mock leave chat"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            raise errors.PeerIdInvalid()
        
        # Remove bot from chat members
        bot_key = f"{chat_id}:123456789"
        if bot_key in self.chat_members:
            del self.chat_members[bot_key]
        
        return True
    
    async def _mock_send_photo(self,
                              chat_id: Union[int, str],
                              photo: Union[str, bytes],
                              caption: Optional[str] = None,
                              **kwargs) -> types.Message:
        """Mock send photo"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            raise ChatWriteForbidden()
        
        message_id = len(self.sent_messages) + 1
        message = self._create_mock_message(
            message_id=message_id,
            text=caption or "",
            chat_id=chat_id,
            has_photo=True
        )
        
        self.sent_messages.append({
            "message": message,
            "timestamp": datetime.utcnow(),
            "chat_id": chat_id,
            "type": "photo"
        })
        
        return message
    
    async def _mock_send_document(self,
                                 chat_id: Union[int, str],
                                 document: Union[str, bytes],
                                 caption: Optional[str] = None,
                                 **kwargs) -> types.Message:
        """Mock send document"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            raise ChatWriteForbidden()
        
        message_id = len(self.sent_messages) + 1
        message = self._create_mock_message(
            message_id=message_id,
            text=caption or "",
            chat_id=chat_id,
            has_document=True
        )
        
        return message
    
    async def _mock_download_media(self,
                                  message: types.Message,
                                  file_name: Optional[str] = None,
                                  **kwargs) -> str:
        """Mock download media"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            raise errors.FileIdInvalid()
        
        # Return mock file path
        return f"/tmp/mock_download_{message.id}_{file_name or 'file'}"
    
    async def _mock_get_history(self,
                               chat_id: Union[int, str],
                               limit: int = 100,
                               offset: int = 0,
                               **kwargs) -> List[types.Message]:
        """Mock get chat history"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            raise errors.PeerIdInvalid()
        
        # Return mock message history
        messages = []
        for i in range(min(limit, 20)):  # Return up to 20 mock messages
            message = self._create_mock_message(
                message_id=offset + i + 1,
                text=f"Mock history message {offset + i + 1}",
                chat_id=chat_id,
                date=datetime.utcnow() - timedelta(hours=i)
            )
            messages.append(message)
        
        return messages
    
    async def _mock_search_messages(self,
                                   chat_id: Union[int, str],
                                   query: str,
                                   limit: int = 100,
                                   **kwargs) -> List[types.Message]:
        """Mock search messages"""
        await self._simulate_network_delay()
        
        if self._should_trigger_error():
            raise errors.SearchQueryEmpty()
        
        # Return mock search results
        messages = []
        num_results = random.randint(0, min(limit, 10))
        
        for i in range(num_results):
            message = self._create_mock_message(
                message_id=i + 1,
                text=f"Search result for '{query}' - message {i + 1}",
                chat_id=chat_id
            )
            messages.append(message)
        
        return messages
    
    def _create_mock_message(self,
                            message_id: int,
                            text: str,
                            chat_id: Union[int, str],
                            date: Optional[datetime] = None,
                            reply_to_message_id: Optional[int] = None,
                            has_photo: bool = False,
                            has_document: bool = False) -> types.Message:
        """Create mock Telegram message"""
        message = Mock(spec=types.Message)
        
        message.id = message_id
        message.text = text
        message.date = date or datetime.utcnow()
        message.reply_to_message_id = reply_to_message_id
        
        # Chat information
        message.chat = self._create_mock_chat(chat_id)
        
        # From user (sender)
        message.from_user = self._create_mock_user(
            123456789, "TestBot", "testbot", is_bot=True
        )
        
        # Media flags
        message.photo = Mock() if has_photo else None
        message.document = Mock() if has_document else None
        
        # Reply to message
        if reply_to_message_id:
            message.reply_to_message = Mock(spec=types.Message)
            message.reply_to_message.id = reply_to_message_id
        else:
            message.reply_to_message = None
        
        return message
    
    def _create_mock_user(self,
                         user_id: int,
                         first_name: str,
                         username: Optional[str] = None,
                         last_name: Optional[str] = None,
                         is_bot: bool = False) -> types.User:
        """Create mock Telegram user"""
        user = Mock(spec=types.User)
        
        user.id = user_id
        user.is_bot = is_bot
        user.first_name = first_name
        user.last_name = last_name
        user.username = username
        user.language_code = "en"
        user.is_premium = random.choice([True, False]) if not is_bot else False
        
        return user
    
    def _create_mock_chat(self, chat_id: Union[int, str]) -> types.Chat:
        """Create mock Telegram chat"""
        chat = Mock(spec=types.Chat)
        
        chat.id = int(chat_id) if isinstance(chat_id, str) and chat_id.lstrip('-').isdigit() else chat_id
        
        if isinstance(chat_id, int) and chat_id < 0:
            # Group/supergroup
            chat.type = "supergroup" if chat_id < -1000000000000 else "group"
            chat.title = f"Test {chat.type.title()}"
            chat.username = f"test_chat_{abs(chat_id)}"
            chat.member_count = random.randint(10, 5000)
            chat.description = f"Test {chat.type} for testing purposes"
        else:
            # Private chat
            chat.type = "private"
            chat.first_name = "Test"
            chat.last_name = "User"
            chat.username = f"testuser_{abs(chat_id)}"
        
        return chat
    
    def _create_mock_chat_member(self,
                                user_id: int,
                                status: str = "member") -> types.ChatMember:
        """Create mock chat member"""
        member = Mock(spec=types.ChatMember)
        
        member.user = self._create_mock_user(
            user_id, f"User{user_id}", f"user{user_id}"
        )
        member.status = status
        member.until_date = None
        member.can_be_edited = False
        member.can_manage_chat = status in ["creator", "administrator"]
        member.can_delete_messages = status in ["creator", "administrator"]
        
        return member
    
    def get_sent_messages(self, chat_id: Optional[Union[int, str]] = None) -> List[Dict]:
        """Get sent messages for verification"""
        if chat_id is None:
            return self.sent_messages.copy()
        
        return [
            msg for msg in self.sent_messages
            if msg["chat_id"] == chat_id
        ]
    
    def get_flood_wait_history(self) -> List[Dict]:
        """Get flood wait history for analysis"""
        return self.flood_wait_history.copy()
    
    def clear_history(self):
        """Clear all tracked history"""
        self.sent_messages.clear()
        self.flood_wait_history.clear()
        self.chat_members.clear()
        self.user_data.clear()
        self.chat_data.clear()
        self.rate_limit_state.clear()
    
    def set_error_probability(self, probability: float):
        """Set error probability for testing"""
        self.error_probability = max(0.0, min(1.0, probability))
    
    def set_flood_wait_probability(self, probability: float):
        """Set flood wait probability for testing"""
        self.flood_wait_probability = max(0.0, min(1.0, probability))
    
    def enable_network_simulation(self, enabled: bool = True):
        """Enable/disable network delay simulation"""
        self.simulate_network_delay = enabled
    
    def inject_specific_error(self,
                             chat_id: Union[int, str],
                             error: Exception,
                             message_count: int = 1):
        """Inject specific error for next N messages to chat"""
        # This would be implemented to track specific errors per chat
        # For now, just raise the error immediately on next call
        pass


# Global mock instance for easy access in tests
telegram_mock = TelegramAPIMock()


def create_telegram_client_mock(**kwargs) -> Mock:
    """Convenience function to create telegram client mock"""
    return telegram_mock.create_mock_client(**kwargs)


def reset_telegram_mock():
    """Reset the global mock instance"""
    telegram_mock.clear_history()
    telegram_mock.set_error_probability(0.02)  # Reset to default
    telegram_mock.set_flood_wait_probability(0.01)  # Reset to default