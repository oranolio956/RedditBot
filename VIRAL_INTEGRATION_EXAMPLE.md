# Viral Sharing Integration Example

This document shows how to integrate the viral sharing mechanics with your existing Telegram bot handlers to create seamless user experiences that drive organic growth.

## Overview

The viral sharing system consists of:

1. **Viral Engine** - Detects shareable moments and generates optimized content
2. **Referral Tracker** - Manages referral codes, rewards, and leaderboards  
3. **Viral Integration** - Seamlessly connects viral mechanics to bot conversations
4. **API Endpoints** - RESTful interface for web/mobile integration

## Integration Points

### 1. Message Handler Integration

```python
# In your existing message handler (app/telegram/handlers/message_handler.py)

from app.services.viral_integration import ViralIntegration
from app.database.session import get_db

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced message handler with viral detection."""
    
    # Your existing message processing...
    user_message = update.message.text
    conversation = get_or_create_conversation(update.effective_user.id)
    
    # Process message with bot
    bot_response = await process_with_llm(user_message, conversation)
    
    # Send bot response
    await update.message.reply_text(bot_response)
    
    # NEW: Check for viral opportunities
    db = next(get_db())
    viral_integration = ViralIntegration(db)
    
    # This will automatically suggest sharing if content is viral-worthy
    await viral_integration.process_conversation_message(
        conversation=conversation,
        message=create_message_object(user_message, bot_response),
        update=update,
        context=context
    )
```

### 2. Callback Query Handler for Sharing

```python
# Add to your callback query handlers

async def handle_sharing_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle viral sharing callback queries."""
    
    query = update.callback_query
    await query.answer()
    
    # Check if this is a sharing-related callback
    if query.data.startswith(('share_', 'customize_', 'preview_', 'view_')):
        db = next(get_db())
        viral_integration = ViralIntegration(db)
        user = get_user_from_telegram(update.effective_user)
        
        await viral_integration.handle_share_callback(
            query_data=query.data,
            user=user,
            update=update,
            context=context
        )
        return
    
    # Your existing callback handling...
```

### 3. Command Handlers for Viral Features

```python
# Add these new command handlers

async def referral_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /referral command to show user's referral stats."""
    
    db = next(get_db())
    referral_tracker = ReferralTracker(db)
    user = get_user_from_telegram(update.effective_user)
    
    # Generate referral content
    shareable_content = await referral_tracker.create_shareable_referral_content(
        user_id=user.id
    )
    
    # Create sharing keyboard
    keyboard = [
        [
            InlineKeyboardButton("📱 Share Link", url=f"tg://msg?text={shareable_content['share_url']}"),
            InlineKeyboardButton("📊 My Stats", callback_data="share_analytics")
        ],
        [
            InlineKeyboardButton("🏆 Leaderboard", callback_data="view_leaderboard"),
            InlineKeyboardButton("🎨 Custom Message", callback_data="customize_referral")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    referral_message = f"""
🚀 **Your Referral Program**

Share this bot with friends and earn $25 for each person who joins and stays active!

**Your Stats:**
• Referrals Sent: {shareable_content['user_stats']['current_referrals']}
• Your Rank: #{shareable_content['user_stats']['rank']}
• Achievements: {len(shareable_content['user_stats']['achievements'])}

**Your Personal Link:**
{shareable_content['share_url']}

Share now to start earning! 💰
    """.strip()
    
    await update.message.reply_text(
        referral_message,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /leaderboard command."""
    
    db = next(get_db())
    referral_tracker = ReferralTracker(db)
    leaderboard = await referral_tracker.get_referral_leaderboard(limit=10)
    
    leaderboard_text = "🏆 **Top Referrers**\n\n"
    
    for entry in leaderboard:
        rank_emoji = {"1": "🥇", "2": "🥈", "3": "🥉"}.get(str(entry.rank), f"{entry.rank}.")
        badges = "".join(entry.badges) if entry.badges else ""
        
        leaderboard_text += f"{rank_emoji} **{entry.display_name}** {badges}\n"
        leaderboard_text += f"   {entry.conversion_count} successful referrals\n\n"
    
    keyboard = [
        [
            InlineKeyboardButton("📈 My Stats", callback_data="share_analytics"),
            InlineKeyboardButton("🎯 Get Referral Link", callback_data="generate_referral")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        leaderboard_text,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def viral_content_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /viral command to show trending content."""
    
    db = next(get_db())
    llm_service = LLMService()
    viral_engine = ViralEngine(db, llm_service)
    
    trending_content = await viral_engine.get_trending_content(limit=5)
    
    if not trending_content:
        await update.message.reply_text("No viral content available right now. Keep chatting to create some! ✨")
        return
    
    content_text = "🔥 **Trending Content**\n\n"
    
    for i, content in enumerate(trending_content, 1):
        content_text += f"{i}. **{content.title}**\n"
        content_text += f"   Viral Score: {content.viral_score:.0f}/100 • {content.share_count} shares\n\n"
    
    keyboard = [
        [InlineKeyboardButton(f"Share #{i+1}", callback_data=f"share_trending:{content.id}") 
         for i, content in enumerate(trending_content[:3])]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        content_text,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
```

### 4. Background Tasks Integration

```python
# Add to your background task scheduler (if using celery/apscheduler)

from app.services.viral_integration import ViralIntegration

async def daily_viral_tasks():
    """Daily maintenance tasks for viral system."""
    
    db = next(get_db())
    viral_integration = ViralIntegration(db)
    
    # Check for referral conversions
    await viral_integration.check_referral_conversions()
    
    # Generate daily metrics
    await viral_integration.generate_daily_viral_metrics()
    
    print("Daily viral tasks completed")

# Schedule this to run daily
scheduler.add_job(daily_viral_tasks, 'cron', hour=2, minute=0)  # 2 AM daily
```

### 5. Welcome Message Enhancement

```python
# Enhance your /start command to handle referrals

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced start command with referral tracking."""
    
    # Check if user came from referral
    referral_code = None
    if context.args:
        referral_code = context.args[0]  # /start REFERRAL_CODE
    
    user = get_or_create_user(update.effective_user)
    
    # Process referral signup if applicable
    if referral_code:
        db = next(get_db())
        referral_tracker = ReferralTracker(db)
        
        referral = await referral_tracker.process_referral_signup(
            referral_code=referral_code,
            new_user=user
        )
        
        if referral:
            welcome_message = f"""
🎉 **Welcome! You were invited by {referral.referrer.get_display_name()}**

You've received a welcome bonus: {referral.program.referee_reward['description']}

I'm your AI companion, ready to have meaningful conversations and help with personal growth!
            """.strip()
        else:
            welcome_message = "🤖 **Welcome!** I'm your AI companion ready to chat!"
    else:
        welcome_message = """
🤖 **Welcome!** I'm your AI companion ready for meaningful conversations.

💡 **Tip:** Share me with friends using /referral and earn $25 for each active referral!
        """.strip()
    
    keyboard = [
        [
            InlineKeyboardButton("💬 Start Chatting", callback_data="start_chat"),
            InlineKeyboardButton("🎯 Get Referral Link", callback_data="generate_referral")
        ],
        [
            InlineKeyboardButton("ℹ️ How It Works", callback_data="show_help"),
            InlineKeyboardButton("🏆 Leaderboard", callback_data="view_leaderboard")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        welcome_message,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
```

## Web Integration Points

### 1. Shareable Content Landing Page

Create a web page at `https://yourbot.com/share/{content_id}` that:

- Displays the viral content in an engaging format
- Shows social sharing buttons optimized for each platform
- Includes the referrer's referral code in all sharing links
- Tracks views and social shares for analytics

### 2. Referral Landing Page

Create `https://yourbot.com/join?ref={referral_code}` that:

- Explains the bot's value proposition
- Shows social proof (testimonials, user count, etc.)
- Has clear call-to-action to start the bot
- Tracks referral clicks and conversions

### 3. Analytics Dashboard

Create a dashboard at `https://yourbot.com/analytics` for:

- Real-time viral metrics
- Referral performance tracking
- Content trending analysis
- User growth attribution

## Viral Content Examples

### 1. Funny AI Response

```json
{
  "content_type": "funny_moment",
  "title": "When AI Gets Too Real 😅",
  "description": "This AI's response was surprisingly human-like and hilarious",
  "content_data": {
    "conversation_excerpt": "Human: I'm having an existential crisis.\nAI: Have you tried turning your existence off and on again?",
    "context": "Discussing life problems"
  },
  "hashtags": ["#AI", "#Funny", "#ExistentialCrisis", "#TechHumor"],
  "optimal_platforms": ["twitter", "reddit", "tiktok"]
}
```

### 2. Personality Insight Card

```json
{
  "content_type": "personality_insight",
  "title": "Your Hidden Strength: Analytical Empathy",
  "description": "You combine logical thinking with deep emotional understanding",
  "content_data": {
    "insight": "People with your personality type excel at solving emotional problems through systematic approaches",
    "strengths": ["Problem-solving", "Emotional intelligence", "Pattern recognition"],
    "famous_examples": ["Therapists", "Researchers", "Counselors"]
  },
  "hashtags": ["#PersonalityInsight", "#SelfDiscovery", "#StrengthsFinder"],
  "optimal_platforms": ["instagram", "linkedin", "facebook"]
}
```

### 3. Wisdom Quote

```json
{
  "content_type": "wisdom_quote",
  "title": "Growth Happens in the Uncomfortable Spaces",
  "description": "A profound realization from an AI conversation about personal development",
  "content_data": {
    "quote": "The magic isn't in avoiding difficult emotions—it's in learning to dance with them while moving toward what matters to you.",
    "context": "Discussion about anxiety and personal growth",
    "visual_style": "minimalist_card"
  },
  "hashtags": ["#Wisdom", "#PersonalGrowth", "#Mindfulness", "#LifeLessons"],
  "optimal_platforms": ["instagram", "linkedin", "pinterest"]
}
```

## Analytics and Tracking

### Key Metrics to Track

1. **Viral Coefficient**: New users per existing user
2. **Content Performance**: Views, shares, engagement per content type
3. **Referral Conversion**: Click → Signup → Active user rates
4. **Platform Performance**: Which social platforms drive most growth
5. **Content Timing**: When viral content is most likely to be shared

### Automated Reporting

Set up daily/weekly reports that include:

- Viral coefficient trends
- Top performing content
- Referral leaderboard changes
- Platform performance comparison
- User growth attribution

## Best Practices

### 1. Content Quality Over Quantity

- Only suggest sharing for content with >65 viral score
- Auto-publish only content with >80 viral score
- Maintain high anonymization standards

### 2. User Experience First

- Never interrupt conversations with sharing suggestions
- Provide clear opt-out mechanisms
- Respect notification preferences

### 3. Reward Structure

- Make rewards meaningful but not excessive
- Implement anti-gaming measures
- Provide non-monetary rewards (badges, recognition)

### 4. Privacy Protection

- Always anonymize shared content
- Get explicit consent for public sharing
- Allow users to delete their shared content

## Testing the Integration

### 1. Create Test Conversations

Generate conversations that should trigger viral content:

```python
# Test conversation that should generate funny content
test_messages = [
    "I'm stressed about work",
    "Have you tried the ancient technique of screaming into a pillow? It's surprisingly effective and much cheaper than therapy. Plus, your pillow never judges you for your life choices. 😄"
]
```

### 2. Verify Viral Detection

Check that the viral engine correctly identifies shareable moments:

```python
# Should detect humor indicators and high sentiment
assert viral_score > 75
assert "humor" in viral_elements
```

### 3. Test Referral Flow

Simulate the complete referral journey:

1. Generate referral code
2. Click referral link
3. Start bot with referral
4. Become active user
5. Verify rewards distributed

This integration creates a seamless viral growth engine that operates naturally within conversations while providing powerful growth mechanics and analytics.