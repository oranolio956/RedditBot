# Telegram Bot Setup Guide

## Creating Your Telegram Bot

### Step 1: Create Bot with BotFather

1. Open Telegram and search for `@BotFather`
2. Start a conversation and send `/newbot`
3. Choose a name for your bot (e.g., "AI Conversation Assistant")
4. Choose a username ending in `bot` (e.g., `ai_convo_assistant_bot`)
5. Save the token BotFather gives you (looks like: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 2: Configure Bot Settings

Send these commands to BotFather:

```
/setdescription
Intelligent AI assistant that learns and adapts to your conversation style. $150/week subscription for premium features.

/setabouttext
🤖 AI-Powered Conversation Bot
✨ Learns and adapts to your personality
💬 Natural, engaging conversations
🔒 Private and secure
💎 Premium features with subscription

/setcommands
start - Start conversation with the bot
subscribe - Subscribe to premium features ($150/week)
status - Check your subscription status
personality - View your personality profile
help - Get help and support
settings - Manage your preferences
cancel - Cancel current operation
feedback - Send feedback
privacy - View privacy policy
terms - View terms of service
```

### Step 3: Configure Bot Privacy

Send to BotFather:
```
/setprivacy
Disable - Bot needs to see all messages for conversation context
```

### Step 4: Enable Inline Mode (Optional)

```
/setinline
Enable inline queries for quick AI responses
```

### Step 5: Set Bot Profile Picture

```
/setuserpic
Upload a professional bot avatar
```

## Environment Configuration

### Create .env file

```bash
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_FROM_BOTFATHER
TELEGRAM_WEBHOOK_URL=https://yourdomain.com/webhook
TELEGRAM_WEBHOOK_SECRET=generate_random_secret_here
TELEGRAM_ADMIN_USER_IDS=123456789,987654321

# Bot Settings
BOT_USERNAME=your_bot_username
BOT_MAX_CONNECTIONS=100
BOT_UPDATES_WORKERS=4
BOT_USE_WEBHOOK=false  # Set to true for production

# Rate Limiting
TELEGRAM_RATE_LIMIT_MESSAGES_PER_MINUTE=20
TELEGRAM_RATE_LIMIT_BURST=5

# Session Settings
TELEGRAM_SESSION_TIMEOUT_MINUTES=30
TELEGRAM_MAX_SESSIONS_PER_USER=3

# Payment Integration
TELEGRAM_PAYMENTS_PROVIDER_TOKEN=  # From BotFather /payments
TELEGRAM_PAYMENTS_ENABLED=true

# Security
TELEGRAM_ENABLE_ANTI_SPAM=true
TELEGRAM_ENABLE_USER_VERIFICATION=true
TELEGRAM_BLOCK_FORWARDED_MESSAGES=false
```

## Production Webhook Setup

### Step 1: Generate Webhook Secret

```python
import secrets
webhook_secret = secrets.token_urlsafe(32)
print(f"TELEGRAM_WEBHOOK_SECRET={webhook_secret}")
```

### Step 2: Set Webhook URL

```python
import requests

BOT_TOKEN = "YOUR_BOT_TOKEN"
WEBHOOK_URL = "https://yourdomain.com/api/v1/telegram/webhook"

response = requests.post(
    f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook",
    json={
        "url": WEBHOOK_URL,
        "max_connections": 100,
        "allowed_updates": [
            "message",
            "edited_message",
            "callback_query",
            "inline_query",
            "my_chat_member",
            "chat_member",
            "pre_checkout_query",
            "successful_payment"
        ]
    }
)
print(response.json())
```

### Step 3: Verify Webhook

```python
response = requests.get(
    f"https://api.telegram.org/bot{BOT_TOKEN}/getWebhookInfo"
)
print(response.json())
```

## Security Best Practices

### 1. Token Security
- **NEVER** commit bot tokens to git
- Store tokens in environment variables
- Use secrets management in production (AWS Secrets Manager, Vault, etc.)
- Rotate tokens regularly

### 2. Webhook Security
- Always verify webhook signatures
- Use HTTPS only (Let's Encrypt for free SSL)
- Implement IP whitelisting for Telegram servers
- Add request size limits

### 3. Rate Limiting
- Implement per-user rate limits
- Add global rate limits
- Use exponential backoff for API calls
- Monitor for abuse patterns

### 4. Data Protection
- Encrypt sensitive user data
- Implement GDPR compliance
- Regular data backups
- Audit logs for all actions

## Telegram API Limits

Be aware of these limits:

- **Messages**: 30 messages/second to different users
- **Same chat**: 1 message/second to same chat
- **Bulk messages**: 20 messages/minute to same user
- **File size**: 50 MB for documents, 20 MB for photos
- **Inline results**: 50 results per query
- **Callback data**: 64 bytes max
- **Message text**: 4096 characters max

## Monitoring Setup

### Health Check Endpoint

```python
@app.get("/health/telegram")
async def telegram_health():
    try:
        # Check bot status
        bot_info = await bot.get_me()
        
        # Check webhook status
        webhook_info = await bot.get_webhook_info()
        
        return {
            "status": "healthy",
            "bot_username": bot_info.username,
            "webhook_url": webhook_info.url,
            "pending_updates": webhook_info.pending_update_count,
            "last_error": webhook_info.last_error_message
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Metrics to Track

1. **Message Metrics**
   - Messages received/sent per minute
   - Average response time
   - Error rate
   - User engagement rate

2. **Subscription Metrics**
   - New subscriptions
   - Active subscribers
   - Churn rate
   - Revenue per user

3. **Performance Metrics**
   - API response times
   - Webhook processing time
   - Database query times
   - Redis cache hit rate

## Testing Your Bot

### Local Development

```bash
# Using polling (for development)
python app/telegram/bot.py

# Using ngrok for webhook testing
ngrok http 8000
# Then set webhook to ngrok URL
```

### Test Commands

```python
# Test bot connection
curl https://api.telegram.org/bot<TOKEN>/getMe

# Test sending message
curl -X POST https://api.telegram.org/bot<TOKEN>/sendMessage \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "USER_CHAT_ID", "text": "Test message"}'

# Test webhook
curl -X POST https://yourdomain.com/api/v1/telegram/webhook \
  -H "Content-Type: application/json" \
  -H "X-Telegram-Bot-Api-Secret-Token: YOUR_SECRET" \
  -d '{"update_id": 1, "message": {...}}'
```

## Common Issues and Solutions

### Issue: Bot not responding
- Check token is correct
- Verify webhook URL is accessible
- Check SSL certificate is valid
- Review error logs

### Issue: Webhook not receiving updates
- Ensure webhook URL is HTTPS
- Check firewall rules
- Verify SSL certificate
- Check webhook info for errors

### Issue: Rate limiting
- Implement message queuing
- Add exponential backoff
- Use batch operations where possible
- Monitor rate limit headers

### Issue: High latency
- Use webhook instead of polling
- Optimize database queries
- Implement caching
- Use connection pooling

## Production Checklist

- [ ] Bot created with BotFather
- [ ] Token stored securely in environment
- [ ] Webhook URL configured with HTTPS
- [ ] SSL certificate valid and not self-signed
- [ ] Webhook secret configured
- [ ] Rate limiting implemented
- [ ] Error handling and logging
- [ ] Monitoring and alerting setup
- [ ] Database indexes optimized
- [ ] Redis caching configured
- [ ] Backup strategy in place
- [ ] GDPR compliance implemented
- [ ] Terms of service and privacy policy
- [ ] Admin commands protected
- [ ] Payment integration tested
- [ ] Load testing completed

## Support Resources

- [Telegram Bot API Documentation](https://core.telegram.org/bots/api)
- [aiogram Documentation](https://docs.aiogram.dev/)
- [Telegram Bot Best Practices](https://core.telegram.org/bots#best-practices)
- [Bot Support Group](https://t.me/BotTalk)