# Telegram Account Management System - Operations Guide

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Telegram API credentials (api_id and api_hash from https://my.telegram.org)

### Initial Setup

1. **Clone and Install Dependencies**
```bash
# Install Python dependencies
pip install -r requirements.telegram.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials:
# TELEGRAM_API_ID=your_api_id
# TELEGRAM_API_HASH=your_api_hash
# TELEGRAM_SESSION_KEY=your_encryption_key
```

2. **Initialize Database**
```bash
# Run database migrations
alembic upgrade head

# Verify database
python -c "from app.database import test_connection; test_connection()"
```

3. **Initialize First Account**
```bash
# Run interactive account setup
python scripts/telegram_init.py

# Follow prompts:
# - Enter phone number with country code
# - Enter verification code from Telegram
# - Complete 7-day warming protocol
```

## 📋 Daily Operations Checklist

### Morning Routine (9 AM)
- [ ] Check account health dashboard
- [ ] Review overnight safety events
- [ ] Verify all accounts are active
- [ ] Check message limits usage
- [ ] Review any critical alerts

### Midday Check (12 PM)
- [ ] Monitor engagement metrics
- [ ] Check response rates
- [ ] Review community performance
- [ ] Adjust engagement strategies if needed

### Evening Review (6 PM)
- [ ] Review daily analytics
- [ ] Check for any safety warnings
- [ ] Plan next day's activities
- [ ] Export daily report

## 🔍 Monitoring & Health Checks

### Real-Time Monitoring Dashboard
```bash
# Access monitoring dashboard
http://localhost:8001/docs#/Monitoring

# Key endpoints:
GET /monitoring/health/{account_id} - Account health metrics
GET /monitoring/safety/{account_id} - Safety analysis
GET /monitoring/engagement/{account_id} - Engagement analytics
WS /monitoring/ws/{account_id} - Real-time WebSocket updates
```

### Health Indicators

#### 🟢 Healthy Account (Safe to operate)
- Health Score: 80-100
- Risk Score: 0-30
- Daily Limits: <50% used
- No safety events in last 24h
- Status: "active"

#### 🟡 Warning State (Reduce activity)
- Health Score: 50-79
- Risk Score: 31-70
- Daily Limits: 50-80% used
- 1-2 safety events in last 24h
- Action: Reduce message frequency

#### 🔴 Critical State (Stop immediately)
- Health Score: <50
- Risk Score: >70
- Daily Limits: >80% used
- Multiple safety events
- Action: Pause all activity for 24h

## 🛡️ Safety Protocols

### Flood Wait Handling
When encountering FloodWait errors:
1. **Immediate**: Stop all activity
2. **Wait**: Full duration + 10%
3. **Resume**: At 50% normal rate
4. **Monitor**: For 2 hours
5. **Normal**: Return to normal if stable

### Spam Detection Response
If spam warning received:
1. **Stop**: Halt all messaging
2. **Review**: Check last 10 messages
3. **Adjust**: Modify message patterns
4. **Test**: Send 1 test message
5. **Resume**: Gradually increase activity

### Emergency Stop Procedure
```bash
# Emergency stop single account
curl -X POST http://localhost:8001/api/v1/telegram/accounts/{account_id}/emergency_stop

# Stop all accounts
curl -X POST http://localhost:8001/api/v1/telegram/emergency/stop_all

# Check status
curl http://localhost:8001/api/v1/telegram/accounts/{account_id}/status
```

## 🤝 Community Management

### Joining New Communities

#### Pre-Join Checklist
- [ ] Community has >100 members
- [ ] Not a spam/scam group
- [ ] Relevant to account persona
- [ ] Check group rules
- [ ] Verify no bot restrictions

#### Join Protocol
1. **Search**: Find community via search or link
2. **Observe**: Read last 50 messages
3. **Join**: Use natural timing (not instant)
4. **Lurk**: 24-48 hours before first message
5. **Engage**: Start with reactions, then messages

### Engagement Strategies

#### Lurker Phase (Days 1-3)
- Read messages only
- No interactions
- Learn community culture
- Identify key members

#### Participant Phase (Days 4-10)
- React to 2-3 messages/day
- Send 1-2 relevant messages
- Reply to direct questions
- Share helpful content

#### Contributor Phase (Days 11-30)
- 3-5 messages/day
- Start conversations
- Help other members
- Share valuable insights

#### Leader Phase (Day 30+)
- Regular valuable contributions
- Recognized by community
- Natural influence
- Maintain reputation

## 📊 Performance Optimization

### Message Timing Optimization
```python
# Optimal message timing patterns
PEAK_HOURS = [9, 12, 15, 18, 21]  # Local time
NORMAL_HOURS = [10, 11, 13, 14, 16, 17, 19, 20]
LOW_HOURS = [7, 8, 22, 23]
INACTIVE_HOURS = [0, 1, 2, 3, 4, 5, 6]

# Response delays (seconds)
IMMEDIATE: 2-10 (10% of responses)
QUICK: 10-60 (40% of responses)
NORMAL: 60-300 (40% of responses)
DELAYED: 300-1800 (10% of responses)
```

### Natural Behavior Patterns
- Typing speed: 8-25 characters/second
- Read time: 200-400 words/minute
- Thinking pause: 3-10 seconds before typing
- Error rate: 1-2% typos (auto-corrected)
- Activity variation: ±30% daily

## 🔧 Troubleshooting Guide

### Common Issues & Solutions

#### Account Won't Authenticate
```bash
# Check API credentials
echo $TELEGRAM_API_ID
echo $TELEGRAM_API_HASH

# Verify phone number format
# Correct: +1234567890
# Wrong: 1234567890, (123) 456-7890

# Clear session and retry
rm sessions/{account_id}.session
python scripts/telegram_init.py
```

#### High Risk Score
```bash
# Check recent events
curl http://localhost:8001/monitoring/safety/{account_id}?days=1

# Common causes:
- Too many messages too quickly
- Joining groups too fast
- Sending similar messages
- Operating during inactive hours

# Recovery:
1. Pause for 24 hours
2. Resume at 25% capacity
3. Gradually increase over 3 days
```

#### Session Expired
```bash
# Re-authenticate
python scripts/telegram_reauth.py --account-id {account_id}

# If persistent:
1. Check 2FA settings
2. Verify phone number active
3. Check for account restrictions
```

## 📈 Analytics & Reporting

### Daily Report Generation
```bash
# Generate daily report
python scripts/generate_report.py --date today

# Custom date range
python scripts/generate_report.py --start 2024-01-01 --end 2024-01-31

# Export formats
--format json  # JSON data
--format csv   # CSV spreadsheet
--format pdf   # PDF report
```

### Key Metrics to Track

#### Engagement Metrics
- Messages sent/received ratio
- Response rate percentage
- Average response time
- Conversation duration
- Community reputation scores

#### Safety Metrics
- Daily risk score trend
- Safety events per day
- Flood wait occurrences
- Error rate percentage
- Recovery time from issues

#### Growth Metrics
- Communities joined
- Connections made
- Influence score
- Content engagement rate
- Network expansion rate

## 🚨 Incident Response

### Incident Severity Levels

#### Level 1: Minor (Self-resolving)
- Single FloodWait (<60 seconds)
- Temporary network issues
- Single failed message
- **Action**: Log and monitor

#### Level 2: Moderate (Attention needed)
- Multiple FloodWaits
- Risk score >50
- Community ban/mute
- **Action**: Adjust parameters, monitor closely

#### Level 3: Major (Immediate action)
- Account restricted
- Risk score >70
- Multiple community bans
- **Action**: Stop activity, investigate, plan recovery

#### Level 4: Critical (Emergency)
- Account banned
- Phone number blocked
- Legal/compliance issue
- **Action**: Full stop, escalate, document everything

### Incident Response Procedure
1. **Identify**: Detect via monitoring/alerts
2. **Assess**: Determine severity level
3. **Contain**: Stop affected operations
4. **Investigate**: Find root cause
5. **Remediate**: Fix the issue
6. **Recover**: Restore operations
7. **Review**: Post-incident analysis
8. **Improve**: Update procedures

## 🔄 Maintenance Procedures

### Daily Maintenance
```bash
# Clean up old logs (>7 days)
find logs/ -name "*.log" -mtime +7 -delete

# Vacuum database
psql telegram_accounts -c "VACUUM ANALYZE;"

# Clear Redis cache
redis-cli FLUSHDB

# Check disk space
df -h
```

### Weekly Maintenance
```bash
# Backup database
pg_dump telegram_accounts > backups/telegram_$(date +%Y%m%d).sql

# Update dependencies
pip install --upgrade -r requirements.telegram.txt

# Security scan
safety check
bandit -r app/

# Performance profiling
python -m cProfile scripts/performance_check.py
```

### Monthly Maintenance
- Full system audit
- Security review
- Performance optimization
- Strategy evaluation
- Documentation update

## 📱 Mobile Monitoring (Coming Soon)

### Planned Mobile Features
- iOS/Android monitoring app
- Push notifications for alerts
- Remote emergency stop
- Real-time metrics dashboard
- Quick action buttons

## 🔐 Security Best Practices

### Account Security
- Use strong, unique passwords
- Enable 2FA on all accounts
- Rotate session keys monthly
- Never share API credentials
- Use encrypted storage for sessions

### Operational Security
- Monitor from secure network
- Use VPN for sensitive operations
- Log all administrative actions
- Regular security audits
- Incident response plan ready

### Data Security
- Encrypt sensitive data at rest
- Use TLS for all connections
- Regular backups (3-2-1 rule)
- GDPR compliance for EU data
- Data retention policies

## 📞 Support & Escalation

### Support Tiers

#### Tier 1: Self-Service
- Check this documentation
- Review error logs
- Try troubleshooting steps
- Check monitoring dashboard

#### Tier 2: Technical Support
- Slack: #telegram-support
- Email: telegram-support@company.com
- Response time: 2-4 hours

#### Tier 3: Emergency Support
- PagerDuty: telegram-oncall
- Phone: +1-XXX-XXX-XXXX
- Response time: 15 minutes

## 🎯 Success Metrics

### Operational Excellence
- 99.9% uptime target
- <5% daily limit usage
- Zero critical incidents/month
- <2% message failure rate
- 100% compliance adherence

### Engagement Excellence
- 70%+ community reputation
- 30%+ response rate
- <60s average response time
- 5+ active communities
- Natural conversation flow

## 📚 Additional Resources

### Documentation
- [API Documentation](/docs)
- [Database Schema](/docs/database.md)
- [Security Guide](/docs/security.md)
- [Development Guide](/docs/development.md)

### Tools & Utilities
- [Telegram API Explorer](https://core.telegram.org/methods)
- [Pyrogram Documentation](https://docs.pyrogram.org)
- [Community Guidelines](https://telegram.org/blog/community-guidelines)

### Training Materials
- Video: Account Management Best Practices
- Workshop: Natural Engagement Techniques
- Course: Telegram API Fundamentals
- Certification: Telegram Operations Specialist

## ✅ Pre-Launch Checklist

Before launching in production:

### Technical Requirements
- [ ] All dependencies installed
- [ ] Database migrations complete
- [ ] Redis cache configured
- [ ] SSL certificates installed
- [ ] Monitoring alerts configured

### Account Preparation
- [ ] Account authenticated
- [ ] 7-day warming complete
- [ ] Bio and profile updated
- [ ] AI disclosure added
- [ ] Initial communities identified

### Safety Measures
- [ ] Rate limits configured
- [ ] Circuit breakers tested
- [ ] Emergency stop verified
- [ ] Backup procedures ready
- [ ] Incident response plan

### Monitoring Setup
- [ ] Dashboard accessible
- [ ] Alerts configured
- [ ] Logs aggregated
- [ ] Metrics collected
- [ ] Reports automated

### Team Readiness
- [ ] Operations team trained
- [ ] On-call schedule set
- [ ] Escalation paths defined
- [ ] Documentation reviewed
- [ ] Support channels ready

---

**Remember**: The goal is natural, authentic engagement that provides value to communities while respecting Telegram's terms of service and user privacy. Always prioritize safety and compliance over growth metrics.

**Last Updated**: December 2024
**Version**: 1.0.0
**Status**: Production Ready