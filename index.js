import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import crypto from 'crypto';
import Anthropic from '@anthropic-ai/sdk';
import { createClient } from '@supabase/supabase-js';
import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';
import dotenv from 'dotenv';
dotenv.config();

const app = express();

app.use(helmet());
app.use(express.json({ limit: '4mb' }));
app.use(cors({
  origin: [
    process.env.FRONTEND_URL,
    'http://localhost:5173',
  ]
}));

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const ratelimit = new Ratelimit({
  redis: Redis.fromEnv(),
  limiter: Ratelimit.slidingWindow(60, '1 h'),
});

function hashKey(key) {
  return crypto.createHash('sha256').update(key).digest('hex');
}

async function logUsage(userId, agentId, statusCode, durationMs, tokensUsed, inputPreview, outputPreview) {
  await supabase.from('usage_logs').insert({
    user_id: userId,
    agent_id: agentId,
    status_code: statusCode,
    duration_ms: durationMs,
    tokens_used: tokensUsed,
    input_preview: inputPreview,
    output_preview: outputPreview,
  });
}

async function authenticate(req, res, next) {
  const key = req.headers['x-api-key'];
  if (!key) return res.status(401).json({ error: 'Missing x-api-key header' });

  const keyHash = hashKey(key);
  const { data: keyRecord, error } = await supabase
    .from('api_keys')
    .select('*, agents(*)')
    .eq('key_hash', keyHash)
    .eq('status', 'active')
    .single();

  if (error || !keyRecord) return res.status(403).json({ error: 'Invalid or revoked API key' });

  const { success } = await ratelimit.limit(keyRecord.id);
  if (!success) return res.status(429).json({ error: 'Rate limit exceeded. Max 60 requests per hour.' });

  req.keyRecord = keyRecord;
  req.agent = keyRecord.agents;
  next();
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.post('/api/:userId/:slug', authenticate, async (req, res) => {
  const startTime = Date.now();
  const { userId, slug } = req.params;
  const { input, messages: incomingMessages } = req.body;
  const agent = req.agent;

  if (!agent) return res.status(404).json({ error: 'Agent not found' });

  let messages = [];
  if (agent.memory && incomingMessages && Array.isArray(incomingMessages)) {
    messages = incomingMessages;
  }
  if (input) messages.push({ role: 'user', content: String(input) });
  if (messages.length === 0) return res.status(400).json({ error: 'No input provided' });

  let tools = [];
  try {
    tools = Array.isArray(agent.tools) ? agent.tools : JSON.parse(agent.tools || '[]');
  } catch { tools = []; }

  try {
    let response = await anthropic.messages.create({
      model: agent.model ?? 'claude-sonnet-4-6',
      system: agent.system_prompt ?? 'You are a helpful assistant.',
      messages,
      max_tokens: agent.max_tokens ?? 4096,
      tools: tools.length > 0 ? tools : undefined,
    });

    while (response.stop_reason === 'tool_use') {
      const toolUseBlock = response.content.find(b => b.type === 'tool_use');
      const toolResult = 'Tool "' + toolUseBlock.name + '" called with: ' + JSON.stringify(toolUseBlock.input);
      messages.push(
        { role: 'assistant', content: response.content },
        { role: 'user', content: [{ type: 'tool_result', tool_use_id: toolUseBlock.id, content: String(toolResult) }] }
      );
      response = await anthropic.messages.create({
        model: agent.model ?? 'claude-sonnet-4-6',
        system: agent.system_prompt ?? 'You are a helpful assistant.',
        messages,
        max_tokens: agent.max_tokens ?? 4096,
        tools: tools.length > 0 ? tools : undefined,
      });
    }

    const textBlock = response.content.find(b => b.type === 'text');
    const result = textBlock?.text ?? '';
    const tokensUsed = (response.usage?.input_tokens ?? 0) + (response.usage?.output_tokens ?? 0);
    const durationMs = Date.now() - startTime;

    logUsage(userId, agent.id, 200, durationMs, tokensUsed, String(input ?? '').slice(0, 200), result.slice(0, 200)).catch(() => {});
    supabase.from('agents').update({ call_count_today: (agent.call_count_today ?? 0) + 1, last_active_at: new Date().toISOString() }).eq('id', agent.id).then(() => {});

    return res.json({
      result,
      tokens_used: tokensUsed,
      duration_ms: durationMs,
      messages: agent.memory ? [...messages, { role: 'assistant', content: result }] : undefined,
    });
  } catch (err) {
    const durationMs = Date.now() - startTime;
    logUsage(userId, agent.id, 500, durationMs, 0, String(input ?? '').slice(0, 200), err.message).catch(() => {});
    console.error('Agent run error:', err);
    return res.status(500).json({ error: 'Agent execution failed', details: err.message });
  }
});

app.post('/api/keys/generate', async (req, res) => {
  const { agentId, userId } = req.body;
  if (!agentId || !userId) return res.status(400).json({ error: 'agentId and userId are required' });
  const rawKey = 'sk-ag-' + crypto.randomBytes(24).toString('hex');
  const keyHash = hashKey(rawKey);
  const { data, error } = await supabase.from('api_keys').insert({ agent_id: agentId, user_id: userId, key_hash: keyHash, status: 'active' }).select().single();
  if (error) return res.status(500).json({ error: 'Failed to create API key' });
  return res.json({ key: rawKey, id: data.id });
});

app.delete('/api/keys/:keyId', async (req, res) => {
  const { keyId } = req.params;
  const { error } = await supabase.from('api_keys').update({ status: 'revoked' }).eq('id', keyId);
  if (error) return res.status(500).json({ error: 'Failed to revoke key' });
  return res.json({ success: true });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => { console.log('Agent gateway running on port ' + PORT); });
