import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import crypto from 'crypto';
import OpenAI from 'openai';
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

// Supabase admin client
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// OpenRouter client (OpenAI-compatible)
const openai = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY,
  baseURL: 'https://openrouter.ai/api/v1',
  defaultHeaders: {
    'HTTP-Referer': process.env.FRONTEND_URL ?? 'https://antimattr.com',
    'X-Title': 'antimattr',
  },
});

// Default model — users can override per-agent
const DEFAULT_MODEL = 'anthropic/claude-sonnet-4-5';

// Rate limiter
const ratelimit = new Ratelimit({
  redis: Redis.fromEnv(),
  limiter: Ratelimit.slidingWindow(60, '1 h'),
});

// — Utility: hash an API key ————————————————————————
function hashKey(key) {
  return crypto
    .createHash('sha256')
    .update(key)
    .digest('hex');
}

// — Utility: log usage ————————————————————————————
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

// — Utility: convert stored tool defs to OpenAI function format ————
function toOpenAITools(tools) {
  return tools.map(t => {
    if (t.type === 'function') return t;
    return {
      type: 'function',
      function: {
        name: t.name,
        description: t.description ?? '',
        parameters: t.input_schema ?? t.parameters ?? { type: 'object', properties: {} },
      },
    };
  });
}

// — Middleware: authenticate ————————————————————————
async function authenticate(req, res, next) {
  const key = req.headers['x-api-key'];
  if (!key) {
    return res.status(401).json({ error: 'Missing x-api-key header' });
  }

  const keyHash = hashKey(key);

  const { data: keyRecord, error } = await supabase
    .from('api_keys')
    .select('*, agents(*)')
    .eq('key_hash', keyHash)
    .eq('status', 'active')
    .single();

  if (error || !keyRecord) {
    return res.status(403).json({ error: 'Invalid or revoked API key' });
  }

  const { success } = await ratelimit.limit(keyRecord.id);
  if (!success) {
    return res.status(429).json({ error: 'Rate limit exceeded. Max 60 requests per hour.' });
  }

  req.keyRecord = keyRecord;
  req.agent = keyRecord.agents;
  next();
}

// — Health check ——————————————————————————————————
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// — List available models ————————————————————————————
app.get('/models', async (req, res) => {
  try {
    const response = await fetch('https://openrouter.ai/api/v1/models', {
      headers: { Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}` },
    });
    const data = await response.json();
    const models = (data.data ?? []).map(m => ({
      id: m.id,
      name: m.name,
      context_length: m.context_length,
      pricing: m.pricing,
    }));
    return res.json({ models });
  } catch (err) {
    return res.status(500).json({ error: 'Failed to fetch models', details: err.message });
  }
});

// — Run agent ————————————————————————————————————
app.post('/api/:userId/:slug', authenticate, async (req, res) => {
  const startTime = Date.now();
  const { userId, slug } = req.params;
  const { input, messages: incomingMessages } = req.body;
  const agent = req.agent;

  if (!agent) {
    return res.status(404).json({ error: 'Agent not found' });
  }

  let messages = [
    { role: 'system', content: agent.system_prompt ?? 'You are a helpful assistant.' },
  ];

  if (agent.memory && incomingMessages && Array.isArray(incomingMessages)) {
    const history = incomingMessages.filter(m => m.role !== 'system');
    messages.push(...history);
  }

  if (input) {
    messages.push({ role: 'user', content: String(input) });
  }

  if (messages.length <= 1) {
    return res.status(400).json({ error: 'No input provided' });
  }

  let rawTools = [];
  try {
    rawTools = Array.isArray(agent.tools) ? agent.tools : JSON.parse(agent.tools || '[]');
  } catch {
    rawTools = [];
  }
  const tools = rawTools.length > 0 ? toOpenAITools(rawTools) : undefined;

  const model = agent.model ?? DEFAULT_MODEL;

  try {
    let totalTokens = 0;

    let completion = await openai.chat.completions.create({
      model,
      messages,
      max_tokens: agent.max_tokens ?? 4096,
      tools,
      tool_choice: tools ? 'auto' : undefined,
    });

    while (completion.choices[0].finish_reason === 'tool_calls') {
      const assistantMessage = completion.choices[0].message;
      messages.push(assistantMessage);
      totalTokens += (completion.usage?.total_tokens ?? 0);

      const toolResults = await Promise.all(
        (assistantMessage.tool_calls ?? []).map(async (call) => {
          let result;
          try {
            const args = JSON.parse(call.function.arguments ?? '{}');
            result = `Tool "${call.function.name}" called with: ${JSON.stringify(args)}`;
          } catch {
            result = `Tool "${call.function.name}" call failed to parse arguments.`;
          }
          return {
            role: 'tool',
            tool_call_id: call.id,
            content: String(result),
          };
        })
      );

      messages.push(...toolResults);

      completion = await openai.chat.completions.create({
        model,
        messages,
        max_tokens: agent.max_tokens ?? 4096,
        tools,
        tool_choice: 'auto',
      });
    }

    const result = completion.choices[0].message?.content ?? '';
    totalTokens += (completion.usage?.total_tokens ?? 0);
    const durationMs = Date.now() - startTime;

    logUsage(
      userId, agent.id, 200, durationMs, totalTokens,
      String(input ?? '').slice(0, 200), result.slice(0, 200)
    ).catch(() => {});

    supabase
      .from('agents')
      .update({
        call_count_today: (agent.call_count_today ?? 0) + 1,
        last_active_at: new Date().toISOString(),
      })
      .eq('id', agent.id)
      .then(() => {});

    const returnMessages = agent.memory
      ? [...messages.filter(m => m.role !== 'system'), { role: 'assistant', content: result }]
      : undefined;

    return res.json({
      result,
      model,
      tokens_used: totalTokens,
      duration_ms: durationMs,
      messages: returnMessages,
    });

  } catch (err) {
    const durationMs = Date.now() - startTime;
    logUsage(userId, agent.id, 500, durationMs, 0, String(input ?? '').slice(0, 200), err.message).catch(() => {});
    console.error('Agent run error:', err);
    return res.status(500).json({ error: 'Agent execution failed', details: err.message });
  }
});

// — Generate API key ————————————————————————————————
app.post('/api/keys/generate', async (req, res) => {
  const { agentId, userId } = req.body;

  if (!agentId || !userId) {
    return res.status(400).json({ error: 'agentId and userId are required' });
  }

  const rawKey = `sk-ag-${crypto.randomBytes(24).toString('hex')}`;
  const keyHash = hashKey(rawKey);

  const { data, error } = await supabase.from('api_keys').insert({
    agent_id: agentId,
    user_id: userId,
    key_hash: keyHash,
    status: 'active',
  }).select().single();

  if (error) {
    return res.status(500).json({ error: 'Failed to create API key' });
  }

  return res.json({ key: rawKey, id: data.id });
});

// — Revoke API key ——————————————————————————————————
app.delete('/api/keys/:keyId', async (req, res) => {
  const { keyId } = req.params;

  const { error } = await supabase
    .from('api_keys')
    .update({ status: 'revoked' })
    .eq('id', keyId);

  if (error) {
    return res.status(500).json({ error: 'Failed to revoke key' });
  }

  return res.json({ success: true });
});

// Start
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Agent gateway running on port ${PORT} | Default model: ${DEFAULT_MODEL}`);
});
