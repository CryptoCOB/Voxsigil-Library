/**
 * VoxSigil Library - OpenClawd Adapter (JavaScript)
 *
 * Complete integration layer for OpenClawd agents into VoxBridge Molt network.
 *
 * Features:
 * - HTTP client for VoxBridge endpoints
 * - Agent registration and authentication
 * - Event emission and heartbeat management
 * - Knowledge base retrieval
 * - Sigil generation utilities
 */

const crypto = require('crypto');
const https = require('https');
const http = require('http');

/**
 * HTTP client for VoxBridge endpoints
 */
class VoxBridgeClient {
  constructor(options = {}) {
    this.agentName = options.agentName;
    this.agentType = options.agentType || 'llm';
    this.sigilPublicKey = options.sigilPublicKey;
    this.description = options.description;
    this.webhookUrl = options.webhookUrl;
    this.canIngest = options.canIngest !== false;
    this.canControl = options.canControl || false;
    this.canApproveIntents = options.canApproveIntents || false;
    this.metadata = options.metadata || {};
    this.timeout = options.timeout || 60000; // 60 seconds
    this.baseUrl = (options.baseUrl || 'https://voxsigil-predict.fly.dev').replace(/\/$/, '');
    this.agentId = null;
    this.registrationTime = null;
    this.lastHeartbeatTime = null;
  }

  /**
   * Make HTTP request to VoxBridge
   */
  async _request(method, path, params = null, jsonBody = null) {
    return new Promise((resolve, reject) => {
      const url = new URL(path, this.baseUrl);
      if (params) {
        Object.keys(params).forEach(key => {
          url.searchParams.append(key, params[key]);
        });
      }

      const options = {
        method: method.toUpperCase(),
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'VoxSigil-Library/2.0.0'
        },
        timeout: this.timeout
      };

      const req = (url.protocol === 'https:' ? https : http).request(url, options, (res) => {
        let data = '';

        res.on('data', (chunk) => {
          data += chunk;
        });

        res.on('end', () => {
          try {
            if (data) {
              const jsonData = JSON.parse(data);
              if (res.statusCode >= 200 && res.statusCode < 300) {
                resolve(jsonData);
              } else {
                reject(new Error(`HTTP ${res.statusCode}: ${jsonData.message || data}`));
              }
            } else {
              resolve({});
            }
          } catch (e) {
            reject(new Error(`Invalid JSON response: ${e.message}`));
          }
        });
      });

      req.on('error', (err) => {
        reject(new Error(`Request failed: ${err.message}`));
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });

      if (jsonBody) {
        req.write(JSON.stringify(jsonBody));
      }

      req.end();
    });
  }

  /**
   * Register agent with VoxBridge
   */
  async register() {
    if (this.agentId) {
      return { status: 'already_registered', agent_id: this.agentId };
    }

    const payload = {
      agent_name: this.agentName,
      agent_type: this.agentType,
      sigil_public_key: this.sigilPublicKey,
      description: this.description || 'Automated OpenClaw agent',
      webhook_url: this.webhookUrl,
      can_ingest: this.canIngest,
      can_control: this.canControl,
      can_approve_intents: this.canApproveIntents,
      metadata: this.metadata
    };

    try {
      const response = await this._request('POST', '/api/v1/voxbridge/agents/register', null, payload);
      this.agentId = response.id || response.agent_id;
      this.registrationTime = new Date();
      console.log(`✅ Registered agent ${this.agentName} with ID ${this.agentId}`);
      return response;
    } catch (error) {
      throw new Error(`Failed to register agent: ${error.message}`);
    }
  }

  /**
   * Fetch full VoxSigil context/knowledge base
   */
  async getKnowledgeBase() {
    try {
      return await this._request('GET', '/api/v1/openclaw/knowledge');
    } catch (error) {
      console.warn('Could not fetch knowledge base:', error.message);
      return { status: 'unavailable', reason: 'No direct knowledge endpoint on core bridge' };
    }
  }

  /**
   * Send heartbeat
   */
  async heartbeat() {
    const agentId = this._requireAgentId();
    try {
      const response = await this._request('POST', `/api/v1/voxbridge/agents/${agentId}/heartbeat`);
      this.lastHeartbeatTime = new Date();
      return response;
    } catch (error) {
      throw new Error(`Heartbeat failed: ${error.message}`);
    }
  }

  /**
   * Send event to VoxBridge feed
   */
  async sendEvent(eventType, title, description = '', impactScore = 50.0, data = null) {
    const agentId = this._requireAgentId();

    let finalDescription = description;
    if (data) {
      const dataBlob = JSON.stringify(data, null, 2);
      finalDescription = description ? `${description}\n\nData: ${dataBlob}` : `Data: ${dataBlob}`;
    }

    return await this._request('POST', '/api/v1/voxbridge/feed/events', null, {
      event_type: eventType,
      title: title,
      description: finalDescription,
      agent_id: agentId,
      impact_score: impactScore
    });
  }

  /**
   * Ensure agent is registered
   */
  _requireAgentId() {
    if (!this.agentId) {
      throw new Error('Agent is not registered yet. Call register() first.');
    }
    return this.agentId;
  }
}

/**
 * OpenClawd Event
 */
class OpenClawdEvent {
  constructor(options = {}) {
    this.outputType = options.outputType;
    this.title = options.title;
    this.description = options.description;
    this.impactScore = options.impactScore || 50.0;
    this.data = options.data;
    this.eventId = options.eventId || this._generateId();
    this.timestamp = options.timestamp || new Date().toISOString();
  }

  toDict() {
    const result = {
      output_type: this.outputType,
      title: this.title,
      description: this.description,
      impact_score: this.impactScore,
      data: this.data,
      event_id: this.eventId,
      timestamp: this.timestamp
    };
    return Object.fromEntries(
      Object.entries(result).filter(([_, v]) => v != null)
    );
  }

  _generateId() {
    return crypto.randomUUID ? crypto.randomUUID() : crypto.randomBytes(16).toString('hex');
  }
}

/**
 * Maps OpenClawd outputs to VoxBridge feed events
 */
class OpenClawdAdapter {
  constructor(client, eventMap = null) {
    this.client = client;
    this.eventMap = {
      forecast: 'llm_insight',
      tool_discovery: 'perception_discovery',
      market_insight: 'market_trigger',
      hypothesis: 'agent_discovery',
      confidence_update: 'consensus_shift',
      alert: 'honeypot_alert',
      research: 'user_research',
      chat: 'user_chat',
      ...eventMap
    };
    this.emittedEvents = [];
    this.heartbeatInterval = null;
    this._stopHeartbeat = false;
    this.knowledgeBase = {};
  }

  /**
   * Register agent and fetch initial knowledge
   */
  async bootstrap() {
    const result = await this.client.register();
    try {
      this.knowledgeBase = await this.client.getKnowledgeBase();
    } catch (error) {
      console.warn('Could not fetch knowledge base on bootstrap:', error.message);
    }
    return result;
  }

  /**
   * Get full VoxSigil context for LLM context window
   */
  async getKnowledge() {
    if (!Object.keys(this.knowledgeBase).length) {
      try {
        this.knowledgeBase = await this.client.getKnowledgeBase();
      } catch (error) {
        console.error('Failed to fetch knowledge:', error.message);
        return { error: 'Failed to fetch knowledge' };
      }
    }
    return this.knowledgeBase;
  }

  /**
   * Build a sigil representation for a data string
   */
  buildSigil(data) {
    const hash = crypto.createHash('sha256').update(data).digest('hex');
    return `🜮{${hash.substring(0, 8)}}`;
  }

  /**
   * Emit event to VoxBridge
   */
  async emit(event) {
    const eventType = this.mapEventType(event.outputType);
    const result = await this.client.sendEvent(
      eventType,
      event.title,
      event.description,
      event.impactScore,
      event.data
    );

    this.emittedEvents.push({
      input_type: event.outputType,
      mapped_type: eventType,
      title: event.title,
      result: result,
      timestamp: new Date().toISOString()
    });

    return result;
  }

  /**
   * Map OpenClawd output type to VoxBridge event type
   */
  mapEventType(outputType) {
    const normalized = outputType.toLowerCase().replace(/[^a-z_]/g, '');
    return this.eventMap[normalized] || 'agent_discovery';
  }

  /**
   * Start heartbeat loop
   */
  startHeartbeatLoop(intervalSeconds = 300) {
    if (this.heartbeatInterval) return;

    this._stopHeartbeat = false;
    this.heartbeatInterval = setInterval(async () => {
      if (this._stopHeartbeat) return;
      try {
        await this.client.heartbeat();
      } catch (error) {
        console.error('Heartbeat failed:', error.message);
      }
    }, intervalSeconds * 1000);
  }

  /**
   * Stop heartbeat loop
   */
  stopHeartbeatLoop() {
    this._stopHeartbeat = true;
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }
}

/**
 * Factory for creating OpenClawd adapters
 */
class OpenClawdAgentFactory {
  static create(options = {}) {
    const name = options.name || options.agentName;
    const agentType = options.agentType || 'llm';
    const voxbridgeUrl = options.voxbridgeUrl || options.baseUrl;
    const description = options.description;
    const generateSigil = options.generateSigil !== false;
    const timeout = options.timeout || 60000;

    const sigilPublicKey = generateSigil
      ? `0x${crypto.randomBytes(16).toString('hex')}`
      : '';

    const client = new VoxBridgeClient({
      agentName: name,
      agentType: agentType,
      sigilPublicKey: sigilPublicKey,
      baseUrl: voxbridgeUrl,
      description: description,
      timeout: timeout
    });

    return new OpenClawdAdapter(client);
  }
}

module.exports = {
  VoxBridgeClient,
  OpenClawdAdapter,
  OpenClawdEvent,
  OpenClawdAgentFactory
};