/**
 * @file api/tally-streaming.ts
 * @brief Real-time tally streaming API for encrypted voting
 *
 * Provides WebSocket-based streaming of encrypted tallies with support for:
 * - Live encrypted running totals
 * - Progressive result disclosure
 * - Multiple concurrent elections
 * - 1,000+ concurrent subscribers
 *
 * Requirements: 18
 */

import type { Ciphertext, PublicKey, ProgressCallback } from './types';
import type { EncryptedBallot, ElectionConfig } from './voting-types';
import type { ElectionId, VoterId } from './audit-trail';
import { EventEmitter } from 'events';

// Re-export for convenience
export type { ElectionId, VoterId };

// Local StreamingEncryptedTally type for streaming (different from voting-types)
export interface StreamingEncryptedTally {
  electionId: ElectionId;
  candidateTotals: Map<string, Ciphertext>;
  totalVotes: Ciphertext;
  timestamp: Date;
  isFinal: boolean;
}

// ============================================================================
// Tally Streaming Types
// ============================================================================

/**
 * Tally update event types
 */
export type TallyEventType =
  | 'ballot_received'
  | 'tally_updated'
  | 'election_started'
  | 'election_ended'
  | 'error'
  | 'subscriber_joined'
  | 'subscriber_left';

/**
 * Tally update event payload
 */
export interface TallyEvent {
  type: TallyEventType;
  electionId: ElectionId;
  timestamp: Date;
  data: TallyEventData;
}

/**
 * Union type for tally event data
 */
export type TallyEventData =
  | BallotReceivedData
  | TallyUpdatedData
  | StreamingElectionStartedData
  | StreamingElectionEndedData
  | StreamingErrorData
  | StreamingSubscriberData;

export interface BallotReceivedData {
  ballotCount: number;
  voterId: VoterId;
}

export interface TallyUpdatedData {
  encryptedTally: StreamingEncryptedTally;
  ballotCount: number;
  progressPercent: number;
}

export interface StreamingElectionStartedData {
  config: ElectionConfig;
  publicKey: PublicKey;
}

export interface StreamingElectionEndedData {
  finalTally: StreamingEncryptedTally;
  totalBallots: number;
}

export interface StreamingErrorData {
  code: string;
  message: string;
}

export interface StreamingSubscriberData {
  subscriberId: string;
  subscriberCount: number;
}

// ============================================================================
// Subscriber Management
// ============================================================================

/**
 * Subscriber connection info
 */
export interface TallySubscriber {
  id: string;
  electionId: ElectionId;
  connectedAt: Date;
  lastEventAt: Date;
  eventCount: number;
}

/**
 * Subscription options
 */
export interface SubscriptionOptions {
  /** Include ballot received events */
  includeBallotEvents?: boolean;
  /** Include tally update events */
  includeTallyEvents?: boolean;
  /** Minimum interval between tally updates (ms) */
  minUpdateIntervalMs?: number;
  /** Filter by specific candidates */
  candidateFilter?: string[];
}

// ============================================================================
// Election State
// ============================================================================

/**
 * Internal election state for streaming
 */
interface ElectionState {
  id: ElectionId;
  config: ElectionConfig;
  publicKey: PublicKey;
  currentTally: StreamingEncryptedTally;
  ballotCount: number;
  startedAt: Date;
  endedAt?: Date;
  subscribers: Map<string, TallySubscriber>;
  lastTallyUpdate: Date;
}

// ============================================================================
// Tally Stream Manager
// ============================================================================

/**
 * Manager for real-time tally streaming across multiple elections
 *
 * Supports:
 * - Multiple concurrent elections
 * - 1,000+ concurrent subscribers per election
 * - Progressive encrypted tally updates
 * - Event-based notification system
 *
 * @example
 * ```typescript
 * const manager = new TallyStreamManager();
 *
 * // Start an election
 * await manager.startElection(electionId, config, publicKey);
 *
 * // Subscribe to updates
 * const unsubscribe = manager.subscribe(electionId, (event) => {
 *   console.log('Tally update:', event);
 * });
 *
 * // Process ballots
 * await manager.processBallot(electionId, ballot);
 *
 * // End election
 * await manager.endElection(electionId);
 * ```
 */
export class TallyStreamManager extends EventEmitter {
  private elections: Map<ElectionId, ElectionState> = new Map();
  private subscriberIdCounter = 0;
  private maxSubscribersPerElection = 10000;
  private defaultUpdateIntervalMs = 100;

  constructor(options?: { maxSubscribersPerElection?: number; defaultUpdateIntervalMs?: number }) {
    super();
    this.setMaxListeners(this.maxSubscribersPerElection);
    if (options?.maxSubscribersPerElection !== undefined) {
      this.maxSubscribersPerElection = options.maxSubscribersPerElection;
    }
    if (options?.defaultUpdateIntervalMs !== undefined) {
      this.defaultUpdateIntervalMs = options.defaultUpdateIntervalMs;
    }
  }

  /**
   * Start a new election for streaming
   */
  async startElection(
    electionId: ElectionId,
    config: ElectionConfig,
    publicKey: PublicKey
  ): Promise<void> {
    if (this.elections.has(electionId)) {
      throw new Error(`Election ${electionId} already exists`);
    }

    const initialTally: StreamingEncryptedTally = {
      electionId,
      candidateTotals: new Map(),
      totalVotes: createZeroCiphertext(publicKey),
      timestamp: new Date(),
      isFinal: false,
    };

    // Initialize candidate totals
    for (const candidate of config.candidates) {
      initialTally.candidateTotals.set(candidate, createZeroCiphertext(publicKey));
    }

    const state: ElectionState = {
      id: electionId,
      config,
      publicKey,
      currentTally: initialTally,
      ballotCount: 0,
      startedAt: new Date(),
      subscribers: new Map(),
      lastTallyUpdate: new Date(),
    };

    this.elections.set(electionId, state);

    this.emitEvent({
      type: 'election_started',
      electionId,
      timestamp: new Date(),
      data: { config, publicKey },
    });
  }

  /**
   * End an election and emit final tally
   */
  async endElection(electionId: ElectionId): Promise<StreamingEncryptedTally> {
    const state = this.getElectionState(electionId);

    state.endedAt = new Date();
    state.currentTally.isFinal = true;
    state.currentTally.timestamp = new Date();

    this.emitEvent({
      type: 'election_ended',
      electionId,
      timestamp: new Date(),
      data: {
        finalTally: state.currentTally,
        totalBallots: state.ballotCount,
      },
    });

    return state.currentTally;
  }

  /**
   * Process a ballot and update the tally
   */
  async processBallot(
    electionId: ElectionId,
    ballot: EncryptedBallot,
    progress?: ProgressCallback
  ): Promise<void> {
    const state = this.getElectionState(electionId);

    if (state.endedAt) {
      throw new Error(`Election ${electionId} has ended`);
    }

    state.ballotCount++;

    // Emit ballot received event
    this.emitEvent({
      type: 'ballot_received',
      electionId,
      timestamp: new Date(),
      data: {
        ballotCount: state.ballotCount,
        voterId: ballot.ballotId, // Use ballotId as voter identifier
      },
    });

    // Update tally (homomorphic addition)
    await this.updateTally(state, ballot, progress);

    // Emit tally update if enough time has passed
    const now = new Date();
    const timeSinceLastUpdate = now.getTime() - state.lastTallyUpdate.getTime();
    if (timeSinceLastUpdate >= this.defaultUpdateIntervalMs) {
      state.lastTallyUpdate = now;
      this.emitTallyUpdate(state);
    }
  }

  /**
   * Process multiple ballots in batch
   */
  async processBallotBatch(
    electionId: ElectionId,
    ballots: EncryptedBallot[],
    progress?: ProgressCallback
  ): Promise<void> {
    const state = this.getElectionState(electionId);

    if (state.endedAt) {
      throw new Error(`Election ${electionId} has ended`);
    }

    const startTime = Date.now();
    for (let i = 0; i < ballots.length; i++) {
      const ballot = ballots[i]!;
      state.ballotCount++;

      await this.updateTally(state, ballot);

      progress?.({
        stage: 'batch_tally',
        current: i + 1,
        total: ballots.length,
        elapsedMs: Date.now() - startTime,
        progressPercent: ((i + 1) / ballots.length) * 100,
      });

      // Emit periodic tally updates during batch processing
      if ((i + 1) % 100 === 0 || i === ballots.length - 1) {
        this.emitTallyUpdate(state);
      }
    }
  }

  /**
   * Subscribe to tally updates for an election
   */
  subscribe(
    electionId: ElectionId,
    callback: (event: TallyEvent) => void,
    options?: SubscriptionOptions
  ): () => void {
    const state = this.getElectionState(electionId);

    if (state.subscribers.size >= this.maxSubscribersPerElection) {
      throw new Error(`Maximum subscribers (${this.maxSubscribersPerElection}) reached for election ${electionId}`);
    }

    const subscriberId = `sub_${++this.subscriberIdCounter}`;
    const subscriber: TallySubscriber = {
      id: subscriberId,
      electionId,
      connectedAt: new Date(),
      lastEventAt: new Date(),
      eventCount: 0,
    };

    state.subscribers.set(subscriberId, subscriber);

    // Create filtered callback based on options
    const filteredCallback = (event: TallyEvent) => {
      if (event.electionId !== electionId) return;

      if (options?.includeBallotEvents === false && event.type === 'ballot_received') return;
      if (options?.includeTallyEvents === false && event.type === 'tally_updated') return;

      subscriber.lastEventAt = new Date();
      subscriber.eventCount++;
      callback(event);
    };

    this.on('tally_event', filteredCallback);

    // Emit subscriber joined event
    this.emitEvent({
      type: 'subscriber_joined',
      electionId,
      timestamp: new Date(),
      data: {
        subscriberId,
        subscriberCount: state.subscribers.size,
      },
    });

    // Return unsubscribe function
    return () => {
      this.off('tally_event', filteredCallback);
      state.subscribers.delete(subscriberId);

      this.emitEvent({
        type: 'subscriber_left',
        electionId,
        timestamp: new Date(),
        data: {
          subscriberId,
          subscriberCount: state.subscribers.size,
        },
      });
    };
  }

  /**
   * Get current tally for an election
   */
  getCurrentTally(electionId: ElectionId): StreamingEncryptedTally {
    return this.getElectionState(electionId).currentTally;
  }

  /**
   * Get election statistics
   */
  getElectionStats(electionId: ElectionId): {
    ballotCount: number;
    subscriberCount: number;
    startedAt: Date;
    endedAt?: Date | undefined;
    isActive: boolean;
  } {
    const state = this.getElectionState(electionId);
    const result: {
      ballotCount: number;
      subscriberCount: number;
      startedAt: Date;
      endedAt?: Date | undefined;
      isActive: boolean;
    } = {
      ballotCount: state.ballotCount,
      subscriberCount: state.subscribers.size,
      startedAt: state.startedAt,
      isActive: state.endedAt === undefined,
    };
    if (state.endedAt !== undefined) {
      result.endedAt = state.endedAt;
    }
    return result;
  }

  /**
   * Get all active elections
   */
  getActiveElections(): ElectionId[] {
    return Array.from(this.elections.entries())
      .filter(([_, state]) => state.endedAt === undefined)
      .map(([id]) => id);
  }

  /**
   * Get subscriber info for an election
   */
  getSubscribers(electionId: ElectionId): TallySubscriber[] {
    return Array.from(this.getElectionState(electionId).subscribers.values());
  }

  /**
   * Force emit a tally update
   */
  forceTallyUpdate(electionId: ElectionId): void {
    const state = this.getElectionState(electionId);
    state.lastTallyUpdate = new Date();
    this.emitTallyUpdate(state);
  }

  /**
   * Clean up ended elections
   */
  cleanupEndedElections(olderThanMs: number = 3600000): number {
    const now = Date.now();
    let cleaned = 0;

    for (const [id, state] of this.elections) {
      if (state.endedAt && now - state.endedAt.getTime() > olderThanMs) {
        this.elections.delete(id);
        cleaned++;
      }
    }

    return cleaned;
  }

  /**
   * Dispose of all resources
   */
  dispose(): void {
    this.removeAllListeners();
    this.elections.clear();
  }

  // ========================================================================
  // Private Methods
  // ========================================================================

  private getElectionState(electionId: ElectionId): ElectionState {
    const state = this.elections.get(electionId);
    if (!state) {
      throw new Error(`Election ${electionId} not found`);
    }
    return state;
  }

  private async updateTally(
    state: ElectionState,
    ballot: EncryptedBallot,
    _progress?: ProgressCallback
  ): Promise<void> {
    // Update total votes (homomorphic addition)
    state.currentTally.totalVotes = await homomorphicAdd(
      state.currentTally.totalVotes,
      ballot.encryptedChoices[0]!
    );

    // Update candidate totals
    // In a real implementation, this would use the ballot's encrypted choice
    // to update the appropriate candidate's total
    state.currentTally.timestamp = new Date();
  }

  private emitTallyUpdate(state: ElectionState): void {
    const expectedTotal = state.ballotCount; // Use actual ballot count
    const progressPercent = expectedTotal > 0 ? (state.ballotCount / expectedTotal) * 100 : 0;

    this.emitEvent({
      type: 'tally_updated',
      electionId: state.id,
      timestamp: new Date(),
      data: {
        encryptedTally: state.currentTally,
        ballotCount: state.ballotCount,
        progressPercent: Math.min(progressPercent, 100),
      },
    });
  }

  private emitEvent(event: TallyEvent): void {
    this.emit('tally_event', event);
  }
}

// ============================================================================
// WebSocket Server Adapter
// ============================================================================

/**
 * WebSocket message types
 */
export type WebSocketMessageType =
  | 'subscribe'
  | 'unsubscribe'
  | 'get_tally'
  | 'get_stats'
  | 'tally_event'
  | 'error';

/**
 * WebSocket message structure
 */
export interface WebSocketMessage {
  type: WebSocketMessageType;
  electionId?: ElectionId;
  data?: unknown;
  requestId?: string;
}

/**
 * WebSocket connection interface (framework-agnostic)
 */
export interface WebSocketConnection {
  send(data: string): void;
  close(): void;
  on(event: 'message', handler: (data: string) => void): void;
  on(event: 'close', handler: () => void): void;
  on(event: 'error', handler: (error: Error) => void): void;
}

/**
 * WebSocket adapter for TallyStreamManager
 *
 * Provides a WebSocket interface for real-time tally streaming.
 * Framework-agnostic - works with ws, socket.io, or any WebSocket implementation.
 *
 * @example
 * ```typescript
 * const manager = new TallyStreamManager();
 * const adapter = new TallyWebSocketAdapter(manager);
 *
 * // Handle new WebSocket connection
 * wss.on('connection', (ws) => {
 *   adapter.handleConnection(ws);
 * });
 * ```
 */
export class TallyWebSocketAdapter {
  private manager: TallyStreamManager;
  private connections: Map<WebSocketConnection, Set<() => void>> = new Map();

  constructor(manager: TallyStreamManager) {
    this.manager = manager;
  }

  /**
   * Handle a new WebSocket connection
   */
  handleConnection(ws: WebSocketConnection): void {
    const unsubscribers = new Set<() => void>();
    this.connections.set(ws, unsubscribers);

    ws.on('message', (data: string) => {
      try {
        const message = JSON.parse(data) as WebSocketMessage;
        this.handleMessage(ws, message, unsubscribers);
      } catch (error) {
        this.sendError(ws, 'Invalid message format', undefined);
      }
    });

    ws.on('close', () => {
      // Clean up all subscriptions
      for (const unsubscribe of unsubscribers) {
        unsubscribe();
      }
      this.connections.delete(ws);
    });

    ws.on('error', () => {
      // Clean up on error
      for (const unsubscribe of unsubscribers) {
        unsubscribe();
      }
      this.connections.delete(ws);
    });
  }

  /**
   * Get connection count
   */
  getConnectionCount(): number {
    return this.connections.size;
  }

  /**
   * Broadcast to all connections subscribed to an election
   */
  broadcast(electionId: ElectionId, event: TallyEvent): void {
    const message: WebSocketMessage = {
      type: 'tally_event',
      electionId,
      data: serializeTallyEvent(event),
    };
    const messageStr = JSON.stringify(message);

    for (const [ws] of this.connections) {
      try {
        ws.send(messageStr);
      } catch {
        // Connection may be closed
      }
    }
  }

  /**
   * Close all connections
   */
  closeAll(): void {
    for (const [ws, unsubscribers] of this.connections) {
      for (const unsubscribe of unsubscribers) {
        unsubscribe();
      }
      ws.close();
    }
    this.connections.clear();
  }

  private handleMessage(
    ws: WebSocketConnection,
    message: WebSocketMessage,
    unsubscribers: Set<() => void>
  ): void {
    switch (message.type) {
      case 'subscribe':
        this.handleSubscribe(ws, message, unsubscribers);
        break;
      case 'unsubscribe':
        this.handleUnsubscribe(message, unsubscribers);
        break;
      case 'get_tally':
        this.handleGetTally(ws, message);
        break;
      case 'get_stats':
        this.handleGetStats(ws, message);
        break;
      default:
        this.sendError(ws, `Unknown message type: ${message.type}`, message.requestId);
    }
  }

  private handleSubscribe(
    ws: WebSocketConnection,
    message: WebSocketMessage,
    unsubscribers: Set<() => void>
  ): void {
    if (!message.electionId) {
      this.sendError(ws, 'electionId required', message.requestId);
      return;
    }

    try {
      const unsubscribe = this.manager.subscribe(message.electionId, (event) => {
        const wsMessage: WebSocketMessage = {
          type: 'tally_event',
          electionId: event.electionId,
          data: serializeTallyEvent(event),
        };
        ws.send(JSON.stringify(wsMessage));
      });

      unsubscribers.add(unsubscribe);

      ws.send(
        JSON.stringify({
          type: 'subscribe',
          electionId: message.electionId,
          data: { success: true },
          requestId: message.requestId,
        })
      );
    } catch (error) {
      this.sendError(ws, (error as Error).message, message.requestId);
    }
  }

  private handleUnsubscribe(_message: WebSocketMessage, unsubscribers: Set<() => void>): void {
    // Unsubscribe from all for this election
    // In a more sophisticated implementation, we'd track subscriptions by election
    for (const unsubscribe of unsubscribers) {
      unsubscribe();
    }
    unsubscribers.clear();
  }

  private handleGetTally(ws: WebSocketConnection, message: WebSocketMessage): void {
    if (!message.electionId) {
      this.sendError(ws, 'electionId required', message.requestId);
      return;
    }

    try {
      const tally = this.manager.getCurrentTally(message.electionId);
      ws.send(
        JSON.stringify({
          type: 'get_tally',
          electionId: message.electionId,
          data: serializeStreamingEncryptedTally(tally),
          requestId: message.requestId,
        })
      );
    } catch (error) {
      this.sendError(ws, (error as Error).message, message.requestId);
    }
  }

  private handleGetStats(ws: WebSocketConnection, message: WebSocketMessage): void {
    if (!message.electionId) {
      this.sendError(ws, 'electionId required', message.requestId);
      return;
    }

    try {
      const stats = this.manager.getElectionStats(message.electionId);
      ws.send(
        JSON.stringify({
          type: 'get_stats',
          electionId: message.electionId,
          data: stats,
          requestId: message.requestId,
        })
      );
    } catch (error) {
      this.sendError(ws, (error as Error).message, message.requestId);
    }
  }

  private sendError(ws: WebSocketConnection, errorMessage: string, requestId?: string): void {
    ws.send(
      JSON.stringify({
        type: 'error',
        data: { message: errorMessage },
        requestId,
      })
    );
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Create a zero ciphertext (placeholder for actual FHE implementation)
 */
function createZeroCiphertext(pk: PublicKey): Ciphertext {
  return {
    __brand: 'Ciphertext',
    handle: BigInt(0),
    keyId: pk.keyId,
    noiseBudget: 100,
    isNtt: false,
    degree: 1,
  };
}

/**
 * Homomorphic addition (placeholder for actual FHE implementation)
 */
async function homomorphicAdd(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext> {
  return {
    __brand: 'Ciphertext',
    handle: BigInt(0),
    keyId: ct1.keyId,
    noiseBudget: Math.min(ct1.noiseBudget, ct2.noiseBudget) - 1,
    isNtt: ct1.isNtt,
    degree: Math.max(ct1.degree, ct2.degree),
  };
}

/**
 * Serialize a tally event for WebSocket transmission
 */
function serializeTallyEvent(event: TallyEvent): Record<string, unknown> {
  return {
    type: event.type,
    electionId: event.electionId,
    timestamp: event.timestamp.toISOString(),
    data: serializeEventData(event.data),
  };
}

/**
 * Serialize event data based on type
 */
function serializeEventData(data: TallyEventData): Record<string, unknown> {
  if ('encryptedTally' in data) {
    const d = data as TallyUpdatedData;
    return {
      ballotCount: d.ballotCount,
      progressPercent: d.progressPercent,
      encryptedTally: serializeStreamingEncryptedTally(d.encryptedTally),
    };
  }
  if ('finalTally' in data) {
    const d = data as StreamingElectionEndedData;
    return {
      totalBallots: d.totalBallots,
      finalTally: serializeStreamingEncryptedTally(d.finalTally),
    };
  }
  if ('publicKey' in data) {
    const d = data as StreamingElectionStartedData;
    return {
      config: d.config,
      publicKey: serializePublicKey(d.publicKey),
    };
  }
  // For other data types, return as-is
  return { ...data };
}

/**
 * Serialize encrypted tally for transmission
 */
function serializeStreamingEncryptedTally(tally: StreamingEncryptedTally): Record<string, unknown> {
  const candidateTotals: Record<string, unknown> = {};
  for (const [id, ct] of tally.candidateTotals) {
    candidateTotals[id] = serializeCiphertext(ct);
  }

  return {
    electionId: tally.electionId,
    candidateTotals,
    totalVotes: serializeCiphertext(tally.totalVotes),
    timestamp: tally.timestamp.toISOString(),
    isFinal: tally.isFinal,
  };
}

/**
 * Serialize ciphertext for transmission
 */
function serializeCiphertext(ct: Ciphertext): Record<string, unknown> {
  return {
    handle: ct.handle.toString(),
    keyId: ct.keyId.toString(),
    noiseBudget: ct.noiseBudget,
    isNtt: ct.isNtt,
    degree: ct.degree,
  };
}

/**
 * Serialize public key for transmission
 */
function serializePublicKey(pk: PublicKey): Record<string, unknown> {
  return {
    handle: pk.handle.toString(),
    keyId: pk.keyId.toString(),
  };
}

// ============================================================================
// Exports
// ============================================================================

export {
  createZeroCiphertext,
  homomorphicAdd,
  serializeTallyEvent,
  serializeStreamingEncryptedTally,
};
