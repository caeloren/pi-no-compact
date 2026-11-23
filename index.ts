/**
 * pi-no-compact: Minimal-tail Retrieval Compaction for Pi.
 *
 * Compaction writes no LLM summary and keeps no recent tail. The new context holds a fixed
 * instruction and the last structurally complete fragment of the active branch. Everything else
 * stays in the session and is retrieved on demand with `history_search` and `history_read`.
 */

import { createHash } from "node:crypto";
import { type ImageContent, StringEnum, type TextContent } from "@earendil-works/pi-ai";
import {
	buildSessionProjection,
	type ContextEditEntry,
	convertToLlm,
	type ExtensionAPI,
	type ExtensionContext,
	type ProjectedSessionEntry,
	type SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

type AgentMessage = ProjectedSessionEntry["messages"][number];
type SessionReader = ExtensionContext["sessionManager"];

const CONTINUATION_INSTRUCTION = [
	"Context was compacted. Earlier records are no longer in your context but remain available through history_search and history_read.",
	"Before continuing, read earlier messages, then the latest records.",
	"For any other missing detail, search instead of guessing, and use history_read for exact text.",
].join("\n");

const HISTORY_SEARCH = "history_search";
const HISTORY_READ = "history_read";
/** Calls to these tools and their results are never searched or read, so the tools do not echo themselves. */
const HISTORY_TOOLS: ReadonlySet<string> = new Set([HISTORY_SEARCH, HISTORY_READ]);

const HISTORY_KINDS = [
	"user",
	"assistant",
	"tool_call",
	"tool_result",
	"bash",
	"custom_message",
	"branch_summary",
] as const;
type HistoryKind = (typeof HISTORY_KINDS)[number];

const KIND_BY_ROLE = {
	user: "user",
	custom: "custom_message",
	bashExecution: "bash",
	branchSummary: "branch_summary",
} as const satisfies Record<string, HistoryKind>;

const CURSOR_VERSION = 2;

const READ_ONLY_ANNOTATIONS = {
	readOnlyHint: true,
	destructiveHint: false,
	idempotentHint: true,
	openWorldHint: false,
} as const;

// ---------------------------------------------------------------------------
// Tool parameters
// ---------------------------------------------------------------------------

const historySearchParameters = Type.Object({
	query: Type.Union([Type.String(), Type.Null()], {
		description: "Text to search for, or null to browse",
	}),
	match: Type.Union([StringEnum(["terms", "literal"] as const), Type.Null()], {
		description:
			'"terms": whole words after NFKC normalization, case-sensitive, CJK text as two-character pairs, any word matches. "literal": exact case-sensitive substring. null when query is null',
	}),
	kinds: Type.Array(StringEnum(HISTORY_KINDS), {
		minItems: 1,
		description: "Record kinds to include, at least one",
	}),
	order: StringEnum(["relevance", "oldest", "newest"] as const, {
		description:
			'"relevance" (search only): most distinct query words first, then newest. "oldest" or "newest": position on the branch',
	}),
	limit: Type.Integer({ minimum: 1, description: "Results per page" }),
	cursor: Type.Union([Type.String(), Type.Null()], {
		description:
			"null for the first page; otherwise nextCursor from the previous page, with the same query, match, kinds and order",
	}),
});

const historyReadParameters = Type.Object({
	entryIds: Type.Array(Type.String(), {
		minItems: 1,
		description: "entryId values of the records to read, at least one",
	}),
	around: Type.Integer({
		minimum: 0,
		description: "Readable records to add before and after each requested record; 0 for none",
	}),
	includeImages: Type.Boolean({
		description: "true to attach images; false to show image placeholders only",
	}),
	maxImages: Type.Integer({
		minimum: 0,
		description: "Images to attach: 0 when includeImages is false, more than 0 when it is true",
	}),
});

type SearchMatch = "terms" | "literal" | null;
type SearchOrder = "relevance" | "oldest" | "newest";

/**
 * The arguments the schemas above accept. Pi validates arguments against the schema before
 * `execute()`; TypeBox's `Static` cannot read the type of a `StringEnum`, so it is spelled out here.
 */
interface HistorySearchInput {
	query: string | null;
	match: SearchMatch;
	kinds: HistoryKind[];
	order: SearchOrder;
	limit: number;
	cursor: string | null;
}

interface HistoryReadInput {
	entryIds: string[];
	around: number;
	includeImages: boolean;
	maxImages: number;
}

// ---------------------------------------------------------------------------
// Effective history view
// ---------------------------------------------------------------------------

/** The latest context edit of each target on the branch; a later edit replaces an earlier one. */
function latestContextEdits(branch: readonly SessionEntry[]): Map<string, ContextEditEntry> {
	const edits = new Map<string, ContextEditEntry>();
	for (const entry of branch) {
		if (entry.type === "context_edit") edits.set(entry.targetId, entry);
	}
	return edits;
}

/**
 * The messages an entry contributes to model context after its latest context edit. Pi's public
 * projection applies the edit to detached copies, so session entries are never changed and
 * compaction boundaries do not hide older records.
 */
function projectEntry(entry: SessionEntry, edit: ContextEditEntry | undefined): AgentMessage[] {
	const source = { ...entry, parentId: null } as SessionEntry;
	const path: SessionEntry[] = edit ? [source, { ...edit, parentId: source.id }] : [source];
	return buildSessionProjection(path, path[path.length - 1].id).entries[0]?.messages ?? [];
}

/** Whether Pi sends the message to the model: prompt state, `!!` commands and failed or aborted attempts are left out. */
function isModelVisible(message: AgentMessage): boolean {
	switch (message.role) {
		case "system":
		case "compactionSummary":
			return false;
		case "bashExecution":
			return message.excludeFromContext !== true;
		case "assistant":
			return !isFailedAttempt(message);
		default:
			return true;
	}
}

/**
 * A failed or aborted attempt. Pi leaves it out of the context, but first gives every tool call that
 * has no result yet its `No result provided` result.
 */
function isFailedAttempt(message: AgentMessage): boolean {
	return message.role === "assistant" && (message.stopReason === "error" || message.stopReason === "aborted");
}

function contextMessages(entry: SessionEntry, edits: ReadonlyMap<string, ContextEditEntry>): AgentMessage[] {
	if (entry.type !== "message" && entry.type !== "custom_message" && entry.type !== "branch_summary") return [];
	return projectEntry(entry, edits.get(entry.id));
}

function visibleMessages(entry: SessionEntry, edits: ReadonlyMap<string, ContextEditEntry>): AgentMessage[] {
	return contextMessages(entry, edits).filter(isModelVisible);
}

/**
 * Content blocks as stored. Session files are not validated, so, as in Pi's own readers of stored
 * history, content that is not an array has no blocks. Items are read with `?.` and skipped unless
 * they have a known type, so an item that is not an object is skipped too; positions are kept.
 */
function storedBlocks<T>(content: readonly T[]): readonly T[] {
	return Array.isArray(content) ? content : [];
}

/** The content Pi sends for a message that reaches the model as a user message. */
function modelContent(message: AgentMessage): readonly (TextContent | ImageContent)[] {
	const [converted] = convertToLlm([message]);
	if (converted?.role !== "user") return [];
	return typeof converted.content === "string"
		? [{ type: "text", text: converted.content }]
		: storedBlocks(converted.content);
}

interface HistoryBlock {
	readonly blockIndex: number;
	readonly kind: HistoryKind;
	/** Searchable text. An image is its placeholder. */
	readonly text: string;
	readonly toolName: string | null;
	readonly toolCallId?: string;
	readonly arguments?: unknown;
	readonly image?: ImageContent;
}

interface HistoryRecord {
	readonly entry: SessionEntry;
	readonly branchIndex: number;
	readonly kind: HistoryKind;
	readonly toolResult?: { readonly toolName: string; readonly toolCallId: string; readonly isError: boolean };
	readonly blocks: readonly HistoryBlock[];
}

function addContent(
	blocks: HistoryBlock[],
	content: readonly (TextContent | ImageContent)[],
	offset: number,
	kind: HistoryKind,
	toolName: string | null,
): void {
	content.forEach((block, index) => {
		const blockIndex = offset + index;
		if (block?.type === "text" && typeof block.text === "string") {
			blocks.push({ blockIndex, kind, text: block.text, toolName });
		} else if (block?.type === "image") {
			blocks.push({ blockIndex, kind, text: `[image: ${block.mimeType}]`, toolName, image: block });
		}
	});
}

/**
 * A tool call as searchable text: its name, then each argument value on its own line, strings exactly
 * as stored and other values as JSON. Argument names are left out, so they do not match every call.
 */
function toolCallText(name: string, args: unknown): string {
	const lines = [name];
	const addValue = (value: unknown): void => {
		if (typeof value === "string") lines.push(value);
		else if (Array.isArray(value)) value.forEach(addValue);
		else if (typeof value === "object" && value !== null) Object.values(value).forEach(addValue);
		else if (value !== undefined) lines.push(JSON.stringify(value));
	};
	addValue(args);
	return lines.join("\n");
}

/** The searchable and readable part of one entry, or undefined when it has none. */
function toHistoryRecord(
	entry: SessionEntry,
	branchIndex: number,
	edits: ReadonlyMap<string, ContextEditEntry>,
): HistoryRecord | undefined {
	const blocks: HistoryBlock[] = [];
	let kind: HistoryKind | undefined;
	let toolResult: HistoryRecord["toolResult"];
	let offset = 0;
	for (const message of visibleMessages(entry, edits)) {
		switch (message.role) {
			case "assistant": {
				kind ??= "assistant";
				const content = storedBlocks(message.content);
				content.forEach((block, index) => {
					const blockIndex = offset + index;
					if (block?.type === "text" && typeof block.text === "string") {
						blocks.push({ blockIndex, kind: "assistant", text: block.text, toolName: null });
					} else if (block?.type === "toolCall" && !HISTORY_TOOLS.has(block.name)) {
						blocks.push({
							blockIndex,
							kind: "tool_call",
							text: toolCallText(block.name, block.arguments),
							toolName: block.name,
							toolCallId: block.id,
							arguments: block.arguments,
						});
					}
				});
				offset += content.length;
				break;
			}
			case "toolResult": {
				const content = storedBlocks(message.content);
				if (!HISTORY_TOOLS.has(message.toolName)) {
					kind ??= "tool_result";
					toolResult ??= { toolName: message.toolName, toolCallId: message.toolCallId, isError: message.isError };
					addContent(blocks, content, offset, "tool_result", message.toolName);
				}
				offset += content.length;
				break;
			}
			case "user":
			case "custom":
			case "bashExecution":
			case "branchSummary": {
				const content = modelContent(message);
				kind ??= KIND_BY_ROLE[message.role];
				addContent(blocks, content, offset, KIND_BY_ROLE[message.role], null);
				offset += content.length;
				break;
			}
		}
	}
	return kind !== undefined && blocks.length > 0 ? { entry, branchIndex, kind, toolResult, blocks } : undefined;
}

function buildHistory(range: readonly SessionEntry[], edits: ReadonlyMap<string, ContextEditEntry>): HistoryRecord[] {
	const records: HistoryRecord[] = [];
	range.forEach((entry, branchIndex) => {
		const record = toHistoryRecord(entry, branchIndex, edits);
		if (record) records.push(record);
	});
	return records;
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

function orphanResultError(entryId: string): Error {
	return new Error(`Entry ${entryId} is a tool result without a model-visible tool call`);
}

function failedAttemptError(entryId: string): Error {
	return new Error(`Entry ${entryId} is a failed or aborted attempt between tool calls and their results`);
}

/**
 * Where the kept fragment starts: the last model-visible message, or, when the branch ends with
 * tool results, the assistant message whose tool calls they answer. Missing results are not an
 * error, because Pi adds them when it sends the context. A failed or aborted attempt between the
 * calls and their results is an error, because Pi would then send two results for one call.
 */
function findFirstKeptEntryId(branch: readonly SessionEntry[]): string {
	const edits = latestContextEdits(branch);
	const trailingResults: { entryId: string; toolCallId: string }[] = [];
	let failedAttemptId: string | undefined;
	for (let index = branch.length - 1; index >= 0; index--) {
		const entry = branch[index];
		const messages = contextMessages(entry, edits);
		for (let position = messages.length - 1; position >= 0; position--) {
			const message = messages[position];
			if (!isModelVisible(message)) {
				if (trailingResults.length > 0 && isFailedAttempt(message)) failedAttemptId ??= entry.id;
				continue;
			}
			if (message.role === "toolResult") {
				trailingResults.push({ entryId: entry.id, toolCallId: message.toolCallId });
				continue;
			}
			const callIds = new Set(
				message.role === "assistant"
					? storedBlocks(message.content).flatMap((block) => (block?.type === "toolCall" ? [block.id] : []))
					: [],
			);
			const orphan = trailingResults.find((result) => !callIds.has(result.toolCallId));
			if (orphan) throw orphanResultError(orphan.entryId);
			if (failedAttemptId) throw failedAttemptError(failedAttemptId);
			return requireBranchEntry(branch, entry.id);
		}
	}
	if (trailingResults.length > 0) throw orphanResultError(trailingResults[0].entryId);
	throw new Error("No complete message to keep");
}

/** Pi keeps nothing for an ID it cannot find, so only an entry ID of the active branch is passed on. */
function requireBranchEntry(branch: readonly SessionEntry[], entryId: string): string {
	if (!branch.some((entry) => entry.id === entryId)) {
		throw new Error(`Entry ${entryId} is not on the active branch`);
	}
	return entryId;
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

const wordSegmenter = new Intl.Segmenter("zh-Hant", { granularity: "word" });
const CJK_CHARACTER = /^[\p{scx=Han}\p{scx=Hiragana}\p{scx=Katakana}\p{scx=Hangul}\p{scx=Bopomofo}]$/u;

/**
 * Word-like segments in order. Contiguous CJK characters become two-character pairs (one
 * character on its own stays a token); pairs never cross spaces, punctuation or other scripts.
 */
function tokenize(text: string): string[] {
	const tokens: string[] = [];
	let run: { chars: string[]; end: number } | undefined;
	const flushRun = (): void => {
		if (!run) return;
		const { chars } = run;
		if (chars.length === 1) tokens.push(chars[0]);
		for (let index = 0; index + 1 < chars.length; index++) tokens.push(chars[index] + chars[index + 1]);
		run = undefined;
	};
	for (const { segment, index, isWordLike } of wordSegmenter.segment(text)) {
		if (!isWordLike) continue;
		let word = "";
		let offset = index;
		for (const char of segment) {
			if (CJK_CHARACTER.test(char)) {
				if (word) {
					tokens.push(word);
					word = "";
				}
				if (!run || run.end !== offset) {
					flushRun();
					run = { chars: [], end: offset };
				}
				run.chars.push(char);
				run.end = offset + char.length;
			} else {
				flushRun();
				word += char;
			}
			offset += char.length;
		}
		if (word) tokens.push(word);
	}
	flushRun();
	return tokens;
}

interface Matcher {
	/** 0 when the text does not match. */
	score(text: string): number;
}

function createMatcher(query: string | null, match: SearchMatch): Matcher {
	if (query === null) return { score: () => 1 };
	if (match === "literal") return { score: (text) => (text.includes(query) ? 1 : 0) };
	const queryTokens = new Set(tokenize(query.normalize("NFKC")));
	const queryTokenList = [...queryTokens];
	return {
		score(text) {
			const normalized = text.normalize("NFKC");
			// A whole-token hit needs the token somewhere in the text; skip segmenting texts without any.
			if (!queryTokenList.some((token) => normalized.includes(token))) return 0;
			const found = new Set<string>();
			for (const token of tokenize(normalized)) {
				if (queryTokens.has(token)) found.add(token);
			}
			return found.size;
		},
	};
}

// ---------------------------------------------------------------------------
// history_search
// ---------------------------------------------------------------------------

type SortKey = readonly [score: number, branchIndex: number, blockIndex: number];

function compareKeys(order: SearchOrder, a: SortKey, b: SortKey): number {
	if (order === "relevance" && a[0] !== b[0]) return b[0] - a[0];
	if (a[1] !== b[1]) return order === "oldest" ? a[1] - b[1] : b[1] - a[1];
	return a[2] - b[2];
}

interface CursorState {
	readonly v: number;
	readonly session: string;
	readonly leaf: string;
	readonly query: string | null;
	readonly match: SearchMatch;
	readonly kinds: readonly HistoryKind[];
	readonly order: SearchOrder;
	readonly edits: string;
	readonly after: SortKey;
}

const INVALID_CURSOR = "Invalid cursor; start again with cursor null";

function isCursorState(value: unknown): value is CursorState {
	if (typeof value !== "object" || value === null) return false;
	const state = value as Record<string, unknown>;
	const { after, kinds } = state;
	return (
		state.v === CURSOR_VERSION &&
		typeof state.session === "string" &&
		typeof state.leaf === "string" &&
		(state.query === null || typeof state.query === "string") &&
		(state.match === null || state.match === "terms" || state.match === "literal") &&
		Array.isArray(kinds) &&
		kinds.every((kind) => (HISTORY_KINDS as readonly unknown[]).includes(kind)) &&
		(state.order === "relevance" || state.order === "oldest" || state.order === "newest") &&
		typeof state.edits === "string" &&
		Array.isArray(after) &&
		after.length === 3 &&
		after.every((value) => Number.isSafeInteger(value))
	);
}

/** Detects a changed cursor. It has no secret, so it does not stop a deliberate rewrite. */
function cursorCheck(body: string): string {
	return createHash("sha256").update(body).digest("base64url");
}

/** The state as base64url JSON, then `.` and the check of that text. */
function encodeCursor(state: CursorState): string {
	const body = Buffer.from(JSON.stringify(state), "utf8").toString("base64url");
	return `${body}.${cursorCheck(body)}`;
}

function decodeCursor(cursor: string): CursorState {
	const [body] = cursor.split(".", 1);
	if (cursor !== `${body}.${cursorCheck(body)}`) throw new Error(INVALID_CURSOR);
	let state: unknown;
	try {
		state = JSON.parse(Buffer.from(body, "base64url").toString("utf8"));
	} catch {
		throw new Error(INVALID_CURSOR);
	}
	if (!isCursorState(state)) throw new Error(INVALID_CURSOR);
	return state;
}

/** Identifies the visibility and replacement content that context edits give the entries in range. */
function contextEditFingerprint(range: readonly SessionEntry[], edits: ReadonlyMap<string, ContextEditEntry>): string {
	const hash = createHash("sha256");
	for (const entry of range) {
		const edit = edits.get(entry.id);
		if (edit) hash.update(`${JSON.stringify([entry.id, edit.replacement])}\n`);
	}
	return hash.digest("base64url");
}

function assertSearchMode(params: HistorySearchInput): void {
	if (params.query === null) {
		if (params.match !== null) throw new Error("match must be null when query is null");
		if (params.order === "relevance")
			throw new Error('order "relevance" needs a query; browse with "oldest" or "newest"');
		return;
	}
	if (params.query === "") throw new Error("query must not be empty; use null to browse");
	if (params.match === null) throw new Error('match must be "terms" or "literal" when query is set');
	if (params.match === "terms" && tokenize(params.query.normalize("NFKC")).length === 0) {
		throw new Error('query has no words to match with "terms"; use "literal"');
	}
}

function searchHistory(params: HistorySearchInput, sessionManager: SessionReader): string {
	assertSearchMode(params);
	const kinds = HISTORY_KINDS.filter((kind) => params.kinds.includes(kind));
	const sessionId = sessionManager.getSessionId();
	const branch = sessionManager.getBranch();
	const edits = latestContextEdits(branch);
	const cursor = params.cursor === null ? undefined : decodeCursor(params.cursor);
	let leafIndex = branch.length - 1;
	if (cursor) {
		if (cursor.session !== sessionId)
			throw new Error("Cursor belongs to another session; start again with cursor null");
		if (
			cursor.query !== params.query ||
			cursor.match !== params.match ||
			cursor.order !== params.order ||
			cursor.kinds.join() !== kinds.join()
		) {
			throw new Error(
				"Cursor belongs to a different search; repeat its query, match, kinds and order, or start again with cursor null",
			);
		}
		leafIndex = branch.findIndex((entry) => entry.id === cursor.leaf);
		if (leafIndex < 0)
			throw new Error("The active branch changed since the first page; start again with cursor null");
	}
	const range = branch.slice(0, leafIndex + 1);
	const fingerprint = contextEditFingerprint(range, edits);
	if (cursor && cursor.edits !== fingerprint) {
		throw new Error("Context edits changed the searched records since the first page; start again with cursor null");
	}

	const matcher = createMatcher(params.query, params.match);
	const hits: { record: HistoryRecord; block: HistoryBlock; key: SortKey }[] = [];
	for (const record of buildHistory(range, edits)) {
		for (const block of record.blocks) {
			if (!kinds.includes(block.kind)) continue;
			// Images are never searched; browsing lists their blocks too.
			if (block.image && params.query !== null) continue;
			const score = matcher.score(block.text);
			if (score > 0) hits.push({ record, block, key: [score, record.branchIndex, block.blockIndex] });
		}
	}
	hits.sort((a, b) => compareKeys(params.order, a.key, b.key));
	let start = 0;
	if (cursor) {
		// The position must be a result of this search; any other position would skip or repeat results.
		start = hits.findIndex(({ key }) => key.every((value, index) => value === cursor.after[index])) + 1;
		if (start === 0) throw new Error(INVALID_CURSOR);
	}
	const remaining = hits.slice(start);
	const page = remaining.slice(0, params.limit);
	const last = page.at(-1);
	const leaf = branch[leafIndex];
	const nextCursor =
		last && leaf && remaining.length > page.length
			? encodeCursor({
					v: CURSOR_VERSION,
					session: sessionId,
					leaf: leaf.id,
					query: params.query,
					match: params.match,
					kinds,
					order: params.order,
					edits: fingerprint,
					after: last.key,
				})
			: null;

	return JSON.stringify({
		query: params.query,
		match: params.match,
		totalMatches: hits.length,
		matches: page.map(({ record, block }) => ({
			entryId: record.entry.id,
			blockIndex: block.blockIndex,
			timestamp: record.entry.timestamp,
			kind: block.kind,
			toolName: block.toolName,
			derived: record.kind === "branch_summary",
		})),
		nextCursor,
	});
}

// ---------------------------------------------------------------------------
// history_read
// ---------------------------------------------------------------------------

/** Only the active branch is consulted: an ID that is not on it gets the same error wherever it came from. */
function unreadableEntryError(entryId: string, branch: readonly SessionEntry[]): Error {
	if (!branch.some((entry) => entry.id === entryId)) return new Error(`Entry ${entryId} is not on the active branch`);
	return new Error(`Entry ${entryId} has no readable content`);
}

function readHistory(params: HistoryReadInput, sessionManager: SessionReader): (TextContent | ImageContent)[] {
	if (!params.includeImages && params.maxImages !== 0)
		throw new Error("maxImages must be 0 when includeImages is false");
	if (params.includeImages && params.maxImages === 0) {
		throw new Error("maxImages must be greater than 0 when includeImages is true");
	}
	const branch = sessionManager.getBranch();
	const records = buildHistory(branch, latestContextEdits(branch));
	const positions = new Set<number>();
	for (const entryId of params.entryIds) {
		const position = records.findIndex((record) => record.entry.id === entryId);
		if (position < 0) throw unreadableEntryError(entryId, branch);
		const last = Math.min(records.length - 1, position + params.around);
		for (let index = Math.max(0, position - params.around); index <= last; index++) positions.add(index);
	}

	const images: ImageContent[] = [];
	const lines = [...positions]
		.sort((a, b) => a - b)
		.map((position) => {
			const { entry, kind, toolResult, blocks } = records[position];
			return JSON.stringify({
				entryId: entry.id,
				timestamp: entry.timestamp,
				kind,
				derived: kind === "branch_summary",
				...(toolResult && {
					toolName: toolResult.toolName,
					toolCallId: toolResult.toolCallId,
					isError: toolResult.isError,
				}),
				blocks: blocks.map((block) => {
					const base = { blockIndex: block.blockIndex, kind: block.kind };
					if (block.image) {
						if (!params.includeImages || images.length >= params.maxImages) return { ...base, image: block.text };
						const { data, mimeType } = block.image;
						if (typeof data !== "string" || data === "" || typeof mimeType !== "string" || mimeType === "") {
							throw new Error(`Image in block ${block.blockIndex} of entry ${entry.id} is unavailable`);
						}
						images.push({ type: "image", data, mimeType });
						return { ...base, image: block.text, attachment: images.length };
					}
					if (block.kind === "tool_call") {
						return {
							...base,
							toolName: block.toolName,
							toolCallId: block.toolCallId,
							arguments: block.arguments,
						};
					}
					return { ...base, text: block.text };
				}),
			});
		});
	return [{ type: "text", text: lines.join("\n") }, ...images];
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function noCompact(pi: ExtensionAPI): void {
	pi.on("session_before_compact", async (event, ctx) => {
		try {
			const firstKeptEntryId = findFirstKeptEntryId(ctx.sessionManager.getBranch());
			return {
				compaction: {
					summary: CONTINUATION_INSTRUCTION,
					firstKeptEntryId,
					tokensBefore: event.preparation.tokensBefore,
				},
			};
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			ctx.ui.notify(`Compaction failed: ${message}`, "error");
			ctx.abort();
			return { cancel: true };
		}
	});

	pi.registerTool({
		name: HISTORY_SEARCH,
		label: "History Search",
		description:
			"Search or browse earlier records of the current conversation (this session's active branch), including records no longer in context. " +
			'Browse: query null, match null, order "oldest" or "newest". Search: a non-empty query with match "terms" or "literal", any order. Matching is case-sensitive. ' +
			"A tool call is searched as its name and argument values. Images are never searched; browsing lists their blocks too. " +
			"Each match is one block of a record and only says where it is: entryId, blockIndex, timestamp, kind, toolName and derived (true for branch summaries). Matches carry no text. " +
			"The first page fixes the searched records; pass nextCursor with the same query, match, kinds and order until nextCursor is null. Read the content with history_read.",
		promptSnippet: "Search or browse earlier records of this conversation",
		promptGuidelines: [
			"Use history_search instead of guessing when a detail from earlier in this conversation is not in context.",
		],
		parameters: historySearchParameters,
		annotations: READ_ONLY_ANNOTATIONS,
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const text = searchHistory(params as HistorySearchInput, ctx.sessionManager);
			return { content: [{ type: "text", text }], details: undefined };
		},
	});

	pi.registerTool({
		name: HISTORY_READ,
		label: "History Read",
		description:
			"Read whole records of the current conversation (this session's active branch) by entryId, with their exact content. " +
			"Returns one JSON object per line in branch order: entryId, timestamp, kind, derived, toolName, toolCallId and isError for tool results, and blocks with text, tool calls, or image placeholders ([image: <mimeType>]). " +
			"around adds that many readable records before and after each requested record; overlapping ranges are merged. " +
			"With includeImages true, the first maxImages images are attached after the text, in order, and the rest stay placeholders.",
		promptSnippet: "Read earlier records of this conversation exactly",
		promptGuidelines: ["Use history_read to read the records that history_search finds; its matches carry no text."],
		parameters: historyReadParameters,
		annotations: READ_ONLY_ANNOTATIONS,
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			return { content: readHistory(params as HistoryReadInput, ctx.sessionManager), details: undefined };
		},
	});
}
