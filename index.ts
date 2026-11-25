/**
 * pi-no-compact: Minimal-tail Retrieval Compaction for Pi.
 *
 * Compaction writes no LLM summary and keeps no recent tail. The new context holds a fixed
 * instruction and the last structurally complete fragment of the active branch. Everything else
 * stays in the session and is retrieved on demand with `history_search` and `history_read`.
 */

import { createHash, randomUUID } from "node:crypto";
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

/**
 * Whether Pi sends the message to the model as part of the conversation, each read as Pi reads it.
 * Left out are `!!` commands, failed or aborted attempts, messages of a role Pi does not send, and
 * system messages: they hold the system prompt, and Pi moves one between tool calls and their
 * results to after the results.
 */
function isModelVisible(message: AgentMessage): boolean {
	switch (message.role) {
		case "user":
		case "toolResult":
		case "custom":
		case "branchSummary":
		case "compactionSummary":
			return true;
		case "bashExecution":
			return !message.excludeFromContext;
		case "assistant":
			return !isFailedAttempt(message);
		default:
			return false;
	}
}

/** A text block with text other than whitespace. */
function isNonEmptyText(block: unknown): boolean {
	return isObject(block) && block.type === "text" && typeof block.text === "string" && block.text.trim() !== "";
}

/**
 * Whether the message gives the model text other than whitespace, an image or a tool call: the
 * plugin's own rule for where the kept fragment may start. Provider adapters differ on a message
 * without any, such as an empty reply, and may leave it out of the request. Pi always gives `!`
 * commands and summaries text, and tool results are paired with their calls instead.
 */
function hasContent(message: AgentMessage): boolean {
	switch (message.role) {
		case "assistant":
			return storedBlocks(message.content).some((block) => block?.type === "toolCall" || isNonEmptyText(block));
		case "user":
		case "custom":
			return modelContent(message).some((block) => block?.type === "image" || isNonEmptyText(block));
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

/** A stored JSON object, as opposed to null, an array or another value. */
function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Whether the entry can be projected and returned: its ID and timestamp are strings, and its latest
 * context edit omits it or replaces its content. Other values would be returned with another type or
 * make Pi's projection throw.
 */
function hasEntryFields(entry: SessionEntry, edit: ContextEditEntry | undefined): boolean {
	return (
		typeof entry.id === "string" &&
		typeof entry.timestamp === "string" &&
		(edit === undefined || edit.replacement === null || isObject(edit.replacement))
	);
}

/**
 * Whether the values Pi turns into the text of a `!` command or a branch summary have the types Pi
 * declares. A missing exit code or output path may also be null, which Pi reads as absent.
 */
function hasTextFields(message: AgentMessage): boolean {
	switch (message.role) {
		case "bashExecution":
			return (
				typeof message.command === "string" &&
				typeof message.output === "string" &&
				(message.exitCode == null || typeof message.exitCode === "number") &&
				(message.fullOutputPath == null || typeof message.fullOutputPath === "string")
			);
		case "branchSummary":
			return typeof message.summary === "string";
		default:
			return true;
	}
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
		} else if (block?.type === "image" && typeof block.data === "string" && typeof block.mimeType === "string") {
			blocks.push({ blockIndex, kind, text: `[image: ${block.mimeType}]`, toolName, image: block });
		}
	});
}

/**
 * A tool call as searchable text: its name, then each argument value on its own line, strings exactly
 * as stored and other values as JSON. Argument names are left out, so they do not match every call.
 * Values are visited in order with a list instead of recursion, so no nesting depth exhausts the stack.
 */
function toolCallText(name: string, args: Record<string, unknown>): string {
	const lines = [name];
	const pending: unknown[] = [args];
	while (pending.length > 0) {
		const value = pending.pop();
		if (typeof value === "string") lines.push(value);
		else if (typeof value === "object" && value !== null) {
			const children = Array.isArray(value) ? value : Object.values(value);
			for (let index = children.length - 1; index >= 0; index--) pending.push(children[index]);
		} else if (value !== undefined) lines.push(JSON.stringify(value));
	}
	return lines.join("\n");
}

/**
 * The JSON text `JSON.stringify` writes for stored JSON data, built with a list instead of recursion,
 * so no nesting depth exhausts the stack and a record nested deeper than the runtime's own serializer
 * allows is still returned whole.
 */
function toJson(value: unknown): string {
	const parts: string[] = [];
	// Values still to write, and the text that separates or closes them.
	const pending: ({ value: unknown } | { text: string })[] = [{ value }];
	for (let item = pending.pop(); item !== undefined; item = pending.pop()) {
		if ("text" in item) {
			parts.push(item.text);
			continue;
		}
		const current = item.value;
		if (Array.isArray(current)) {
			parts.push("[");
			pending.push({ text: "]" });
			for (let index = current.length - 1; index >= 0; index--) {
				pending.push({ value: current[index] ?? null });
				if (index > 0) pending.push({ text: "," });
			}
		} else if (typeof current === "object" && current !== null) {
			parts.push("{");
			pending.push({ text: "}" });
			const fields = Object.entries(current).filter(([, field]) => field !== undefined);
			for (let index = fields.length - 1; index >= 0; index--) {
				const [key, field] = fields[index];
				pending.push({ value: field }, { text: `${index > 0 ? "," : ""}${JSON.stringify(key)}:` });
			}
		} else {
			parts.push(JSON.stringify(current));
		}
	}
	return parts.join("");
}

/** The searchable and readable part of one entry, or undefined when it has none. */
function toHistoryRecord(
	entry: SessionEntry,
	branchIndex: number,
	edits: ReadonlyMap<string, ContextEditEntry>,
): HistoryRecord | undefined {
	if (!hasEntryFields(entry, edits.get(entry.id))) return undefined;
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
					} else if (
						block?.type === "toolCall" &&
						typeof block.id === "string" &&
						typeof block.name === "string" &&
						isObject(block.arguments) &&
						!HISTORY_TOOLS.has(block.name)
					) {
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
				if (
					typeof message.toolCallId === "string" &&
					typeof message.toolName === "string" &&
					typeof message.isError === "boolean" &&
					!HISTORY_TOOLS.has(message.toolName)
				) {
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
				const content = hasTextFields(message) ? modelContent(message) : [];
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

function missingCallError(entryId: string, resultEntryId: string): Error {
	return new Error(`Entry ${entryId} has no tool call for the tool result in entry ${resultEntryId}`);
}

function failedAttemptError(entryId: string): Error {
	return new Error(`Entry ${entryId} is a failed or aborted attempt between tool calls and their results`);
}

function repeatedCallError(entryId: string, toolCallId: string): Error {
	return new Error(`Entry ${entryId} has two tool calls with ID ${toolCallId}`);
}

function secondResultError(entryId: string, toolCallId: string): Error {
	return new Error(`Entry ${entryId} is a second result for tool call ${toolCallId}`);
}

/**
 * Where the kept fragment starts: the last model-visible message with content, or, when tool results
 * follow it, the assistant message whose tool calls they answer, right before them. Missing results
 * are not an error, because Pi adds them when it sends the context. Any other message between the
 * calls and their results is an error, even one without content, and so is a failed or aborted
 * attempt there, because Pi would then send two results for one call. So are two tool calls with one
 * ID and a second result for one call, which Pi's conversion keeps, and a kept entry whose latest
 * context edit comes before the fragment.
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
				const later = trailingResults.find((result) => result.toolCallId === message.toolCallId);
				if (later) throw secondResultError(later.entryId, message.toolCallId);
				trailingResults.push({ entryId: entry.id, toolCallId: message.toolCallId });
				continue;
			}
			// A message without content is kept but cannot start the fragment; between calls and their
			// results it still closes the calls, so it is checked like any other message.
			if (trailingResults.length === 0 && !hasContent(message)) continue;
			const calls =
				message.role === "assistant"
					? storedBlocks(message.content).flatMap((block) => (block?.type === "toolCall" ? [block.id] : []))
					: [];
			const repeated = calls.findIndex((id, position) => calls.indexOf(id) !== position);
			if (repeated !== -1) throw repeatedCallError(entry.id, calls[repeated]);
			const callIds = new Set(calls);
			const orphan = trailingResults.find((result) => !callIds.has(result.toolCallId));
			if (orphan) throw missingCallError(entry.id, orphan.entryId);
			if (failedAttemptId) throw failedAttemptError(failedAttemptId);
			const firstKeptEntryId = requireBranchEntry(branch, entry.id);
			requireEditsInFragment(branch, index, edits);
			return firstKeptEntryId;
		}
	}
	if (trailingResults.length > 0) throw orphanResultError(trailingResults[0].entryId);
	throw new Error("No complete message to keep");
}

/**
 * After a compaction Pi applies only the context edits stored in the kept fragment. So every kept
 * entry that Pi turns into messages, one an edit hides included, must have its latest edit inside the
 * fragment: Pi would stop applying an earlier one, and the original content could return or change
 * how tool calls and their results pair up. Pi leaves kept system messages out after a compaction.
 */
function requireEditsInFragment(
	branch: readonly SessionEntry[],
	start: number,
	edits: ReadonlyMap<string, ContextEditEntry>,
): void {
	for (let index = start; index < branch.length; index++) {
		const entry = branch[index];
		const contributes =
			entry.type === "custom_message" ||
			entry.type === "branch_summary" ||
			(entry.type === "message" && entry.message.role !== "system");
		const edit = contributes ? edits.get(entry.id) : undefined;
		if (edit && branch.indexOf(edit) < start) {
			throw new Error(`Entry ${entry.id} has its latest context edit in entry ${edit.id}, before the kept fragment`);
		}
	}
}

/**
 * Pi takes the start as a string entry ID and keeps nothing for an ID it cannot find, so only a string
 * ID of an entry on the active branch is passed on. Another kind of ID, which Pi never writes, is not
 * shown in the message, because it may not convert to text.
 */
function requireBranchEntry(branch: readonly SessionEntry[], entryId: unknown): string {
	if (typeof entryId !== "string") {
		throw new Error("The kept fragment would start at an entry whose ID is not a string");
	}
	if (!branch.some((entry) => entry.id === entryId)) {
		throw new Error(`Entry ${entryId} is not on the active branch`);
	}
	return entryId;
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/** Word boundaries in the runtime's default language; the plugin fixes no language. */
const wordSegmenter = new Intl.Segmenter(undefined, { granularity: "word" });
/** A CJK character that is not punctuation or a symbol, so punctuation such as U+30A0 ends a run. */
const CJK_CHARACTER = /^(?![\p{P}\p{S}])[\p{scx=Han}\p{scx=Hiragana}\p{scx=Katakana}\p{scx=Hangul}\p{scx=Bopomofo}]$/u;
/** A token is a word only with a letter or a number in it; punctuation alone never is. */
const WORD_CHARACTER = /[\p{L}\p{N}]/u;

/**
 * Passes the tokens of the text to visit, in order, and returns as soon as visit returns true, without
 * taking another segment from the segmenter. Tokens are the word-like segments, except that contiguous
 * CJK characters become two-character pairs, each passed as soon as its second character is read; a
 * CJK character without a neighbour is a token as soon as its run ends, at the first space,
 * punctuation, symbol, other script or end of text after it. Pairs never cross spaces, punctuation or
 * other scripts, and tokens without a letter or a number are left out.
 */
function visitTokens(text: string, visit: (token: string) => boolean): void {
	let stopped = false;
	const add = (token: string): void => {
		if (!stopped && WORD_CHARACTER.test(token)) stopped = visit(token);
	};
	// The open run of contiguous CJK characters: its last character, where that character ends (read
	// only while a run is open) and whether the run has made a pair.
	let last = "";
	let end = 0;
	let paired = false;
	const endRun = (): void => {
		if (last && !paired) add(last);
		last = "";
		paired = false;
	};
	for (const { segment, index, isWordLike } of wordSegmenter.segment(text)) {
		if (!isWordLike) {
			// A space, punctuation or symbol ends the open run, so a lone character there is a token now.
			endRun();
			if (stopped) return;
			continue;
		}
		let word = "";
		let offset = index;
		for (const char of segment) {
			if (CJK_CHARACTER.test(char)) {
				if (word) {
					add(word);
					word = "";
				}
				if (last && end === offset) {
					add(last + char);
					paired = true;
				} else {
					endRun();
				}
				last = char;
				end = offset + char.length;
			} else {
				endRun();
				word += char;
			}
			if (stopped) return;
			offset += char.length;
		}
		if (word) add(word);
		if (stopped) return;
	}
	endRun();
}

/** Every token of the text, in order; see visitTokens. */
function tokenize(text: string): string[] {
	const tokens: string[] = [];
	visitTokens(text, (token) => {
		tokens.push(token);
		return false;
	});
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
			// Once every query token is found the score cannot grow: token processing stops and no further
			// segments are taken from the segmenter.
			visitTokens(normalized, (token) => {
				if (queryTokens.has(token)) found.add(token);
				return found.size === queryTokens.size;
			});
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
	readonly session: string;
	/** The ID of the snapshot's last entry. */
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

/** Tells entry IDs apart for a cursor; see createEntryIdentity. */
type EntryIdentity = (id: unknown) => unknown;

/**
 * Tells entry IDs apart as Pi does, by the ID value itself. A string stays that string, so normal
 * fingerprints never change. Another value, which only damaged or non-standard session data has,
 * becomes a JSON array that no string equals: its type and value, so Infinity and -Infinity differ,
 * or for an object or array the object itself, numbered in memory. Each plugin instance numbers
 * objects anew, so its random nonce keeps those numbers apart from any earlier instance's. Nothing of
 * this is written to a file or the session.
 */
function createEntryIdentity(): EntryIdentity {
	const nonce = randomUUID();
	const numbers = new WeakMap<object, number>();
	let count = 0;
	return (id) => {
		if (typeof id === "string") return id;
		if (typeof id !== "object" || id === null) return [id === null ? "null" : typeof id, String(id)];
		let number = numbers.get(id);
		if (number === undefined) {
			count += 1;
			number = count;
			numbers.set(id, number);
		}
		return ["object", nonce, number];
	};
}

/**
 * Identifies the latest context edit of each entry in range by the edit's own entry ID, told apart as
 * Pi tells entries apart. Pi never changes an entry, so another edit means a new edit or another
 * branch; the replacement is not read.
 */
function contextEditFingerprint(
	range: readonly SessionEntry[],
	edits: ReadonlyMap<string, ContextEditEntry>,
	identity: EntryIdentity,
): string {
	const hash = createHash("sha256");
	for (const entry of range) {
		const edit = edits.get(entry.id);
		if (edit) hash.update(`${toJson([identity(entry.id), identity(edit.id)])}\n`);
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

function searchHistory(params: HistorySearchInput, sessionManager: SessionReader, identity: EntryIdentity): string {
	assertSearchMode(params);
	const kinds = HISTORY_KINDS.filter((kind) => params.kinds.includes(kind));
	const sessionId = sessionManager.getSessionId();
	const branch = sessionManager.getBranch();
	const edits = latestContextEdits(branch);
	const cursor = params.cursor === null ? undefined : decodeCursor(params.cursor);
	// The snapshot ends at the last entry whose ID is a string, normally the leaf, because the cursor
	// names it by that ID. Every entry after it has another kind of ID and is never readable.
	let endIndex = branch.findLastIndex((entry) => typeof entry.id === "string");
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
		endIndex = branch.findIndex((entry) => entry.id === cursor.leaf);
		if (endIndex < 0) throw new Error("The active branch changed since the first page; start again with cursor null");
	}
	const range = branch.slice(0, endIndex + 1);
	const fingerprint = contextEditFingerprint(range, edits, identity);
	if (cursor && cursor.edits !== fingerprint) {
		throw new Error(
			"A searched entry has a different latest context edit than on the first page; start again with cursor null",
		);
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
	const end = branch[endIndex];
	const nextCursor =
		last && end && remaining.length > page.length
			? encodeCursor({
					session: sessionId,
					leaf: end.id,
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
			return toJson({
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
						if (data === "" || mimeType === "") {
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
	// Shared by every page of this plugin instance; a new instance starts its own.
	const identity = createEntryIdentity();

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
			const text = searchHistory(params as HistorySearchInput, ctx.sessionManager, identity);
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
