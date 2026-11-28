# pi-no-compact

## Introduction

pi-no-compact is a Pi extension that replaces what Pi's compaction keeps. When Pi compacts — `/compact`, the context threshold, or context overflow recovery — no LLM summary is written and Pi's recent tail is not kept. The new context holds only a fixed instruction and the last structurally complete fragment of the active branch. That fragment starts at the last message the model can see, apart from tool results, that has text other than whitespace, an image or a tool call, and runs to the end of the branch. When tool results come after that message, it must be the assistant message that made those tool calls, directly before its results. A tool call without a result is kept as it is; Pi adds its `No result provided` result when it sends the context.

A message without such content, such as an empty reply, never starts the fragment, because provider adapters may leave it out of the request; when it comes later, it is still kept. This is the plugin's rule for the start, not a description of what every provider sends.

Everything else stays in the session and is retrieved on demand with two tools, `history_search` and `history_read`. The fixed instruction, stored as the compaction summary, is:

```text
Context was compacted. Earlier records are no longer in your context but remain available through history_search and history_read.
Before continuing, read earlier messages, then the latest records.
For any other missing detail, search instead of guessing, and use history_read for exact text.
```

When no complete fragment can be kept, the plugin reports `Compaction failed: <reason>` as a Pi error notification, stops the current operation and cancels the compaction. For example, a tool result may have no tool call the model can see, two tool calls may share an ID, one tool call may have a second result, or the entry where the fragment would start may have an ID that is not a string. Any other message the model can see between tool calls and their results also cancels it, even an empty one, and so does a failed or aborted attempt there, because Pi would send both its `No result provided` result and the real result. It is also cancelled when an entry kept in the fragment has its latest context edit stored before the fragment: after the compaction Pi applies only the edits stored in the fragment, so the original content could return or change how tool calls and their results pair up. It never falls back to Pi's built-in compaction.

## Usage

Install, update and uninstall the plugin with Pi's git package commands. It is downloaded only through git, never from npm. `pi install` and `pi update` get the latest commit of main, and `pi uninstall`, the same as `pi remove`, removes the plugin:

```bash
pi install git:github.com/caeloren/pi-no-compact
pi update git:github.com/caeloren/pi-no-compact
pi uninstall git:github.com/caeloren/pi-no-compact
```

The repository has no tags. To stay on one version, install a commit by its full 40-character hash:

```bash
pi install git:github.com/caeloren/pi-no-compact@<full-commit-hash>
```

`pi update` then keeps that commit. Installing another full hash moves to that commit, and installing without a hash returns to the latest commit of main. Pi cannot update a shortened hash, so always use the full one.

Both tools are active as soon as the plugin loads. Every parameter is required; there are no defaults.

`history_search` searches or browses earlier records of the current conversation:

| Parameter | Type | Meaning |
|---|---|---|
| `query` | string or `null` | Text to search for. `null` browses. |
| `match` | `"terms"`, `"literal"` or `null` | How `query` is matched. `null` when browsing. |
| `kinds` | array, at least one of `user`, `assistant`, `tool_call`, `tool_result`, `bash`, `custom_message`, `branch_summary` | Record kinds to include. |
| `order` | `"relevance"`, `"oldest"` or `"newest"` | `relevance` only when searching. |
| `limit` | integer, at least 1 | Results per page. The plugin sets no maximum. |
| `cursor` | string or `null` | `null` for the first page, then the previous page's `nextCursor`. |

Browse with `query: null`, `match: null` and `order` `"oldest"` or `"newest"`. Search with a non-empty `query`, `match` `"terms"` or `"literal"` and any `order`. Other combinations are rejected, and so is a `terms` query without any word, such as `!!!`; use `literal` for it.

Each block is searched as the text the model sees. `!` commands and branch summaries include the text Pi wraps around them. A tool call is its name followed by each argument value on its own line, nested values one by one, strings exactly as stored and other values as JSON, without argument names. Images are never searched; browsing lists their blocks too.

- `literal` finds an exact, contiguous, case-sensitive substring of that text. Use it for paths, error codes, function and class names, commit hashes and whole error messages.
- `terms` applies NFKC normalization (case is kept), splits the text into words with `Intl.Segmenter` in the runtime's default language, keeps word-like segments, and turns each run of contiguous CJK characters into overlapping two-character tokens, so a run of seven characters gives six tokens; a lone CJK character stays one token. Punctuation and symbols are never CJK characters, even those of CJK scripts such as U+30A0, so pairs never cross them, spaces, Latin text or separate runs. A token needs at least one letter or number: punctuation or symbols alone are never a word. Query tokens are deduplicated and compared as whole, case-sensitive tokens, and a block matches when it contains any of them: `data` does not match `database`, and `schema` does not match `schemas`. Use `literal` for substrings.
- Case is never folded or substituted: `ErrorCode` and `errorcode` are different. No results is a normal, successful answer.

Results are blocks of records, ordered by their position on the branch, never by timestamp. `relevance` sorts by the number of distinct query tokens found, then newest first; with `literal` every hit counts once, so `relevance` is newest first. With `terms`, the query tokens found as substrings of a block's NFKC text are counted first: a token that is not there cannot match, and with none the block is not split at all. Once every one of them is found as a whole token the count cannot grow, so token processing stops and no further segments of that block are taken from the segmenter. A substring alone is not a match, and results are the same. `oldest` and `newest` follow the branch, and blocks of one record keep their order.

The result is one JSON object: `query`, `match`, `totalMatches` (matching blocks in the snapshot), `matches` and `nextCursor`. Each match only says where a block is: `entryId`, `blockIndex`, `timestamp`, `kind`, `toolName` and `derived` (true for branch summaries). Matches carry no text; read the content with `history_read`.

The first page fixes a snapshot: the session, the last entry of the active branch whose ID is a string (normally its leaf), the entries up to that entry, `query`, `match`, `kinds` and `order`. Entries added later are not part of that search. `nextCursor` holds the sort position of the page's last result, not an offset, and a SHA-256 check of its content, so pages never repeat or skip a result. Every page applies the latest context edits. A cursor fails, and the search starts again with `cursor: null`, when it was changed, when its position is not a result of the search, when it comes from another session, when the active branch no longer contains the snapshot's last entry, when an entry in the snapshot has a different latest context edit than on the first page, even one with the same content, or when it is used with another `query`, `match`, `kinds` or `order`; `kinds` is compared as a set, so its order and repeats do not matter. Edits are told apart by their entry IDs, as Pi tells entries apart. An ID that is an object or array, which only damaged or non-standard session data has, is known only to the plugin instance that read it, so a cursor that depends on one also fails once the plugin instance or those entries are created again, for example after Pi restarts. `limit` may change between pages. Page until `nextCursor` is `null` to list every candidate.

`history_read` reads whole records by `entryId`:

| Parameter | Type | Meaning |
|---|---|---|
| `entryIds` | array of strings, at least one | Records to read, such as `entryId` values from `history_search`. |
| `around` | integer, at least 0 | Readable records to add before and after each requested record. |
| `includeImages` | boolean | `true` attaches images; `false` shows placeholders only. |
| `maxImages` | integer, at least 0 | Images to attach: `0` when `includeImages` is `false`, more than `0` when it is `true`. |

It returns one JSON object per line, in branch order: `entryId`, `timestamp`, `kind`, `derived`, `toolName`, `toolCallId` and `isError` for tool results, and `blocks` with their `blockIndex`, `kind` and the full `text` the model sees, the tool call (`toolName`, `toolCallId` and the exact `arguments`) or `image`, the image placeholder `[image: <mimeType>]`. Overlapping `around` ranges are merged. With `includeImages: true`, the first `maxImages` images are attached after the text as image content, in order, and their `attachment` gives that position, starting at 1; the rest stay placeholders.

Argument types, required fields and ranges are checked by Pi before the tools run, with Pi's own `Validation failed for tool ...` errors. Pi first converts values that can be converted, for example `1.5` becomes `1`. Other problems are tool errors, such as `Entry <id> is not on the active branch`, `Entry <id> has no readable content` or an invalid cursor.

## Notes and limitations

- Both tools see only the current session and its active branch. Other sessions, conversations and branches are never read, and there are no parameters to reach them. Every `entryId` is checked: one that is not on the active branch gets the same error whether it is on another branch, in another session or does not exist.
- The tools return only what the model was entitled to see: user messages, assistant text and tool calls, tool results, visible `!` commands, custom messages and branch summaries, after the latest context edit of each entry. A context edit that omits an entry hides it from both tools; a replacement shows only the replaced content. Thinking, signatures, the system prompt, tool declarations, usage, `details`, `!!` commands, failed or aborted attempts, compaction summaries, and the calls and results of `history_search` and `history_read` themselves are never searched or read.
- Session files are not validated, so a record may hold malformed content. As in Pi's own readers of stored history, a string is read as text where Pi allows one, in user and custom messages; any other content that is not an array counts as empty, and an item that is not an object is skipped. A stored value whose type is not the one Pi declares makes only the smallest part that holds it unreadable, and every other record can still be searched and read:
  - a block: a text block's `text`, an image's `data` or `mimeType`, or a tool call's `id`, `name` or `arguments`;
  - a message: a tool result's `toolCallId`, `toolName` or `isError`, a `!` command's `command`, `output`, `exitCode`, `fullOutputPath` or `fullOutputError`, or a branch summary's `summary`. `exitCode`, `fullOutputPath` and `fullOutputError` may also be null, which Pi reads as absent;
  - the whole entry: an `id` or `timestamp` that is not a string, or a latest context edit whose `replacement` is neither null nor an object.
- While a malformed record is still in the context, Pi itself can fail on it. Apart from where the kept fragment starts and how tool calls and results pair up in it, the plugin does not check what Pi sends.
- The plugin sets no output limits: no byte or line cap, no maximum for `limit`, `entryIds`, `around` or `maxImages`, and `history_read` never truncates or splits a record. Nested values are read without recursion, so their depth alone stops neither tool. Model, provider and runtime limits still apply, so a very large result, or a very large kept fragment, can make the next request fail.
- The cursor check has no secret key. It catches a cursor changed by mistake, but a cursor rewritten on purpose to the position of another result, with its check recomputed, is accepted. That can only skip or repeat results of that search; it reaches nothing the tools would not return anyway.
- A cursor is not guaranteed to stay valid or to return correct results across plugin updates. After updating the plugin, start every search again with `cursor: null`.
- Whether images reach the model also depends on the model's image support, Pi's image settings, the branch and context edits.
- The plugin stores no data of its own: no transcript copy, cache, database, persistent index, embeddings, remote service or background work. Its only state is held in memory: a random value and the numbers that tell apart context edits whose IDs are objects or arrays, never written to a file or the session. It reads the session through Pi's API on each call and never changes, deletes or rewrites it. A compaction appends Pi's normal compaction entry.
- Pi decides when to compact, and its own checks, such as "Nothing to compact", still apply. Instructions passed to `/compact` are ignored, because no summary is written.
- Pi shows notifications only where it has a UI. The interactive mode shows `Compaction failed: <reason>` as an error, and RPC clients receive it as a `notify` request. Print (`-p`) and JSON modes have no UI, so the reason is not shown there. The compaction is still cancelled and the operation stopped, so a prompt that needed compaction first is not sent: print mode ends that run with Pi's abort error, and in JSON mode the `compaction_end` event has `aborted: true` and the run ends with an `agent_settled` event that has `aborted: true`.
- When the plugin is not loaded — disabled, removed, or failed to load — Pi may use its built-in compaction.
- Pi runs the `session_before_compact` handlers of all extensions in load order. If an extension loaded after this plugin also returns a result, Pi uses that result instead, which can mean Pi's built-in compaction. A cancel from this plugin is final.
- After the plugin is removed, earlier compaction entries still contain the fixed instruction, which names `history_search` and `history_read`. Those tools no longer exist, so calls to them fail with Pi's `Tool ... not found` error. Removing the plugin does not change or damage the session; it stays readable and usable.
- The plugin is downloaded only through git, and the package has only four files: `LICENSE`, `README.md`, `package.json` and `index.ts`. It has no dependencies, so the dependency install that Pi runs in the checkout has nothing to download. With npm, Pi's default, that install only writes a `package-lock.json` there. `"private": true` keeps the package off npm.
- Works on macOS, Windows and Linux. The plugin has no platform-specific code, and CRLF line endings, which Git uses on Windows by default, do not change its behavior.
- Only the latest Pi release is supported; checked with Pi 1.7.0.
- Licensed under the Sustainable Use License 1.0 (`SUL-1.0`); see `LICENSE`.


https://github.com/user-attachments/assets/01938534-1405-45e4-942e-82672c4655bc

