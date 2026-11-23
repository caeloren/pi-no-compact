# pi-no-compact

## Introduction

pi-no-compact is a Pi extension that replaces what Pi's compaction keeps. When Pi compacts — `/compact`, the context threshold, or context overflow recovery — no LLM summary is written and Pi's recent tail is not kept. The new context holds only a fixed instruction and the last structurally complete fragment of the active branch:

- the last message the model can see, or
- when the branch ends with tool results, the assistant message that made those tool calls, followed by its results. A tool call without a result is kept as it is; Pi adds its `No result provided` result when it sends the context.

Everything else stays in the session and is retrieved on demand with two tools, `history_search` and `history_read`. The fixed instruction, stored as the compaction summary, is:

```text
Context was compacted. Earlier records are no longer in your context but remain available through history_search and history_read.
Before continuing, read earlier messages, then the latest records.
For any other missing detail, search instead of guessing, and use history_read for exact text.
```

When no complete fragment can be kept, the plugin reports `Compaction failed: <reason>` as a Pi error notification, stops the current operation and cancels the compaction. For example, the branch may end with a tool result whose tool call the model cannot see, or with tool results separated from their calls by a failed or aborted attempt, where Pi would send both its `No result provided` result and the real result. It never falls back to Pi's built-in compaction.

## Usage

Install, update and uninstall the plugin with Pi's git package commands. It is downloaded only through git, never from npm:

```bash
pi install git:github.com/caeloren/pi-no-compact    # install the latest commit of main
pi update git:github.com/caeloren/pi-no-compact     # update to the latest commit of main
pi uninstall git:github.com/caeloren/pi-no-compact  # remove the plugin; same as pi remove
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

Each block is searched as the text the model sees. `!` commands and branch summaries include the text Pi wraps around them. A tool call is its name followed by each argument value on its own line, strings exactly as stored, without argument names. Images are never searched; browsing lists their blocks too.

- `literal` finds an exact, contiguous, case-sensitive substring of that text. Use it for paths, error codes, function and class names, commit hashes and whole error messages.
- `terms` applies NFKC normalization (case is kept), splits all text into words with `Intl.Segmenter`, keeps word-like segments, and turns each run of contiguous CJK characters into overlapping two-character tokens, so a run of seven characters gives six tokens; a lone CJK character stays one token. Pairs never cross spaces, punctuation, Latin text or separate runs. Query tokens are deduplicated and compared as whole, case-sensitive tokens, and a block matches when it contains any of them: `data` does not match `database`, and `schema` does not match `schemas`. Use `literal` for substrings.
- Case is never folded or substituted: `ErrorCode` and `errorcode` are different. No results is a normal, successful answer.

Results are blocks of records, ordered by their position on the branch, never by timestamp. `relevance` sorts by the number of distinct query tokens found, then newest first; with `literal` every hit counts once, so `relevance` is newest first. `oldest` and `newest` follow the branch, and blocks of one record keep their order.

The result is one JSON object: `query`, `match`, `totalMatches` (matching blocks in the snapshot), `matches` and `nextCursor`. Each match only says where a block is: `entryId`, `blockIndex`, `timestamp`, `kind`, `toolName` and `derived` (true for branch summaries). Matches carry no text; read the content with `history_read`.

The first page fixes a snapshot: the session, the leaf of the active branch, the entries up to that leaf, `query`, `match`, `kinds`, `order` and the cursor format. Entries added later are not part of that search. `nextCursor` holds the sort position of the page's last result, not an offset, and a SHA-256 check of its content, so pages never repeat or skip a result. Every page applies the latest context edits. A cursor fails, and the search starts again with `cursor: null`, when it was changed, when its position is not a result of the search, when it comes from another session, when the active branch no longer contains the snapshot leaf, when a new context edit changes the visibility or content of an entry in the snapshot, or when it is used with another `query`, `match`, `kinds` or `order`. `limit` may change between pages. Page until `nextCursor` is `null` to list every candidate.

`history_read` reads whole records by `entryId`:

| Parameter | Type | Meaning |
|---|---|---|
| `entryIds` | array of strings, at least one | Records to read, such as `entryId` values from `history_search`. |
| `around` | integer, at least 0 | Readable records to add before and after each requested record. |
| `includeImages` | boolean | `true` attaches images; `false` shows placeholders only. |
| `maxImages` | integer, at least 0 | Images to attach: `0` when `includeImages` is `false`, more than `0` when it is `true`. |

It returns one JSON object per line, in branch order: `entryId`, `timestamp`, `kind`, `derived`, `toolName`, `toolCallId` and `isError` for tool results, and `blocks` with their `blockIndex`, `kind` and the full `text` the model sees, the tool call (`toolName`, `toolCallId` and the exact `arguments`) or the image placeholder (`[image: <mimeType>]`). Overlapping `around` ranges are merged. With `includeImages: true`, the first `maxImages` images are attached after the text as image content, in order, and their `attachment` gives that position; the rest stay placeholders.

Argument types, required fields and ranges are checked by Pi before the tools run, with Pi's own `Validation failed for tool ...` errors. Pi first converts values that can be converted, for example `1.5` becomes `1`. Other problems are tool errors, such as `Entry <id> is not on the active branch`, `Entry <id> has no readable content` or an invalid cursor.

## Notes and limitations

- Both tools see only the current session and its active branch. Other sessions, conversations and branches are never read, and there are no parameters to reach them. Every `entryId` is checked: one that is not on the active branch gets the same error whether it is on another branch, in another session or does not exist.
- The tools return only what the model was entitled to see: user messages, assistant text and tool calls, tool results, visible `!` commands, custom messages and branch summaries, after the latest context edit of each entry. A context edit that omits an entry hides it from both tools; a replacement shows only the replaced content. Thinking, signatures, the system prompt, tool declarations, usage, `details`, `!!` commands, failed or aborted attempts, compaction summaries, and the calls and results of `history_search` and `history_read` themselves are never searched or read.
- Session files are not validated, so a record may hold malformed content. As in Pi's own readers of stored history, content that is not an array counts as empty and an item that is not an object is skipped: only the malformed part cannot be searched or read, and every other record still can. While such a record is still in the context, Pi itself can fail on it; the plugin does not check what Pi sends.
- The plugin sets no output limits: no byte or line cap, no maximum for `limit`, `entryIds`, `around` or `maxImages`, and `history_read` never truncates or splits a record. Model, provider and runtime limits still apply, so a very large result, or a very large kept fragment, can make the next request fail.
- The cursor check has no secret key. It catches a cursor changed by mistake, but a cursor rewritten on purpose to the position of another result, with its check recomputed, is accepted. That can only skip or repeat results of that search; it reaches nothing the tools would not return anyway.
- Whether images reach the model also depends on the model's image support, Pi's image settings, the branch and context edits.
- The plugin keeps no data of its own: no transcript copy, database, index, embeddings, remote service or background work. It reads the session through Pi's API on each call and never changes, deletes or rewrites it. A compaction appends Pi's normal compaction entry.
- Pi decides when to compact, and its own checks, such as "Nothing to compact", still apply. Instructions passed to `/compact` are ignored, because no summary is written.
- Pi shows notifications only where it has a UI. The interactive mode shows `Compaction failed: <reason>` as an error, and RPC clients receive it as a `notify` request. Print (`-p`) and JSON modes have no UI, so the reason is not shown there. The compaction is still cancelled and the operation stopped: a prompt that needed compaction first is not sent and ends with Pi's `This operation was aborted` error.
- When the plugin is not loaded — disabled, removed, or failed to load — Pi may use its built-in compaction.
- Pi runs the `session_before_compact` handlers of all extensions in load order. If an extension loaded after this plugin also returns a result, Pi uses that result instead, which can mean Pi's built-in compaction. A cancel from this plugin is final.
- After the plugin is removed, earlier compaction entries still contain the fixed instruction, which names `history_search` and `history_read`. Those tools no longer exist, so calls to them fail with Pi's `Tool ... not found` error. Removing the plugin does not change or damage the session; it stays readable and usable.
- The plugin is downloaded only through git, and the package has only four files: `LICENSE`, `README.md`, `package.json` and `index.ts`. It has no dependencies, so the dependency install that Pi runs in the checkout has nothing to download. With npm, Pi's default, that install only writes a `package-lock.json` there. `"private": true` keeps the package off npm.
- Checked with Pi 1.6.0.
- Licensed under the Sustainable Use License 1.0 (`SUL-1.0`); see `LICENSE`.
