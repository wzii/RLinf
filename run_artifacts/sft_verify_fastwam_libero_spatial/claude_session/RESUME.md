# Resume this Claude Code session on another machine

- `session_transcript.jsonl` — full transcript of this session. To resume context,
  place it at `~/.claude/projects/-workspace/<id>.jsonl` on the other machine
  (keep the same filename/id), then open Claude Code in that project dir.
- `memory/` — the persistent project memory written during this session
  (copy into `~/.claude/projects/-workspace/memory/`). `fastwam-rlinf-sft-verify.md`
  is the full status/findings record.

Key state: SFT verification COMPLETE & POSITIVE (see ../README.md). RL follow-up is
interconnect-limited on the original box (A100 PCIe, no NVLink) — run it on A100-SXM.
