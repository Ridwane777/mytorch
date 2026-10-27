# Privacy policy

Applies to `my_torch_analyzer`, `my_torch_generator`, the `my_torch` library and the scripts of this
repository.

- **No personal data is collected.** The programs read the files given on the command line (chess
  positions, configurations, networks) and write their results to the standard output or to the
  files asked for.
- **Nothing leaves the machine.** No network access, no telemetry, no analytics, no account.
- **No log is kept.** Messages go to the terminal (standard output and standard error) only.
- The chess positions handled are not personal data: a FEN describes a board, not a person.

The GDPR (Regulation (EU) 2016/679) therefore has nothing to apply to. A hosted version (a web
service receiving positions) would change this: it would need a legal basis, a record of processing
and a retention period (risk R6 in [risk-analysis.md](risk-analysis.md)).

Contact: the authors listed in [legal-notice.md](legal-notice.md).
