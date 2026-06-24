"""Allow `python -m amora` to dispatch to the CLI."""

from amora.cli import main

raise SystemExit(main())
