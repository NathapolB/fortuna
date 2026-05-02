"""Allow `python -m fortuna` to invoke CLI."""
import sys
from fortuna.cli import main

sys.exit(main())
