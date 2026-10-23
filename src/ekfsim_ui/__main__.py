"""lets the UI start with: python -m ekfsim_ui"""

import sys

from ekfsim_ui.app import main

if __name__ == "__main__":  # the compute process is spawned and imports this module again; it must not start a UI
    sys.exit(main())
