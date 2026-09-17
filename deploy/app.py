"""Hugging Face entry point; the pinned wheel contains the UI and evidence."""

import runpy

runpy.run_module("judgeguard._workbench", run_name="__main__")
