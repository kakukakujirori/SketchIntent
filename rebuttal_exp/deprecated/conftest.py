"""These files are archived provenance, not a package.

Their ``from rebuttal_exp.…`` imports would resolve to the *new* modules, so collecting the
archived tests here would fail on names (``MAIN_ARM``) that no longer exist.
"""
collect_ignore_glob = ["*"]
