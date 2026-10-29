"""Headless UI test: drives app.py with Streamlit AppTest for every sample. Run: python tests/ui_test.py"""
import os, sys
from streamlit.testing.v1 import AppTest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
samples = sorted(f for f in os.listdir("samples") if f.endswith((".jpg", ".png", ".dcm")))
bad = 0
for s in samples:
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240)
    at.run()
    at.selectbox[0].select(s).run()
    if at.exception:
        bad += 1; print("EXC", s, at.exception[0].message); continue
    banner = [e.value for e in (list(at.error) + list(at.warning) + list(at.success) + list(at.info)) if "###" in str(e.value)]
    print("ok ", s, "->", banner[0].splitlines()[0] if banner else "?", "| metrics:", [m.value for m in at.metric])
at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240); at.run(); at.checkbox[0].check().run()
print("red-flag override:", [e.value.splitlines()[0] for e in at.error if "###" in e.value])
print("UI TEST:", "PASSED" if not bad else f"FAILED ({bad})"); sys.exit(bad)
