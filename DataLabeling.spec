# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: python -m PyInstaller DataLabeling.spec --noconfirm
from PyInstaller.utils.hooks import collect_all

datas = [("assets", "assets")]
binaries = []
hiddenimports = []
# ultralytics is imported lazily and ships cfg/*.yaml data files;
# onnx/onnxslim are pulled in at export time (no pip in frozen builds)
for pkg in ("ultralytics", "onnx", "onnxslim"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DataLabeling",
    icon="assets/icon.ico",
    console=False,
    upx=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="DataLabeling",
    upx=False,
)
