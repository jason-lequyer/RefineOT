# Debleed_Run.py
# This script is the entry point for the Fiji plugin.
# It is designed to find and run a platform-specific PyInstaller executable
# that has been downloaded via the same Fiji update site.

import os
import sys
import itertools
import tempfile
import subprocess
import re
import traceback

from ij import IJ
from ij.gui import GenericDialog
from javax.swing import (JPanel, JScrollPane, JList, DefaultListModel,
                         JButton, BoxLayout, JOptionPane, JLabel)
from java.awt import Dimension, Toolkit, BorderLayout
from java.lang import System as JavaSystem

# ----------------------------------------------------------------------
# 0 ▸ INSERT your master list here
# This configuration section remains the same.
# ----------------------------------------------------------------------
groups_by_name = [
    # The big immune / structural set
    ["CA2", "CD116", "CD14", "CD20", "CD3", "CD4", "CD44", "CD45",
     "CD45RO", "CD56", "CD57", "CD68", "CD8",
     "DNA1", "DNA2",
     "FOXP3", "Granzyme B", "Ki67", "NF-KB",
     "NKX6.1", "PDX-1", "pan-Keratin"],
    # Vascular / stem-cell pair
    ["NESTIN", "CD31"],
    # Singletons
    ["HLA-ABC"], ["C-PEPTIDE"], ["GLUCAGON"], ["SOMATOSTATIN"], ["B-ACTIN"],
    ["Collagen type I"], ["pS6"], ["HLA-DR"], ["PANCREATIC POLYPEPTIDE"], ["GHRELIN"]
]
# ----------------------------------------------------------------------

DIALOG_W, DIALOG_H = 900, 700
MARGIN_W, MARGIN_H = 40, 40
BUTTON_COL_W = 120

def abort(msg):
    """Displays an error and stops the script."""
    IJ.error("Debleed Plugin Error", msg)
    sys.exit()

def get_plugin_directory():
    """Tries to determine the directory where this script and its assets are installed."""
    try:
        # This is the most reliable method if the script is run from a .py file.
        # e.g., /path/to/Fiji.app/plugins/Debleed/
        return os.path.dirname(os.path.realpath(__file__))
    except NameError:
        # Fallback for when script is run from an unsaved editor window.
        plugins_dir = IJ.getDirectory("plugins")
        # Assumes your update site folder is named "Debleed"
        return os.path.join(plugins_dir, "Debleed")

# --- Start of main logic ---

# 1 ▸ Get active image
try:
    imp = IJ.getImage()
except Exception:
    abort("No image is open.")

# 2 ▸ Ensure TIFF path exists (no change needed here)
fi = imp.getOriginalFileInfo()
img_path_is_temporary = False
if fi and fi.directory and fi.fileName:
    img_path = os.path.join(fi.directory, fi.fileName)
else:
    img_path_is_temporary = True
    tmp = tempfile.NamedTemporaryFile(prefix="debleed_", suffix=".tif", delete=False)
    img_path = tmp.name
    IJ.saveAsTiff(imp, img_path)
    IJ.log("Image not saved, created temporary file at: " + img_path)

# 3 ▸ Channel axis detection (no change needed here)
axis_counts = {"channels": imp.getNChannels(),
               "slices":   imp.getNSlices(),
               "frames":   imp.getNFrames()}
axis_used, n_ch = max(axis_counts.items(), key=lambda kv: kv[1])
def slice_idx(c): return (imp.getStackIndex(c, 1, 1) if axis_used == "channels"
                          else imp.getStackIndex(1, c, 1) if axis_used == "slices"
                          else imp.getStackIndex(1, 1, c))

# 4 ▸ Cleaned-up channel names (no change needed here)
raw = [(imp.getStack().getSliceLabel(slice_idx(i)) or "Ch{}".format(i)).split("\n")[0].strip()
       for i in range(1, n_ch + 1)]
def common_pref(lst):
    if not lst: return ""
    s1, s2 = min(lst), max(lst); i = 0
    while i < len(s1) and s1[i] == s2[i]: i += 1
    return s1[:i]
def common_suff(lst): return common_pref([s[::-1] for s in lst])[::-1]
pre, suf = common_pref(raw), common_suff(raw)
names = [(s[len(pre):len(s)-len(suf)] or s) if suf else s[len(pre):] or s for s in raw]

# 5 ▸ Build initial groups from rules (no change needed here)
groups = []; assigned = set()
def tok_match(ref, cand):
    pat = r'(?i)(?:^|[^0-9A-Z])' + re.escape(ref) + r'(?:[^0-9A-Z]|$)'
    return re.search(pat, cand) is not None
compiled = [(ref.upper(), gi, len(ref)) for gi, g in enumerate(groups_by_name) for ref in g]
for idx, ch in enumerate([n.upper() for n in names]):
    best = None
    for ref, gi, rl in compiled:
        if tok_match(ref, ch) and (best is None or rl > best[1]):
            best = (gi, rl)
    if best:
        gi = best[0]
        while len(groups) <= gi: groups.append([])
        groups[gi].append(idx); assigned.add(idx)
for idx in range(n_ch):
    if idx not in assigned:
        groups.append([idx])
groups = [sorted(g) for g in groups if g]

# All GUI code for editing groups remains the same...
all_in_groups = set(idx for g in groups for idx in g)
avail_model = DefaultListModel()
for i, nm in enumerate(names):
    if i not in all_in_groups:
        avail_model.addElement(nm)
def build_group_model():
    model = DefaultListModel(); meta = []
    for gi, g in enumerate(groups, 1):
        model.addElement("Group {}".format(gi)); meta.append(("H", gi - 1, None))
        for idx in g:
            model.addElement("   - " + names[idx]); meta.append(("C", gi - 1, idx))
    return model, meta
groups_model, meta = build_group_model()
avail_list, group_list = JList(avail_model), JList(groups_model)
for lst in (avail_list, group_list): lst.setVisibleRowCount(18)
btn_new, btn_add, btn_rem = JButton("New  ->"), JButton("Add ->"), JButton("<- Remove")
def refresh():
    global groups_model, meta
    groups_model, meta = build_group_model()
    group_list.setModel(groups_model)
def on_new(_):
    sel = avail_list.getSelectedValuesList()
    if sel:
        groups.append([names.index(s) for s in sel])
        [avail_model.removeElement(s) for s in sel]
        refresh()
def on_add(_):
    sel = avail_list.getSelectedValuesList()
    if not sel: return
    selG = group_list.getSelectedIndex()
    gi = meta[selG][1] if selG >= 0 and meta[selG][0] == "H" else len(groups)
    if gi == len(groups): groups.append([])
    for s in sel:
        idx = names.index(s)
        if idx not in groups[gi]:
            groups[gi].append(idx); avail_model.removeElement(s)
    refresh()
def on_rem(_):
    for li in sorted(group_list.getSelectedIndices(), reverse=True):
        typ, gi, idx = meta[li]
        if typ == "C":
            groups[gi].remove(idx); avail_model.addElement(names[idx])
            if not groups[gi]: groups.pop(gi)
    refresh()
for b, f in ((btn_new, on_new), (btn_add, on_add), (btn_rem, on_rem)): b.addActionListener(f)
mid = JPanel(); mid.setLayout(BoxLayout(mid, BoxLayout.Y_AXIS))
for b in (btn_new, btn_add, btn_rem): mid.add(b)
mid.setPreferredSize(Dimension(BUTTON_COL_W, mid.getPreferredSize().height))
mid.setMaximumSize(Dimension(BUTTON_COL_W, 1000))
editor = JPanel(); editor.setLayout(BoxLayout(editor, BoxLayout.X_AXIS))
editor.add(JScrollPane(avail_list)); editor.add(mid); editor.add(JScrollPane(group_list))
panel = JPanel(BorderLayout())
panel.add(JLabel("<html><b>Co-localising groups</b> – edit if needed</html>"), BorderLayout.NORTH)
panel.add(editor, BorderLayout.CENTER)
scr = Toolkit.getDefaultToolkit().getScreenSize()
panel.setPreferredSize(Dimension(min(DIALOG_W, scr.width - MARGIN_W), min(DIALOG_H, scr.height - MARGIN_H)))
if JOptionPane.showConfirmDialog(None, panel, "Bleed-through groups", JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION:
    sys.exit()

# 7 ▸ CSV generation (no change needed here)
zero = {(i, j) for g in groups for i, j in itertools.permutations(g, 2)}
csv = ["," + ",".join(names)]
for r in range(n_ch):
    csv.append(",".join([names[r]] + ["0" if r == c or (r, c) in zero else "1" for c in range(n_ch)]))
csv_path = os.path.splitext(img_path)[0] + ".csv"
with open(csv_path, "w") as f:
    f.write("\n".join(csv))

# 8 ▸ Choose channel to debleed (dialog simplified)
dlg = GenericDialog("Run Debleed")
dlg.addNumericField("Channel to debleed (1-based):", 1, 0)
dlg.showDialog()
if dlg.wasCanceled(): sys.exit()
chan = str(int(dlg.getNextNumber()))

# 9 ▸ **NEW**: Find and launch the 'debleed' executable
plugin_dir = get_plugin_directory()
os_name = JavaSystem.getProperty("os.name").lower()
executable_path_relative = None

# Determine the relative path to the executable based on OS
if "linux" in os_name:
    executable_path_relative = os.path.join("Linux", "debleed")
# Add other platforms here as you build executables for them
# elif "win" in os_name:
#     executable_path_relative = os.path.join("Windows", "debleed.exe")
# elif "mac" in os_name or "darwin" in os_name:
#     executable_path_relative = os.path.join("macOS", "debleed")
else:
    abort("This plugin currently only supports Linux.")

# Construct the full path and check for existence
executable_full_path = os.path.join(plugin_dir, executable_path_relative)
if not os.path.exists(executable_full_path):
    abort("Debleed executable not found. Expected at:\n" + executable_full_path)

# Ensure the file is executable on Linux/macOS
if ("linux" in os_name or "mac" in os_name) and not os.access(executable_full_path, os.X_OK):
    IJ.log("Setting execute permission for: " + executable_full_path)
    os.chmod(executable_full_path, 0755)

# Build the command and run the process
command = [executable_full_path, img_path, chan]
IJ.log("Executing command: " + " ".join(command))
try:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()  # Waits for the process to finish
    
    if stdout: IJ.log("Debleed stdout:\n" + stdout)
    if stderr: IJ.error("Debleed stderr:\n" + stderr) # Use IJ.error to show in red

    if process.returncode != 0:
        abort("Debleed process failed with return code: " + str(process.returncode))

except Exception, e:
    abort("Failed to execute debleed:\n" + str(e) + "\n" + traceback.format_exc())

# Clean up the temporary image file if one was created
if img_path_is_temporary and os.path.exists(img_path):
    os.remove(img_path)

# 10 ▸ Open the result
out_path = "{}_Channel_{}_debleed.tif".format(img_path[:-4], chan)
if not os.path.exists(out_path):
    abort("Result image not found. Expected at:\n" + out_path)

IJ.openImage(out_path).show()
IJ.showStatus("Debleed finished successfully.")
