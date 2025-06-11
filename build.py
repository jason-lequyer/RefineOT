import os
import hashlib
import time
import gzip
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- Configuration ---
# Path to your local 'docs' folder (where db.xml.gz will be saved)
LOCAL_DOCS_DIR = "docs"

# Path to the directory WITHIN 'docs/plugins/' that contains your plugin files.
PLUGIN_SUBDIR = "Debleed" # Matches your folder name

# Define your files and their platform specifics.
FILES_TO_INCLUDE = [
    {
        "path_in_plugin_dir": "Debleed_Run.py",    # The Jython script Fiji runs
        "os": None,                                # Platform-independent
        "arch": None,
        "description": "Launcher for the Debleed tool"
    },
    {
        "path_in_plugin_dir": "Linux/debleed",     # Your PyInstaller executable for Linux
        "os": "Linux",
        "arch": "amd64",  # Or "x86_64". Change if your build is for a different arch.
        "description": "Debleed executable (Linux x64)"
    },
    {
        "path_in_plugin_dir": "debleed.py",        # The pure Python script
        "os": None,                                # Assuming it might be used by the launcher or for other platforms
        "arch": None,
        "description": "Debleed Python script (main logic or fallback)"
    }
    # Add entries for other platform-specific PyInstaller executables if you create them
    # e.g., "windows_x64/debleed.exe" for Windows
]
# --- End Configuration ---

def get_sha1(file_path):
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)  # Read in 64k chunks
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()

def create_db_xml(base_docs_dir, plugin_subdir_name, files_to_include):
    output_xml_path = os.path.join(base_docs_dir, "db.xml")
    # This is the absolute path to where your plugin files are located locally for the script
    plugin_base_abs_path = os.path.join(base_docs_dir, "plugins", plugin_subdir_name)

    if not os.path.exists(plugin_base_abs_path):
        print(f"ERROR: Plugin directory not found: {plugin_base_abs_path}")
        print(f"Please ensure '{base_docs_dir}/plugins/{plugin_subdir_name}/' exists and contains your files.")
        return

    root = ET.Element("pluginRecords")

    for file_info in files_to_include:
        path_in_plugin_dir = file_info["path_in_plugin_dir"]
        # This is the path as it will appear in db.xml.gz, relative to the update site root (docs/)
        relative_file_path_for_xml = f"plugins/{plugin_subdir_name}/{path_in_plugin_dir}".replace("\\", "/")
        # This is the local path to the file for checksumming etc.
        absolute_file_path = os.path.join(plugin_base_abs_path, path_in_plugin_dir)

        if not os.path.exists(absolute_file_path):
            print(f"WARNING: File not found, skipping: {absolute_file_path}")
            continue

        print(f"Processing: {absolute_file_path}")

        stat_info = os.stat(absolute_file_path)
        checksum = get_sha1(absolute_file_path)
        timestamp = int(stat_info.st_mtime * 1000)  # Milliseconds
        filesize = stat_info.st_size
        description = file_info.get("description", os.path.basename(path_in_plugin_dir))

        plugin_el = ET.SubElement(root, "plugin", filename=relative_file_path_for_xml)

        if file_info.get("os"):
            platform_attrs = {"os": file_info["os"]}
            if file_info.get("arch"):
                platform_attrs["arch"] = file_info["arch"]
            ET.SubElement(plugin_el, "platform", **platform_attrs)
        
        ET.SubElement(plugin_el, "previous-version",
                      checksum="da39a3ee5e6b4b0d3255bfef95601890afd80709",
                      timestamp="0")
        
        ET.SubElement(plugin_el, "version",
                      checksum=checksum,
                      timestamp=str(timestamp),
                      filesize=str(filesize))
        
        desc_el = ET.SubElement(plugin_el, "description")
        desc_el.text = description

    xml_str = ET.tostring(root, encoding="utf-8")
    dom = minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml(indent="  ", encoding="UTF-8")

    with open(output_xml_path, 'wb') as f_xml:
        f_xml.write(pretty_xml_str)
    print(f"\nGenerated XML: {output_xml_path}")

    gzipped_path = output_xml_path + ".gz"
    with open(output_xml_path, 'rb') as f_in, gzip.open(gzipped_path, 'wb') as f_out:
        f_out.write(f_in.read())
    
    print(f"Generated GZipped XML: {gzipped_path}")
    print(f"\nMake sure your '{base_docs_dir}/' folder and its contents (especially '{plugins_root_in_docs}/' and '{gzipped_path}') are committed to GitHub.")

if __name__ == "__main__":
    if not os.path.exists(LOCAL_DOCS_DIR):
        os.makedirs(LOCAL_DOCS_DIR)
        print(f"Created directory: {LOCAL_DOCS_DIR}")

    plugins_root_in_docs = os.path.join(LOCAL_DOCS_DIR, "plugins")
    if not os.path.exists(plugins_root_in_docs):
        os.makedirs(plugins_root_in_docs)
        print(f"Created directory: {plugins_root_in_docs}")
        
    specific_plugin_dir_in_docs = os.path.join(plugins_root_in_docs, PLUGIN_SUBDIR)
    if not os.path.exists(specific_plugin_dir_in_docs):
        os.makedirs(specific_plugin_dir_in_docs)
        print(f"Created directory: {specific_plugin_dir_in_docs}")
        print(f"IMPORTANT: Please place your plugin files into '{specific_plugin_dir_in_docs}'")
        print("according to the paths in FILES_TO_INCLUDE before running this script if it reported files not found.")

    create_db_xml(LOCAL_DOCS_DIR, PLUGIN_SUBDIR, FILES_TO_INCLUDE)
