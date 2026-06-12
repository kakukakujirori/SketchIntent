import argparse
import glob
import os
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
ensure_freecad_imported()

import FreeCAD as App
import FreeCADGui as Gui

Gui.setupWithoutGUI()


SKETCH_EXTENSIONS_XML = """<Extensions Count="3">
<Extension type="Gui::ViewProviderFaceTexture" name="ViewProviderFaceTexture"> </Extension>
<Extension type="PartGui::ViewProviderAttachExtension" name="ViewProviderAttachExtension"> </Extension>
<Extension type="PartGui::ViewProviderGridExtension" name="ViewProviderGridExtension"> </Extension>
</Extensions>"""

CAMERA_DEFAULT = (
    'OrthographicCamera { viewportMapping ADJUST_CAMERA '
    'position 0 0 1 orientation 0 0 1 0 '
    'nearDistance 0.99900001 farDistance 1.001 '
    'aspectRatio 1 focalDistance 5 height 4.1421356 } '
)


def pack_color(r: float, g: float, b: float, a: float = 1.0) -> int:
    return (int(r * 255) << 24) | (int(g * 255) << 16) | (int(b * 255) << 8) | int(a * 255)


def build_property_elem(name: str, value) -> ET.Element:
    """Generate color, line width, and visibility flags as XML Elements"""
    if name == "LineColor":
        r, g, b = value
        prop = ET.Element("Property", name="LineColor",
                          type="App::PropertyColor", status="1")
        ET.SubElement(prop, "PropertyColor", value=str(pack_color(r, g, b)))

    elif name == "LineWidth":
        prop = ET.Element("Property", name="LineWidth",
                          type="App::PropertyFloatConstraint", status="1")
        ET.SubElement(prop, "Float", value=f"{value:.16f}")

    elif name == "Selectable":
        prop = ET.Element("Property", name="Selectable",
                          type="App::PropertyBool", status="1")
        ET.SubElement(prop, "Bool", value="true" if value else "false")

    elif name == "VisualLayerList":
        # value = [(visible, linePattern, lineWidth), ...]
        prop = ET.Element("Property", name="VisualLayerList",
                          type="BadType", status="16777217")
        vll = ET.SubElement(prop, "VisualLayerList", count=str(len(value)))
        for visible, pattern, width in value:
            ET.SubElement(vll, "VisualLayer",
                          visible="true" if visible else "false",
                          linePattern=str(pattern),
                          lineWidth=f"{width:.16f}")
        return prop

    else:
        raise ValueError(f"Unsupported property: {name}")

    return prop


def make_gui_document(all_object_names: list[str],
                      color_map: dict) -> bytes:
    """
    Create a new GuiDocument.xml.
    all_object_names : List of Labels for all objects in the document
    color_map        : {label: {LineColor, LineWidth, Selectable}}
    """
    root = ET.Element("Document", SchemaVersion="1", HasExpansion="1")
    ET.SubElement(root, "Expand")

    vp_data = ET.SubElement(root, "ViewProviderData",
                             Count=str(len(all_object_names)))

    for label in all_object_names:
        vp = ET.SubElement(root, "dummy")  # To be replaced later

        settings = color_map.get(label, {})
        prop_elems = [build_property_elem(k, v) for k, v in settings.items()]

        vp = ET.SubElement(vp_data, "ViewProvider",
                           name=label, expanded="0")

        # SketchObject requires Extensions
        ext_frag = ET.fromstring(SKETCH_EXTENSIONS_XML)
        vp.append(ext_frag)

        props_elem = ET.SubElement(vp, "Properties",
                                   Count=str(len(prop_elems)),
                                   TransientCount="0")
        for pe in prop_elems:
            props_elem.append(pe)

    ET.SubElement(root, "Camera", settings=CAMERA_DEFAULT)

    header = b'<?xml version="1.0" encoding="UTF-8"?>\n'
    body = ET.tostring(root, encoding="unicode").encode("utf-8")
    return header + body


def patch_existing_gui_document(raw_bytes: bytes, color_map: dict) -> bytes:
    """Patch existing GuiDocument.xml with color settings"""
    root = ET.fromstring(raw_bytes)
    vp_data = root.find("ViewProviderData")
    if vp_data is None:
        return raw_bytes

    existing_vps = {vp.get("name"): vp for vp in vp_data.findall("ViewProvider")}

    for label, settings in color_map.items():
        if label not in existing_vps:
            # Add new objects
            vp = ET.SubElement(vp_data, "ViewProvider", name=label, expanded="0")
            ext_frag = ET.fromstring(SKETCH_EXTENSIONS_XML)
            vp.append(ext_frag)
            existing_vps[label] = vp
            vp_data.set("Count", str(len(vp_data.findall("ViewProvider"))))

        vp = existing_vps[label]
        props_elem = vp.find("Properties")
        if props_elem is None:
            props_elem = ET.SubElement(vp, "Properties", Count="0", TransientCount="0")

        # Remove existing property with the same name before adding
        existing_props = {p.get("name"): p for p in props_elem.findall("Property")}
        for prop_name, prop_value in settings.items():
            if prop_name in existing_props:
                props_elem.remove(existing_props[prop_name])
            props_elem.append(build_property_elem(prop_name, prop_value))

        props_elem.set("Count", str(len(props_elem.findall("Property"))))

    header = b'<?xml version="1.0" encoding="UTF-8"?>\n'
    return header + ET.tostring(root, encoding="unicode").encode("utf-8")


def apply_gui_colors(fcstd_path: str,
                     all_object_labels: list[str],
                     color_map: dict):
    with tempfile.NamedTemporaryFile(suffix=".FCStd", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with zipfile.ZipFile(fcstd_path, "r") as zin, \
             zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:

            names = zin.namelist()
            has_gui = "GuiDocument.xml" in names

            for item in zin.infolist():
                raw = zin.read(item.filename)
                if item.filename == "GuiDocument.xml":
                    raw = patch_existing_gui_document(raw, color_map)
                zout.writestr(item, raw)

            # Create and add GuiDocument.xml if it does not exist
            if not has_gui:
                print(f"  ★ Creating new GuiDocument.xml: {os.path.basename(fcstd_path)}")
                gui_xml = make_gui_document(all_object_labels, color_map)
                zout.writestr("GuiDocument.xml", gui_xml)

        shutil.move(tmp_path, fcstd_path)

    except Exception:
        os.unlink(tmp_path)
        raise


def process_all_files(directory: str):
    files = glob.glob(os.path.join(directory, "*.FCStd"))
    if not files:
        print(f"No .FCStd files found in the specified directory: {directory}")
        return

    for filepath in files:
        doc = App.openDocument(filepath)
        if not doc:
            print(f"Could not open file: {filepath}")
            continue

        # Collect all object labels in the document (required for GuiDocument generation)
        all_labels = [obj.Label for obj in doc.Objects]

        target_sketch = None
        for obj in doc.Objects:
            if obj.isDerivedFrom("Sketcher::SketchObject"):
                target_sketch = obj
                break

        if target_sketch is None:
            App.closeDocument(doc.Name)
            continue

        target_sketch.Label = "Sketch"

        normal_geos, const_geos = [], []
        for i, geo in enumerate(target_sketch.Geometry):
            if target_sketch.getConstruction(i):
                const_geos.append(geo.copy())
            else:
                normal_geos.append(geo.copy())

        color_map = {}

        if normal_geos:
            ref = doc.addObject("Sketcher::SketchObject", "Sketch_Original_Normal")
            for geo in normal_geos:
                ref.addGeometry(geo)
            all_labels.append(ref.Label)
            doc.recompute()
            color_map["Sketch_Original_Normal"] = {
                "LineColor": (0.5, 0.5, 0.5),
                "LineWidth": 2.0,
                "Selectable": False,
            }

        if const_geos:
            ref = doc.addObject("Sketcher::SketchObject", "Sketch_Original_Const")
            for geo in const_geos:
                ref.addGeometry(geo)
            all_labels.append(ref.Label)
            doc.recompute()
            color_map["Sketch_Original_Const"] = {
                "LineColor": (0.5, 0.5, 0.5),
                "LineWidth": 1.0,
                "Selectable": False,
                "VisualLayerList": [
                    (True,  32382, 1.0),  # layer0: Regular line -> Changed to dashed
                    (True,  32382, 1.0),  # layer1: Construction line
                    (False, 65535, 1.0),  # layer2: External reference line
                ],
            }

        doc.save()
        App.closeDocument(doc.Name)

        if color_map:
            apply_gui_colors(filepath, all_labels, color_map)

        print(f"Completed: {os.path.basename(filepath)}")

    # Delete all .FCStd.bak and .FCBak files
    backup_files = glob.glob(os.path.join(directory, "*.FCStd.bak")) + glob.glob(os.path.join(directory, "*.FCBak"))
    for p in backup_files:
        os.remove(p)

    print("All processing completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory containing .FCStd files")
    args = parser.parse_args()
    process_all_files(args.data_dir)