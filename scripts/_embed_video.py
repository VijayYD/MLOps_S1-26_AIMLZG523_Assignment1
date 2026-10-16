"""One-off: transplant the already-validated embedded-video OLE object
(and its icon image) out of a previous build of the report and into a
freshly (re)built one, in place of build_report.py's VIDEO_OLE_MARKER
placeholder paragraph.

Why this exists at all: python-docx has no API for constructing a Word
OLE "Package" object (the Compound File Binary Format wrapper Word uses
to embed an arbitrary file as a clickable icon), and hand-rolling that
binary format from scratch is unverifiable and risky (see docs/DECISIONS.md
Phase 9 notes on this). The object was instead created once, by hand, in
real Microsoft Word, and is copied here byte-for-byte on every subsequent
rebuild of the report -- this script never constructs OLE bytes itself,
only relocates an already-working one.

Usage:
    python scripts/_embed_video.py <source_docx_with_working_embed> <target_docx_to_inject_into>

Edits <target_docx_to_inject_into> in place.
"""

import re
import shutil
import sys
import zipfile
from pathlib import Path

MARKER = "{{VIDEO_OLE_OBJECT_PLACEHOLDER}}"


def main(source_path: str, target_path: str) -> None:
    source_path, target_path = Path(source_path), Path(target_path)
    work_src = Path("/tmp/_embed_video_src")
    work_dst = Path("/tmp/_embed_video_dst")
    for d in (work_src, work_dst):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    with zipfile.ZipFile(source_path) as z:
        z.extractall(work_src)
    with zipfile.ZipFile(target_path) as z:
        z.extractall(work_dst)

    # --- find the object's relationship ids in the source doc ---
    src_doc = (work_src / "word" / "document.xml").read_text(encoding="utf-8")
    obj_start = src_doc.find("<w:object")
    obj_end = src_doc.find("</w:object>", obj_start) + len("</w:object>")
    object_xml = src_doc[obj_start:obj_end]

    icon_rid, ole_rid = re.findall(r'r:id="(rId\d+)"', object_xml)

    src_rels = (work_src / "word" / "_rels" / "document.xml.rels").read_text(encoding="utf-8")
    icon_target = re.search(rf'<Relationship Id="{icon_rid}"[^>]*Target="([^"]*)"', src_rels).group(
        1
    )
    ole_target = re.search(rf'<Relationship Id="{ole_rid}"[^>]*Target="([^"]*)"', src_rels).group(1)

    # --- copy the binary parts into the target package ---
    (work_dst / "word" / "embeddings").mkdir(exist_ok=True)
    new_icon_name = "media/video_ole_icon.png"
    new_ole_name = "embeddings/oleObject1.bin"
    shutil.copy(work_src / "word" / icon_target, work_dst / "word" / new_icon_name)
    shutil.copy(work_src / "word" / ole_target, work_dst / "word" / new_ole_name)

    # --- add relationships in the target, with fresh ids ---
    dst_rels_path = work_dst / "word" / "_rels" / "document.xml.rels"
    dst_rels = dst_rels_path.read_text(encoding="utf-8")
    existing_ids = [int(n) for n in re.findall(r'Id="rId(\d+)"', dst_rels)]
    new_icon_rid = f"rId{max(existing_ids) + 1}"
    new_ole_rid = f"rId{max(existing_ids) + 2}"
    new_rels = (
        f'<Relationship Id="{new_icon_rid}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="{new_icon_name}"/>'
        f'<Relationship Id="{new_ole_rid}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/oleObject" '
        f'Target="{new_ole_name}"/>'
    )
    dst_rels = dst_rels.replace("</Relationships>", new_rels + "</Relationships>")
    dst_rels_path.write_text(dst_rels, encoding="utf-8")

    # --- content-type override for .bin, if not already present ---
    ct_path = work_dst / "[Content_Types].xml"
    ct = ct_path.read_text(encoding="utf-8")
    if 'Extension="bin"' not in ct:
        bin_default = (
            '<Default Extension="bin" '
            'ContentType="application/vnd.openxmlformats-officedocument.oleObject"/>'
        )
        ct = ct.replace("<Default", bin_default + "<Default", 1)
        ct_path.write_text(ct, encoding="utf-8")

    # --- splice the object XML (with updated rIds) into the target doc ---
    object_xml_updated = object_xml.replace(icon_rid, new_icon_rid).replace(ole_rid, new_ole_rid)
    run_wrapped = f"<w:r><w:rPr><w:noProof/></w:rPr>{object_xml_updated}</w:r>"
    replacement_paragraph = f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{run_wrapped}</w:p>'

    dst_doc_path = work_dst / "word" / "document.xml"
    dst_doc = dst_doc_path.read_text(encoding="utf-8")
    marker_idx = dst_doc.find(MARKER)
    if marker_idx == -1:
        raise RuntimeError(f"Marker {MARKER!r} not found in {target_path} -- nothing to replace.")
    p_start = dst_doc.rfind("<w:p>", 0, marker_idx)
    p_end = dst_doc.find("</w:p>", marker_idx) + len("</w:p>")
    dst_doc = dst_doc[:p_start] + replacement_paragraph + dst_doc[p_end:]
    dst_doc_path.write_text(dst_doc, encoding="utf-8")

    # --- repack ---
    if target_path.exists():
        target_path.unlink()
    with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(work_dst.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(work_dst))

    print(f"Embedded video OLE object into {target_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
