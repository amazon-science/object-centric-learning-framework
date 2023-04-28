"""Generate the code reference pages and navigation."""
from collections import defaultdict
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

override_section = {
    "ocl.typing": "# {ident}\n"
    "::: {ident}\n"
    "    options:\n"
    "      members_order: source\n"
    "      show_if_no_docstring: true",
}

# Collect high level packages into pages
package_to_page_mapping = defaultdict(default_factory=list)
ignorelist = set()

for path in sorted(Path("ocl").rglob("*.py")):
    if any(path.as_posix().startswith(prefix) for prefix in ignorelist):
        continue
    module_path = path.with_suffix("")
    doc_path = path.with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    is_summary_page = False
    if parts[-1] == "__init__":
        with path.open("r") as f:
            content = f.read()
        if content == "":
            # Skip empty __init__ files.
            continue
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        if ident in override_section:
            fd.write(override_section[ident].format(ident=ident))
        else:
            fd.write(f"# {ident}\n")
            fd.write(f"::: {ident}\n")
            fd.write("    options:\n")
            fd.write("      members_order: source\n")
            if is_summary_page:
                # Ignore direct members as these should normally be listed in the modules.
                fd.write("      members: []\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

nav["routed"] = "routed.md"

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:  #
    nav_file.writelines(nav.build_literate_nav())  #
