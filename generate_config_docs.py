import os
import shutil
from io import StringIO
from pathlib import Path

import mkdocs_gen_files
import ruamel.yaml
import ruamel.yaml.comments
from git import Repo

yaml = ruamel.yaml.YAML()
yaml.default_flow_style = False
yaml.indent(mapping=2, sequence=4, offset=2)

nav = mkdocs_gen_files.Nav()
current_repo_tree = Repo(os.curdir).head.commit.tree


def file_in_repo(filePath):
    try:
        current_repo_tree.join(filePath)
        return True
    except KeyError:
        return False


def convert_dl_entry_to_link(entry: str, current_path: Path) -> str:
    # Handle hydra package annotations.
    if "@" in entry:
        entry, _ = entry.split("@")
    # Special handling for configs defined in code.
    if entry == "/training_config":
        return f"[{entry}][training-config]"
    if entry == "/evaluation_config":
        return f"[{entry}][evaluation-config]"
    if entry == "/evaluation_clustering_config":
        return f"[{entry}][evaluation-clustering-config]"
    elif entry.startswith("/"):
        # Global path.
        anchor_name = "configs" + entry.lower().replace("/", "").replace(".", "") + "yaml"
        return f"[{entry}][{anchor_name}]"
    else:
        # Path relative to group.
        prefix = current_path.relative_to("configs").parts[0]
        anchor_name = "configs" + prefix + entry.lower().replace("/", "").replace(".", "") + "yaml"
        return f"[/{prefix}/{entry}][{anchor_name}]"


def augment_defaults_list_links(defaults_list: ruamel.yaml.comments.CommentedSeq, path: Path):
    links = []
    if isinstance(defaults_list, ruamel.yaml.comments.CommentedSeq):
        for i, entry in enumerate(defaults_list):
            if isinstance(entry, str):
                if entry == "_self_":
                    continue
                defaults_list.yaml_add_eol_comment(f"({i+1})!", i)
                links.append(f"{i+1}. {convert_dl_entry_to_link(entry, path)}")
            elif isinstance(entry, ruamel.yaml.comments.CommentedMap):
                # Should only have a single key value combination as we are in the defaults list.
                key, value = next(iter(entry.items()))
                if not (key.startswith("/experiment") or key.startswith("/dataset")):
                    continue
                entry.yaml_add_eol_comment(f"({i+1})!", key)
                link = convert_dl_entry_to_link(f"{key}/{value}", path)
                links.append(f"{i+1}. {link}")

    return links


def get_doc_page_source(config_path: Path):
    with config_path.open("r") as f:
        data = yaml.load(f)
    links = {}
    if "defaults" in data:
        links = augment_defaults_list_links(data["defaults"], config_path)

    strio = StringIO()
    yaml.dump(data, strio)
    md_code = strio.getvalue()

    output = f"# {config_path.as_posix()}\n" f"```yaml\n" f"{md_code}\n" "```\n\n"
    for link in links:
        output += link + "\n"
    return output


def add_config_as_page(config_path: Path):
    doc_path = config_path.with_suffix(".md")
    doc_source = get_doc_page_source(config_path)
    with mkdocs_gen_files.open(doc_path, "w") as f:
        print(f"Adding doc {doc_path} based on {config_path}")
        f.write(doc_source)

    nav_path = doc_path.with_suffix("").parts
    print(f"Adding nav element {nav_path} pointing to {doc_path}")

    nav[nav_path[1:]] = doc_path.relative_to("configs")  # mkdocs needs relative paths
    mkdocs_gen_files.set_edit_path(doc_path, Path("..") / config_path)


def add_summary_page(page_path: Path):
    doc_path = page_path.with_name("index.md")
    with mkdocs_gen_files.open(doc_path, "w") as f:
        print(f"Adding config summary based on {page_path}")
        with page_path.open("r") as source:
            shutil.copyfileobj(source, f)

    nav_path = doc_path.parts[:-1]
    print(f"Adding nav element {nav_path} pointing to {doc_path}")

    if len(nav_path) == 1:
        # Root config in configs.
        nav["configs"] = doc_path.relative_to("configs")  # mkdocs needs relative paths
    else:
        nav[nav_path[1:]] = doc_path.relative_to("configs")  # mkdocs needs relative paths
    mkdocs_gen_files.set_edit_path(doc_path, Path("..") / page_path)


# Explicity add general page here to ensure it is added to nav first and thus becomes the landing
# page for configuration.
#
add_summary_page(Path("configs/README.md"))
nav["training_config"] = "training_config.md"

for summary in sorted(Path("configs/cluster").rglob("README.md")):
    add_summary_page(summary)

for summary in sorted(Path("configs/dataset").rglob("README.md")):
    add_summary_page(summary)

for summary in sorted(Path("configs/experiment").rglob("README.md")):
    add_summary_page(summary)

nav["evaluation_config"] = "evaluation_config.md"
nav["evaluation_clustering_config"] = "evaluation_clustering_config.md"

for summary in sorted(Path("configs/evaluation").rglob("README.md")):
    add_summary_page(summary)

# The following files should not be included in the docs.
ignorelist = [
    "configs/experiment/slot_attention_vit/",
    "configs/experiment/slot_attention_vit_consistency",
]
for config in sorted(Path("configs").rglob("*.yaml")):
    if not file_in_repo(str(config)) or any(
        config.as_posix().startswith(prefix) for prefix in ignorelist
    ):
        print(f"Skipping {config}")
        continue
    add_config_as_page(config)


with mkdocs_gen_files.open("configs/SUMMARY.md", "w") as f:
    f.writelines(nav.build_literate_nav())
