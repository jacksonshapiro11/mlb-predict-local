#!/usr/bin/env python3

def update_readme():
    badge = "![CI](https://github.com/jacksonshapiro11/mlb-predict-local/actions/workflows/ci.yml/badge.svg)"
    
    try:
        with open("README.md", "r") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""
    
    # Split content into lines
    lines = content.split("\n")
    
    # If README is empty or first line isn't the badge, insert it
    if not lines or not lines[0].startswith("![CI]"):
        lines.insert(0, badge)
        if not lines[1:]:  # If only badge exists, add setup section
            lines.extend(["", "## Setup", ""])
    
    # Write back to README
    with open("README.md", "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    update_readme() 