#!/usr/bin/env python3
"""Kiểm tra một skill có hợp lệ theo Agent Skills spec không.

NGUỒN: chép từ `anthropics/skills`, script `skill-creator/scripts/quick_validate.py`.
Giữ nguyên luật gốc để khỏi lệch chuẩn; chỉ thêm ghi chú tiếng Việt và cờ
--allow-extra.

LƯU Ý VỀ TRƯỜNG NGOÀI CHUẨN. Một số bộ skill của bên thứ ba thêm trường riêng
(ví dụ skill Databricks dùng `parent:` để nhóm skill con). Chúng chạy được
trong môi trường của họ nhưng KHÔNG hợp lệ theo spec gốc. Dùng --allow-extra
để bỏ qua, thay vì đi xoá trường của người khác.

    python scripts/validate_skill.py ~/.claude/skills/my-skill
    python scripts/validate_skill.py --allow-extra parent ~/.claude/skills/foo
"""

import sys
import os
import re
import yaml
from pathlib import Path

def validate_skill(skill_path, allow_extra=()):
    """Kiểm tra một thư mục skill. Trả (hợp_lệ, thông_điệp)."""
    skill_path = Path(skill_path)

    # Check SKILL.md exists
    skill_md = skill_path / 'SKILL.md'
    if not skill_md.exists():
        return False, "SKILL.md not found"

    # Read and validate frontmatter
    content = skill_md.read_text()
    if not content.startswith('---'):
        return False, "No YAML frontmatter found"

    # Extract frontmatter
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return False, "Invalid frontmatter format"

    frontmatter_text = match.group(1)

    # Parse YAML frontmatter
    try:
        frontmatter = yaml.safe_load(frontmatter_text)
        if not isinstance(frontmatter, dict):
            return False, "Frontmatter must be a YAML dictionary"
    except yaml.YAMLError as e:
        return False, f"Invalid YAML in frontmatter: {e}"

    # Define allowed properties
    ALLOWED_PROPERTIES = {'name', 'description', 'license', 'allowed-tools', 'metadata', 'compatibility'}

    # Check for unexpected properties (excluding nested keys under metadata)
    unexpected_keys = set(frontmatter.keys()) - ALLOWED_PROPERTIES - set(allow_extra)
    if unexpected_keys:
        return False, (
            f"Unexpected key(s) in SKILL.md frontmatter: {', '.join(sorted(unexpected_keys))}. "
            f"Allowed properties are: {', '.join(sorted(ALLOWED_PROPERTIES))}"
        )

    # Check required fields
    if 'name' not in frontmatter:
        return False, "Missing 'name' in frontmatter"
    if 'description' not in frontmatter:
        return False, "Missing 'description' in frontmatter"

    # Extract name for validation
    name = frontmatter.get('name', '')
    if not isinstance(name, str):
        return False, f"Name must be a string, got {type(name).__name__}"
    name = name.strip()
    if name:
        # Check naming convention (kebab-case: lowercase with hyphens)
        if not re.match(r'^[a-z0-9-]+$', name):
            return False, f"Name '{name}' should be kebab-case (lowercase letters, digits, and hyphens only)"
        if name.startswith('-') or name.endswith('-') or '--' in name:
            return False, f"Name '{name}' cannot start/end with hyphen or contain consecutive hyphens"
        # Check name length (max 64 characters per spec)
        if len(name) > 64:
            return False, f"Name is too long ({len(name)} characters). Maximum is 64 characters."

    # Extract and validate description
    description = frontmatter.get('description', '')
    if not isinstance(description, str):
        return False, f"Description must be a string, got {type(description).__name__}"
    description = description.strip()
    if description:
        # Check for angle brackets
        if '<' in description or '>' in description:
            return False, "Description cannot contain angle brackets (< or >)"
        # Check description length (max 1024 characters per spec)
        if len(description) > 1024:
            return False, f"Description is too long ({len(description)} characters). Maximum is 1024 characters."

    # Validate compatibility field if present (optional)
    compatibility = frontmatter.get('compatibility', '')
    if compatibility:
        if not isinstance(compatibility, str):
            return False, f"Compatibility must be a string, got {type(compatibility).__name__}"
        if len(compatibility) > 500:
            return False, f"Compatibility is too long ({len(compatibility)} characters). Maximum is 500 characters."

    return True, "Skill is valid!"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("skill_dir", nargs="+", help="thư mục skill cần kiểm")
    ap.add_argument("--allow-extra", default="", metavar="a,b",
                    help="trường frontmatter ngoài chuẩn được chấp nhận")
    args = ap.parse_args()
    extra = [x.strip() for x in args.allow_extra.split(",") if x.strip()]

    bad = 0
    for d in args.skill_dir:
        ok, msg = validate_skill(d, extra)
        if not ok:
            bad += 1
            print(f"✗ {Path(d).name}: {msg}")
    total = len(args.skill_dir)
    print(f"{total - bad}/{total} skill hợp lệ")
    sys.exit(1 if bad else 0)